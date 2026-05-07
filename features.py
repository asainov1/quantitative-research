"""Microstructure features computed from reconstructed book + trades.

All features are causal — at time t they use only information available
strictly at or before t. This matters when you later join with future
returns for ML.

Inputs:
  book_states  — list of BookState (output of book.replay)
  trades       — list of trade dicts (output of book.load_trades)

Output:
  pandas.DataFrame indexed by received_at_ns with feature columns,
  joined onto the book event grid (one row per book event).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from book import BookState


def book_states_to_df(states: Iterable[BookState], top_k: int = 5) -> pd.DataFrame:
    """Flatten book states with top-K bid/ask prices and qtys."""
    rows = []
    for s in states:
        if s.best_bid is None or s.best_ask is None:
            continue
        row = {
            "event_time": s.event_time,
            "received_at_ns": s.received_at_ns,
            "mid": s.mid,
            "microprice": s.microprice,
            "spread": s.spread,
            "imbalance_top": s.imbalance,
        }
        for i in range(top_k):
            if i < len(s.bids):
                row[f"bid_px_{i}"] = s.bids[i][0]
                row[f"bid_qty_{i}"] = s.bids[i][1]
            else:
                row[f"bid_px_{i}"] = math.nan
                row[f"bid_qty_{i}"] = math.nan
            if i < len(s.asks):
                row[f"ask_px_{i}"] = s.asks[i][0]
                row[f"ask_qty_{i}"] = s.asks[i][1]
            else:
                row[f"ask_px_{i}"] = math.nan
                row[f"ask_qty_{i}"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_multilevel_imbalance(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Volume imbalance across top-K levels: (sum_bids - sum_asks) / total."""
    bid_cols = [f"bid_qty_{i}" for i in range(top_k)]
    ask_cols = [f"ask_qty_{i}" for i in range(top_k)]
    bsum = df[bid_cols].sum(axis=1)
    asum = df[ask_cols].sum(axis=1)
    total = bsum + asum
    df[f"imbalance_top{top_k}"] = (bsum - asum) / total.where(total > 0, np.nan)
    df[f"depth_top{top_k}"] = total
    return df


def add_ofi(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Per-level order-flow imbalance (Cont, Kukanov, Stoikov 2014).

    For each level, OFI per event:
      bid side: +Δqty if price unchanged or rose; -prev_qty if price fell
      ask side: +Δqty if price unchanged or fell; -prev_qty if price rose
    Then sum across top-K levels.

    Intuition: positive OFI = net liquidity entering on the bid side
    (or leaving on the ask side) = upward pressure.
    """

    def level_ofi(prev_px, prev_qty, px, qty, side: str) -> float:
        # px / qty are arrays per row, we work elementwise via numpy
        if side == "bid":
            up = px > prev_px
            same = px == prev_px
            down = px < prev_px
            return np.where(up, qty, np.where(same, qty - prev_qty, -prev_qty))
        else:  # ask
            up = px > prev_px
            same = px == prev_px
            down = px < prev_px
            return np.where(down, qty, np.where(same, qty - prev_qty, -prev_qty))

    total = np.zeros(len(df))
    for i in range(top_k):
        bp = df[f"bid_px_{i}"].to_numpy()
        bq = df[f"bid_qty_{i}"].to_numpy()
        ap = df[f"ask_px_{i}"].to_numpy()
        aq = df[f"ask_qty_{i}"].to_numpy()

        bp_prev = np.concatenate([[bp[0]], bp[:-1]])
        bq_prev = np.concatenate([[bq[0]], bq[:-1]])
        ap_prev = np.concatenate([[ap[0]], ap[:-1]])
        aq_prev = np.concatenate([[aq[0]], aq[:-1]])

        # bid contribution: positive when bid pressure builds
        bid_term = np.where(
            bp > bp_prev, bq,
            np.where(bp == bp_prev, bq - bq_prev, -bq_prev),
        )
        # ask contribution: subtract — ask pressure is bearish
        ask_term = np.where(
            ap < ap_prev, aq,
            np.where(ap == ap_prev, aq - aq_prev, -aq_prev),
        )
        total = total + bid_term - ask_term

    df[f"ofi_top{top_k}"] = total
    return df


def add_trade_flow(
    book_df: pd.DataFrame,
    trades: list[dict],
    windows_ms: tuple[int, ...] = (200, 1000, 5000),
) -> pd.DataFrame:
    """Signed taker volume rolled over time windows.

    For each book event timestamp, compute:
        sum(qty * sign) over trades in (t - W, t]
    where sign = +1 if taker is Buy, -1 if Sell.
    """
    if not trades:
        for w in windows_ms:
            book_df[f"trade_flow_{w}ms"] = 0.0
            book_df[f"trade_count_{w}ms"] = 0
        return book_df

    trades_df = pd.DataFrame(trades)
    trades_df["signed_qty"] = trades_df["qty"] * np.where(
        trades_df["side"] == "Buy", 1.0, -1.0
    )
    trades_df = trades_df.sort_values("received_at_ns").reset_index(drop=True)

    book_df = book_df.sort_values("received_at_ns").reset_index(drop=True)
    book_ts = book_df["received_at_ns"].to_numpy()
    trade_ts = trades_df["received_at_ns"].to_numpy()
    signed = trades_df["signed_qty"].to_numpy()

    for w_ms in windows_ms:
        w_ns = w_ms * 1_000_000
        # for each book event, find slice of trades in (t - w, t]
        right_idx = np.searchsorted(trade_ts, book_ts, side="right")
        left_idx = np.searchsorted(trade_ts, book_ts - w_ns, side="right")

        # cumulative sum trick for fast windowed sums
        cum_signed = np.concatenate([[0.0], np.cumsum(signed)])
        cum_count = np.arange(len(signed) + 1)

        flow = cum_signed[right_idx] - cum_signed[left_idx]
        count = cum_count[right_idx] - cum_count[left_idx]

        book_df[f"trade_flow_{w_ms}ms"] = flow
        book_df[f"trade_count_{w_ms}ms"] = count

    return book_df


def add_realized_vol(
    df: pd.DataFrame, windows_events: tuple[int, ...] = (100, 500, 2000)
) -> pd.DataFrame:
    """Rolling realized volatility of log-microprice returns over event windows."""
    df = df.sort_values("received_at_ns").reset_index(drop=True)
    log_mp = np.log(df["microprice"].to_numpy())
    log_ret = np.concatenate([[0.0], np.diff(log_mp)])
    df["log_ret"] = log_ret
    for w in windows_events:
        df[f"rv_{w}"] = (
            pd.Series(log_ret).pow(2).rolling(window=w, min_periods=w // 4).sum().pow(0.5)
        )
    return df


def add_spread_features(df: pd.DataFrame, tick: float | None = None) -> pd.DataFrame:
    """Spread regime indicators."""
    df["spread_bps"] = (df["spread"] / df["mid"]) * 1e4
    if tick is None:
        tick = df["spread"].quantile(0.5)  # rough proxy for tick size
    df["spread_in_ticks"] = df["spread"] / tick
    df["spread_wide"] = (df["spread_in_ticks"] > 1.5).astype(int)
    return df


def build_feature_frame(
    states: Iterable[BookState],
    trades: list[dict],
    top_k: int = 5,
) -> pd.DataFrame:
    df = book_states_to_df(states, top_k=top_k)
    df = add_multilevel_imbalance(df, top_k=top_k)
    df = add_ofi(df, top_k=top_k)
    df = add_trade_flow(df, trades)
    df = add_realized_vol(df)
    df = add_spread_features(df)
    return df
