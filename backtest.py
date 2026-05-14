"""Stage 4 starter: event-driven taker backtest on top of the Stage 3 model.

Walk-forward predictions (same folds and embargo as model.py) drive a
positioning rule with a per-fold threshold. Trades cross the spread and
pay a taker fee. PnL is marked-to-market on microprice; markouts are
reported at multiple event horizons.

This is intentionally minimal — no queue model, no maker fills, no
latency. The point is a first end-to-end equity curve / cost picture
that lives on top of the IC signal, so the next iterations (queue model,
maker logic, slippage curves) have something to compare against.

Run:
    uv run python backtest.py BTCUSDT
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from book import load_book_events, load_trades, replay
from features import build_feature_frame
from evaluate import add_forward_returns
from model import (
    EMBARGO,
    HORIZON,
    N_SPLITS,
    TRAIN_FEATURES,
    train_one_fold,
    walk_forward_splits,
)


PLOTS_DIR = Path("plots")

# Bybit linear perp taker fee (post-Aug 2023): 5.5 bps each side.
TAKER_FEE_BPS = 5.5
# Trade only when |pred| exceeds this quantile of |pred| on the train fold.
# Higher quantile -> fewer, more confident trades.
THRESHOLD_QUANTILE = 0.70
MARKOUT_H = (10, 50, 200, 1000)


def predict_oos(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    embargo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward predict. Returns (preds, per-row entry threshold)."""
    preds = np.full(len(df), np.nan)
    thresholds = np.full(len(df), np.nan)
    for _fold_i, tr, te in walk_forward_splits(len(df), n_splits, embargo):
        model = train_one_fold(X[tr], y[tr], X[te], y[te], TRAIN_FEATURES)
        p_tr = model.predict(X[tr])
        preds[te] = model.predict(X[te])
        thresholds[te] = float(np.quantile(np.abs(p_tr), THRESHOLD_QUANTILE))
    return preds, thresholds


def simulate(
    df: pd.DataFrame,
    preds: np.ndarray,
    thresholds: np.ndarray,
    fee_bps: float = TAKER_FEE_BPS,
) -> pd.DataFrame:
    """Run the taker backtest in fractional units (PnL as fraction of $1 notional).

    PnL contribution per event is pos_{t-1} * microprice_return_t. Trade costs
    (half-spread + taker fee) are charged in the same fractional units at fill.
    """
    n = len(df)
    mp = df["microprice"].to_numpy()
    bid = df["bid_px_0"].to_numpy()
    ask = df["ask_px_0"].to_numpy()

    pos = np.zeros(n)
    pnl_step = np.zeros(n)
    fees_step = np.zeros(n)
    spread_step = np.zeros(n)

    current_pos = 0.0
    fee_frac = fee_bps / 1e4

    for i in range(n):
        if i > 0:
            mp_ret = (mp[i] - mp[i - 1]) / mp[i - 1]
            pnl_step[i] = current_pos * mp_ret

        if np.isfinite(preds[i]) and np.isfinite(thresholds[i]):
            if preds[i] > thresholds[i]:
                target = 1.0
            elif preds[i] < -thresholds[i]:
                target = -1.0
            else:
                target = 0.0
        else:
            target = 0.0

        delta = target - current_pos
        if delta != 0.0:
            half_spread_frac = (ask[i] - bid[i]) / 2.0 / mp[i]
            cost = abs(delta) * (half_spread_frac + fee_frac)
            pnl_step[i] -= cost
            fees_step[i] = abs(delta) * fee_frac
            spread_step[i] = abs(delta) * half_spread_frac
            current_pos = target

        pos[i] = current_pos

    equity = np.cumsum(pnl_step)
    equity_gross = np.cumsum(pnl_step + fees_step + spread_step)

    return pd.DataFrame(
        {
            "received_at_ns": df["received_at_ns"].to_numpy(),
            "microprice": mp,
            "bid": bid,
            "ask": ask,
            "pred": preds,
            "thr": thresholds,
            "pos": pos,
            "pnl_step": pnl_step,
            "equity": equity,
            "equity_gross": equity_gross,
            "fees": fees_step,
            "spread_cost": spread_step,
        }
    )


def summarize(bt_oos: pd.DataFrame) -> dict:
    """Aggregate run-level metrics. All PnL fields are in fraction of $1 notional."""
    pos = bt_oos["pos"].to_numpy()
    pnl_step = bt_oos["pnl_step"].to_numpy()
    ts_ns = bt_oos["received_at_ns"].to_numpy()

    pos_change = np.diff(pos, prepend=0.0)
    fill_idx = np.where(pos_change != 0.0)[0]
    n_fills = len(fill_idx)

    duration_sec = max((ts_ns[-1] - ts_ns[0]) / 1e9, 1.0)

    # Per-event Sharpe annualized — only meaningful with many hours of data.
    if pnl_step.std(ddof=0) > 0:
        events_per_year = len(pnl_step) / duration_sec * 86400.0 * 365.0
        sharpe = pnl_step.mean() / pnl_step.std(ddof=0) * np.sqrt(events_per_year)
    else:
        sharpe = 0.0

    return {
        "final_pnl_bps": float(bt_oos["equity"].iloc[-1]) * 1e4,
        "gross_pnl_bps": float(bt_oos["equity_gross"].iloc[-1]) * 1e4,
        "fees_total_bps": float(bt_oos["fees"].sum()) * 1e4,
        "spread_cost_total_bps": float(bt_oos["spread_cost"].sum()) * 1e4,
        "n_fills": int(n_fills),
        "duration_min": duration_sec / 60.0,
        "sharpe_annual_naive": float(sharpe),
        "long_frac": float((pos > 0).mean()),
        "short_frac": float((pos < 0).mean()),
        "flat_frac": float((pos == 0).mean()),
        "fill_idx": fill_idx,
        "fill_dir": np.sign(pos_change[fill_idx]),
    }


def markouts(
    bt_oos: pd.DataFrame,
    fill_idx: np.ndarray,
    fill_dir: np.ndarray,
    fee_bps: float = TAKER_FEE_BPS,
) -> pd.DataFrame:
    """Per-fill PnL at multiple horizons, in bps of entry price.

    `mean_bps` is the gross markout: sign * (mp_{t+h} - exec_px) / exec_px.
    `net_bps` subtracts a round-trip cost estimate (2 * fee + exit half-spread),
    using current half-spread as an unbiased proxy for the exit cost — a rough
    estimate, real exit happens later but for a starter this is fine.
    """
    if len(fill_idx) == 0:
        return pd.DataFrame()
    mp = bt_oos["microprice"].to_numpy()
    bid = bt_oos["bid"].to_numpy()
    ask = bt_oos["ask"].to_numpy()
    half_spread_frac = (ask - bid) / 2.0 / mp

    exec_px = np.where(fill_dir > 0, ask[fill_idx], bid[fill_idx])
    round_trip_cost_bps = (2 * fee_bps) + (2 * half_spread_frac[fill_idx] * 1e4)

    rows = []
    for h in MARKOUT_H:
        future_idx = np.minimum(fill_idx + h, len(mp) - 1)
        mk_bps = fill_dir * (mp[future_idx] - exec_px) / exec_px * 1e4
        rows.append(
            {
                "horizon": h,
                "mean_bps": float(np.mean(mk_bps)),
                "median_bps": float(np.median(mk_bps)),
                "net_bps": float(np.mean(mk_bps - round_trip_cost_bps)),
                "hit_rate": float((mk_bps > 0).mean()),
                "n": int(len(mk_bps)),
            }
        )
    return pd.DataFrame(rows)


def plot_backtest(symbol: str, bt_oos: pd.DataFrame, out_path: Path) -> None:
    t = (bt_oos["received_at_ns"] - bt_oos["received_at_ns"].iloc[0]) / 1e9 / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, bt_oos["equity_gross"] * 1e4, color="C2", lw=1.0, label="gross (no costs)")
    axes[0].plot(t, bt_oos["equity"] * 1e4, color="C0", lw=1.2, label="net (fees + half-spread)")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("Cumulative PnL (bps of notional)")
    axes[0].set_title(
        f"{symbol} — backtest (taker, fee={TAKER_FEE_BPS} bps, "
        f"|pred|>q{int(THRESHOLD_QUANTILE * 100)})"
    )
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, bt_oos["pos"], drawstyle="steps-post", color="C2")
    axes[1].set_ylabel("Position")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylim(-1.4, 1.4)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, bt_oos["microprice"], color="C3", lw=0.6)
    axes[2].set_ylabel("Microprice")
    axes[2].set_xlabel("OOS time (minutes)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "BTCUSDT"
    PLOTS_DIR.mkdir(exist_ok=True)

    print(f"loading {symbol} ...")
    events = load_book_events(symbol)
    trades = load_trades(symbol)
    states = list(replay(events, top_n=10))

    print("building features ...")
    df = build_feature_frame(states, trades, top_k=5)
    df = add_forward_returns(df, (HORIZON,))
    df = df.sort_values("received_at_ns").reset_index(drop=True)

    target = f"fwd_{HORIZON}"
    df = df.dropna(subset=TRAIN_FEATURES + [target]).reset_index(drop=True)
    print(f"  cleaned frame: {len(df):,} rows")

    X = df[TRAIN_FEATURES].to_numpy()
    y = df[target].to_numpy()

    print(f"\n=== walk-forward predict ({N_SPLITS} splits, embargo={EMBARGO}) ===")
    preds, thresholds = predict_oos(df, X, y, N_SPLITS, EMBARGO)
    oos_mask = np.isfinite(preds)
    print(f"  OOS rows: {oos_mask.sum():,} / {len(df):,}")

    print(f"\n=== simulate (taker, fee={TAKER_FEE_BPS} bps, q={THRESHOLD_QUANTILE}) ===")
    bt = simulate(df, preds, thresholds)
    bt_oos = bt[oos_mask].reset_index(drop=True)

    s = summarize(bt_oos)
    print(f"  duration         : {s['duration_min']:.1f} min")
    print(f"  fills            : {s['n_fills']:,}")
    print(f"  long/short/flat  : {s['long_frac']:.1%} / {s['short_frac']:.1%} / {s['flat_frac']:.1%}")
    print(f"  gross PnL        : {s['gross_pnl_bps']:+.2f} bps")
    print(f"  fees paid        : {s['fees_total_bps']:.2f} bps")
    print(f"  half-spread cost : {s['spread_cost_total_bps']:.2f} bps")
    print(f"  net PnL          : {s['final_pnl_bps']:+.2f} bps")
    print(f"  Sharpe (annual)  : {s['sharpe_annual_naive']:.2f}   <- naive, tiny sample")

    print("\n=== markouts (bps of entry price) ===")
    mk_df = markouts(bt_oos, s["fill_idx"], s["fill_dir"])
    pd.set_option("display.float_format", lambda x: f"{x:+.2f}")
    print(mk_df.to_string(index=False))

    out = PLOTS_DIR / f"{symbol.lower()}_backtest.png"
    plot_backtest(symbol, bt_oos, out)
    print(f"\nplot: {out}")


if __name__ == "__main__":
    main()
