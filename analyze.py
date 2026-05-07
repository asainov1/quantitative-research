"""Stage 2 first-pass analysis: replay book, compute microprice and friends,
plot vs trades, dump summary stats. Run after collecting some data:

    uv run python collector.py   # let it run for a while
    uv run python analyze.py BTCUSDT
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd

from book import load_book_events, load_trades, replay


PLOTS_DIR = Path("plots")


def book_features(events: list[dict], top_n: int = 5) -> pd.DataFrame:
    """Replay book events into per-event feature rows."""
    rows = []
    for state in replay(events, top_n=top_n):
        if state.best_bid is None or state.best_ask is None:
            continue
        rows.append(
            {
                "event_time": state.event_time,
                "received_at_ns": state.received_at_ns,
                "best_bid_px": state.best_bid[0],
                "best_bid_qty": state.best_bid[1],
                "best_ask_px": state.best_ask[0],
                "best_ask_qty": state.best_ask[1],
                "mid": state.mid,
                "microprice": state.microprice,
                "spread": state.spread,
                "imbalance": state.imbalance,
            }
        )
    return pd.DataFrame(rows)


def summarize(symbol: str, book_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    print(f"\n=== {symbol} ===")
    print(f"book events: {len(book_df):,}")
    print(f"trades:      {len(trades_df):,}")
    if book_df.empty:
        return

    duration_s = (book_df["received_at_ns"].iloc[-1] - book_df["received_at_ns"].iloc[0]) / 1e9
    print(f"duration:    {duration_s:.1f}s ({duration_s / 60:.1f} min)")
    print(f"book rate:   {len(book_df) / duration_s:.1f}/s")
    if len(trades_df):
        print(f"trade rate:  {len(trades_df) / duration_s:.2f}/s")

    px_pct = book_df["mid"].mean()
    bps_per_unit = 1e4 / px_pct if px_pct else 1.0
    print(f"mid:         mean={px_pct:.4f}  range=[{book_df['mid'].min():.4f}, {book_df['mid'].max():.4f}]")
    print(f"spread:      mean={book_df['spread'].mean():.6f}  ({book_df['spread'].mean() * bps_per_unit:.2f} bps)")
    print(f"             p50={book_df['spread'].median():.6f}  p95={book_df['spread'].quantile(0.95):.6f}")
    print(f"imbalance:   mean={book_df['imbalance'].mean():+.3f}  std={book_df['imbalance'].std():.3f}")

    # microprice − mid: how much does microprice 'tilt' off the midpoint?
    book_df["mp_minus_mid"] = book_df["microprice"] - book_df["mid"]
    print(f"micro−mid:   mean={book_df['mp_minus_mid'].mean():+.6f}  std={book_df['mp_minus_mid'].std():.6f}")


def plot_microprice_vs_trades(
    symbol: str, book_df: pd.DataFrame, trades_df: pd.DataFrame, out: Path
) -> None:
    if book_df.empty:
        return
    book_t = (book_df["received_at_ns"] - book_df["received_at_ns"].iloc[0]) / 1e9
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. microprice + mid + trade ticks
    ax = axes[0]
    ax.plot(book_t, book_df["mid"], lw=0.6, alpha=0.6, label="mid", color="grey")
    ax.plot(book_t, book_df["microprice"], lw=0.5, alpha=0.9, label="microprice", color="C0")
    if len(trades_df):
        trade_t = (trades_df["received_at_ns"] - book_df["received_at_ns"].iloc[0]) / 1e9
        buys = trades_df["side"] == "Buy"
        ax.scatter(trade_t[buys], trades_df.loc[buys, "price"], s=4, color="green", alpha=0.5, label="taker buy")
        ax.scatter(trade_t[~buys], trades_df.loc[~buys, "price"], s=4, color="red", alpha=0.5, label="taker sell")
    ax.set_ylabel("price")
    ax.set_title(f"{symbol} — microprice vs mid vs trades")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # 2. spread
    ax = axes[1]
    ax.plot(book_t, book_df["spread"], lw=0.5, color="C2")
    ax.set_ylabel("spread")
    ax.set_title("bid-ask spread")
    ax.grid(alpha=0.3)

    # 3. imbalance
    ax = axes[2]
    ax.plot(book_t, book_df["imbalance"], lw=0.4, color="C3", alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("imbalance")
    ax.set_xlabel("seconds since start")
    ax.set_title("top-of-book volume imbalance  (bq − aq) / (bq + aq)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"plot: {out}")


def plot_imbalance_vs_microprice_change(
    symbol: str, book_df: pd.DataFrame, out: Path, horizon_events: int = 50
) -> None:
    """Quick sanity: does imbalance predict next-window microprice change?"""
    if len(book_df) < horizon_events * 5:
        return
    df = book_df.copy()
    df["future_mp"] = df["microprice"].shift(-horizon_events)
    df["mp_change"] = df["future_mp"] - df["microprice"]
    df = df.dropna(subset=["imbalance", "mp_change"])
    if df.empty:
        return

    # bucket imbalance into deciles, average forward microprice change per bucket
    df["bucket"] = pd.qcut(df["imbalance"], q=10, labels=False, duplicates="drop")
    grouped = df.groupby("bucket").agg(
        imb_mean=("imbalance", "mean"),
        mp_change_mean=("mp_change", "mean"),
        n=("mp_change", "size"),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["imb_mean"], grouped["mp_change_mean"], "o-")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("imbalance bucket mean")
    ax.set_ylabel(f"mean microprice change after {horizon_events} events")
    ax.set_title(f"{symbol} — does imbalance predict next-window move?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"plot: {out}")


def main() -> None:
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "BTCUSDT"
    PLOTS_DIR.mkdir(exist_ok=True)

    print(f"loading {symbol} ...")
    events = load_book_events(symbol)
    trades = load_trades(symbol)
    print(f"  raw events: {len(events):,}, raw trades: {len(trades):,}")

    print("replaying book ...")
    book_df = book_features(events)
    trades_df = pd.DataFrame(trades)

    summarize(symbol, book_df, trades_df)

    plot_microprice_vs_trades(symbol, book_df, trades_df, PLOTS_DIR / f"{symbol.lower()}_overview.png")
    plot_imbalance_vs_microprice_change(symbol, book_df, PLOTS_DIR / f"{symbol.lower()}_imbalance_predict.png")


if __name__ == "__main__":
    main()
