"""Rank features by predictive power vs forward microprice change.

For each candidate feature x_t we compute Spearman correlation with
the forward microprice change   r_t^h = mp_{t+h} - mp_t   over a fixed
horizon h (in book events). Higher |corr| = more predictive feature.

Then for the top-K we draw decile-bucket plots: x bucket mean → mean
forward return. Monotone, S-shaped, or weakly-monotone curves are the
shape we want to see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from book import load_book_events, load_trades, replay
from features import build_feature_frame


PLOTS_DIR = Path("plots")

CANDIDATE_FEATURES = [
    "imbalance_top",
    "imbalance_top5",
    "depth_top5",
    "ofi_top5",
    "trade_flow_200ms",
    "trade_flow_1000ms",
    "trade_flow_5000ms",
    "rv_100",
    "rv_500",
    "rv_2000",
    "spread_bps",
    "spread_in_ticks",
    "spread_wide",
]


def add_forward_returns(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    df = df.sort_values("received_at_ns").reset_index(drop=True)
    mp = df["microprice"].to_numpy()
    for h in horizons:
        future = np.concatenate([mp[h:], np.full(h, np.nan)])
        df[f"fwd_{h}"] = future - mp
    return df


def rank_features(df: pd.DataFrame, features: list[str], horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        for h in horizons:
            target = f"fwd_{h}"
            mask = df[[f, target]].dropna().index
            if len(mask) < 100:
                continue
            x = df.loc[mask, f].to_numpy()
            y = df.loc[mask, target].to_numpy()
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            rho, _ = spearmanr(x, y)
            rows.append({"feature": f, "horizon": h, "spearman": rho, "n": len(mask)})
    return pd.DataFrame(rows).sort_values(["horizon", "spearman"], key=abs, ascending=[True, False])


def bucket_plot(
    df: pd.DataFrame, feature: str, target: str, ax: plt.Axes, n_buckets: int = 10
) -> None:
    sub = df[[feature, target]].dropna()
    if len(sub) < n_buckets * 5:
        return
    try:
        sub["bucket"] = pd.qcut(sub[feature], q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return
    g = sub.groupby("bucket").agg(
        x=(feature, "mean"),
        y=(target, "mean"),
        n=(target, "size"),
    )
    ax.plot(g["x"], g["y"], "o-")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    rho, _ = spearmanr(sub[feature], sub[target])
    ax.set_title(f"{feature} vs {target}  (ρ={rho:+.3f}, n={len(sub):,})", fontsize=9)
    ax.grid(alpha=0.3)


def main() -> None:
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "BTCUSDT"
    horizons = (10, 50, 200, 1000)  # in book events

    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"loading {symbol} ...")
    events = load_book_events(symbol)
    trades = load_trades(symbol)
    states = list(replay(events, top_n=10))
    print(f"  states: {len(states):,}, trades: {len(trades):,}")

    print("building features ...")
    df = build_feature_frame(states, trades, top_k=5)
    df = add_forward_returns(df, horizons)
    print(f"  feature frame: {df.shape}")

    print("\n=== feature ranking by Spearman vs forward microprice change ===")
    ranking = rank_features(df, CANDIDATE_FEATURES, horizons)
    pivot = ranking.pivot(index="feature", columns="horizon", values="spearman")
    pivot = pivot.reindex(CANDIDATE_FEATURES)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print(pivot.to_string())

    # plot grid: one subplot per feature, at horizon=50
    target = "fwd_50"
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, feat in enumerate(CANDIDATE_FEATURES):
        if i >= len(axes):
            break
        bucket_plot(df, feat, target, axes[i])
    for j in range(len(CANDIDATE_FEATURES), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{symbol} — features vs {target} (decile buckets)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOTS_DIR / f"{symbol.lower()}_feature_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nplot: {out}")


if __name__ == "__main__":
    main()
