"""Parameter sweep over (horizon, threshold quantile) for the taker backtest.

Cheap sensitivity analysis on the two knobs that matter most:
- HORIZON     — how far ahead the model targets. Longer = more edge per fill,
                same per-fill cost.
- THRESHOLD_Q — quantile of |pred| on train above which we open a position.
                Higher = fewer, more confident trades.

We train one model per horizon (5 folds), then re-use those predictions to
simulate every threshold cheaply. Output: results table + heatmap of net PnL.

Run:
    uv run python sweep.py BTCUSDT
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
from model import N_SPLITS, TRAIN_FEATURES, train_one_fold, walk_forward_splits
from backtest import simulate, summarize, TAKER_FEE_BPS


PLOTS_DIR = Path("plots")

HORIZONS = (50, 200, 1000)
THRESHOLD_QUANTILES = (0.50, 0.70, 0.90, 0.95)


def train_and_predict_one_horizon(
    df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[int, np.ndarray, np.ndarray]]]:
    """For one horizon: build target, run walk-forward, return predictions.

    Returns (df_h, preds, fold_records) where fold_records is
    [(fold_i, train_idx, test_idx, train_preds, test_idx), ...] so the caller
    can compute different per-fold thresholds without retraining.
    """
    target = f"fwd_{horizon}"
    df_h = add_forward_returns(df.copy(), (horizon,))
    df_h = df_h.dropna(subset=TRAIN_FEATURES + [target]).reset_index(drop=True)

    X = df_h[TRAIN_FEATURES].to_numpy()
    y = df_h[target].to_numpy()

    preds = np.full(len(df_h), np.nan)
    fold_records: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_i, tr, te in walk_forward_splits(len(df_h), N_SPLITS, embargo=horizon):
        model = train_one_fold(X[tr], y[tr], X[te], y[te], TRAIN_FEATURES)
        train_pred = model.predict(X[tr])
        test_pred = model.predict(X[te])
        preds[te] = test_pred
        fold_records.append((fold_i, tr, te, train_pred))

    return df_h, preds, fold_records


def thresholds_from_quantile(
    n: int, fold_records: list, quantile: float
) -> np.ndarray:
    """Build per-row threshold by mapping each test row to its fold's train-q."""
    thr = np.full(n, np.nan)
    for _fold_i, _tr, te, train_pred in fold_records:
        thr[te] = float(np.quantile(np.abs(train_pred), quantile))
    return thr


def run_sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cache: dict[int, tuple[pd.DataFrame, np.ndarray, list]] = {}

    for h in HORIZONS:
        if h not in cache:
            print(f"  training horizon={h} ...")
            cache[h] = train_and_predict_one_horizon(df, h)
        df_h, preds, fold_records = cache[h]
        oos_mask = np.isfinite(preds)

        for q in THRESHOLD_QUANTILES:
            thr = thresholds_from_quantile(len(df_h), fold_records, q)
            bt = simulate(df_h, preds, thr)
            bt_oos = bt[oos_mask].reset_index(drop=True)
            s = summarize(bt_oos)
            rows.append(
                {
                    "horizon": h,
                    "threshold_q": q,
                    "n_oos": int(oos_mask.sum()),
                    "n_fills": s["n_fills"],
                    "flat_frac": s["flat_frac"],
                    "gross_pnl_bps": s["gross_pnl_bps"],
                    "fees_bps": s["fees_total_bps"],
                    "spread_bps": s["spread_cost_total_bps"],
                    "net_pnl_bps": s["final_pnl_bps"],
                }
            )
    return pd.DataFrame(rows)


def plot_heatmap(res: pd.DataFrame, symbol: str, out_path: Path) -> None:
    pivot_net = res.pivot(index="threshold_q", columns="horizon", values="net_pnl_bps")
    pivot_gross = res.pivot(index="threshold_q", columns="horizon", values="gross_pnl_bps")
    pivot_fills = res.pivot(index="threshold_q", columns="horizon", values="n_fills")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, data, title, cmap in (
        (axes[0], pivot_gross, "Gross PnL (bps)", "RdYlGn"),
        (axes[1], pivot_net, "Net PnL (bps)", "RdYlGn"),
        (axes[2], pivot_fills, "Number of fills", "viridis"),
    ):
        vmax = max(abs(data.values.min()), abs(data.values.max())) if "PnL" in title else None
        im = ax.imshow(
            data.values,
            aspect="auto",
            cmap=cmap,
            vmin=-vmax if vmax else None,
            vmax=vmax,
            origin="lower",
        )
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels([f"{q:.2f}" for q in data.index])
        ax.set_xlabel("horizon (events)")
        ax.set_ylabel("threshold quantile")
        ax.set_title(title)
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                v = data.values[i, j]
                txt = f"{v:+.0f}" if "PnL" in title else f"{int(v)}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{symbol} — taker backtest sweep (fee={TAKER_FEE_BPS} bps)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
    df = df.sort_values("received_at_ns").reset_index(drop=True)

    print(f"\n=== sweep: {len(HORIZONS)} horizons × {len(THRESHOLD_QUANTILES)} thresholds ===")
    res = run_sweep(df)

    pd.set_option("display.float_format", lambda x: f"{x:+.2f}")
    print("\n=== results ===")
    print(res.to_string(index=False))

    print("\n=== net PnL (bps) ===")
    print(res.pivot(index="threshold_q", columns="horizon", values="net_pnl_bps").to_string())

    print("\n=== gross PnL (bps) ===")
    print(res.pivot(index="threshold_q", columns="horizon", values="gross_pnl_bps").to_string())

    out = PLOTS_DIR / f"{symbol.lower()}_sweep.png"
    plot_heatmap(res, symbol, out)
    print(f"\nplot: {out}")


if __name__ == "__main__":
    main()
