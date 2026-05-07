"""Stage 3: LightGBM regression on microstructure features with
walk-forward CV and target-horizon embargo.

Predicts forward microprice change at horizon=H book events. Out-of-
sample evaluation is chronological — train always precedes test, with
an `embargo` of H samples on both sides of the boundary so a sample's
target window cannot overlap the validation period (Lopez de Prado).

Reports:
- per-fold Spearman IC (information coefficient): rank-correlation of
  prediction vs. realized forward return on the held-out fold.
- benchmark IC: same metric but using `imbalance_top` as the prediction
  (i.e. how much does the full model beat a single best feature?).
- RMSE vs the naive 'predict 0' baseline.
- mean feature importance across folds.

Run:
    uv run python model.py BTCUSDT
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from book import load_book_events, load_trades, replay
from features import build_feature_frame
from evaluate import add_forward_returns


PLOTS_DIR = Path("plots")
HORIZON = 50          # book events forward (~1.5s at typical rate)
N_SPLITS = 5
EMBARGO = HORIZON     # purge between train and test on each side

TRAIN_FEATURES = [
    "imbalance_top",
    "imbalance_top5",
    "depth_top5",
    "ofi_top5",
    "trade_flow_200ms",
    "trade_flow_1000ms",
    "trade_flow_5000ms",
    "trade_count_200ms",
    "trade_count_1000ms",
    "rv_100",
    "rv_500",
    "rv_2000",
    "spread_bps",
    "spread_in_ticks",
]


def walk_forward_splits(n: int, n_splits: int, embargo: int):
    """Yield (fold_idx, train_idx, test_idx) for chronological CV with embargo.

    Train indices are [0, train_end - embargo).
    Test indices are [train_end + embargo, fold_end).
    Train set grows monotonically with fold index.
    """
    fold_size = n // n_splits
    for i in range(1, n_splits):
        train_end = i * fold_size
        train_idx = np.arange(0, max(0, train_end - embargo))
        test_start = min(n, train_end + embargo)
        test_end = min(n, (i + 1) * fold_size)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) < 200 or len(test_idx) < 100:
            continue
        yield i, train_idx, test_idx


def train_one_fold(X_tr, y_tr, X_te, y_te, feature_names):
    params = {
        "objective": "regression_l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    valid_set = lgb.Dataset(X_te, label=y_te, feature_name=feature_names, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False), lgb.log_evaluation(0)],
    )
    return model


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
    print(f"  cleaned frame: {len(df):,} rows, {len(TRAIN_FEATURES)} features")

    X = df[TRAIN_FEATURES].to_numpy()
    y = df[target].to_numpy()
    bench = df["imbalance_top"].to_numpy()

    fold_results = []
    importances = []

    print(f"\n=== walk-forward CV: {N_SPLITS} splits, embargo={EMBARGO} ===")
    for fold_i, tr, te in walk_forward_splits(len(df), N_SPLITS, EMBARGO):
        model = train_one_fold(X[tr], y[tr], X[te], y[te], TRAIN_FEATURES)
        pred = model.predict(X[te])

        ic, _ = spearmanr(pred, y[te])
        ic_bench, _ = spearmanr(bench[te], y[te])
        rmse = float(np.sqrt(np.mean((pred - y[te]) ** 2)))
        rmse_naive = float(np.sqrt(np.mean(y[te] ** 2)))

        print(
            f"fold {fold_i}: train={len(tr):>6,}  test={len(te):>6,}  "
            f"IC={ic:+.3f}  bench IC={ic_bench:+.3f}  "
            f"rmse={rmse:.4f} naive={rmse_naive:.4f}  best_iter={model.best_iteration}"
        )

        fold_results.append(
            {
                "fold": fold_i,
                "n_train": len(tr),
                "n_test": len(te),
                "ic_model": ic,
                "ic_bench": ic_bench,
                "rmse": rmse,
                "rmse_naive": rmse_naive,
            }
        )
        importances.append(
            dict(zip(TRAIN_FEATURES, model.feature_importance(importance_type="gain")))
        )

    res_df = pd.DataFrame(fold_results)
    print("\n=== summary ===")
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")
    print(res_df.to_string(index=False))
    print(
        f"\nmean IC: {res_df['ic_model'].mean():+.3f} (LightGBM)  "
        f"vs  {res_df['ic_bench'].mean():+.3f} (imbalance_top alone)"
    )

    # feature importance plot
    imp = pd.DataFrame(importances).mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color="C0")
    ax.set_xlabel("mean gain importance across folds")
    ax.set_title(f"{symbol} — LightGBM feature importance (fwd_{HORIZON})")
    fig.tight_layout()
    out1 = PLOTS_DIR / f"{symbol.lower()}_feature_importance.png"
    fig.savefig(out1, dpi=120)
    plt.close(fig)

    # IC per fold
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(res_df))
    width = 0.38
    ax.bar(x - width / 2, res_df["ic_model"], width, label="LightGBM (all features)", color="C0")
    ax.bar(x + width / 2, res_df["ic_bench"], width, label="imbalance_top alone", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {f}" for f in res_df["fold"]])
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("out-of-sample Spearman IC")
    ax.set_title(f"{symbol} — walk-forward IC, horizon={HORIZON} events (~1.5s)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out2 = PLOTS_DIR / f"{symbol.lower()}_ic_per_fold.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)

    print(f"plots: {out1.name}, {out2.name}")


if __name__ == "__main__":
    main()
