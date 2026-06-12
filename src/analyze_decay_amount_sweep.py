from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METRIC_COLS = [
    "probe_start_val_loss",
    "loss_variance",
    "loss_oscillation",
    "loss_improvement_rate",
    "grad_norm",
    "grad_snr",
    "grad_weight_ratio",
    "grad_cosine_sim",
    "adam_v_norm",
    "weight_norm",
    "param_update_norm",
    "learning_rate",
    "final_lr_ratio",
    "log_final_lr_ratio",
    "lr_drop_fraction",
]


def read_jsonl(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        return pd.DataFrame(json.loads(line) for line in f if line.strip())


def latest_amount_sweep(log_root: Path) -> Path:
    candidates = sorted(
        p for p in log_root.iterdir() if (p / "decay_amount_sweep.jsonl").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No decay_amount_sweep.jsonl found under {log_root}")
    return candidates[-1]


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    y = df["probe_final_val_loss"].to_numpy()
    rows = []
    for col in [c for c in METRIC_COLS if c in df.columns]:
        x = df[col].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or np.allclose(x[mask], x[mask][0]) or np.allclose(y[mask], y[mask][0]):
            continue
        pearson_r, pearson_p = stats.pearsonr(x[mask], y[mask])
        spearman_r, spearman_p = stats.spearmanr(x[mask], y[mask])
        rows.append(
            {
                "metric": col,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a WSD decay amount sweep.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Run directory containing metrics.jsonl and decay_amount_sweep.jsonl.",
    )
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    args = parser.parse_args()

    run_dir = args.run_dir or latest_amount_sweep(args.log_root)
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    sweep = read_jsonl(run_dir / "decay_amount_sweep.jsonl")
    trajectory_path = run_dir / "decay_amount_trajectory.jsonl"
    trajectory = read_jsonl(trajectory_path) if trajectory_path.exists() else pd.DataFrame()

    val_rows = metrics[metrics["validation_loss"].notna()][["step", "validation_loss"]].copy()
    val_rows = val_rows.rename(
        columns={"step": "probe_start_step", "validation_loss": "probe_start_val_loss"}
    )
    sweep = sweep.merge(val_rows, on="probe_start_step", how="left")
    sweep["log_final_lr_ratio"] = np.log10(sweep["final_lr_ratio"])
    sweep["lr_drop_fraction"] = 1.0 - sweep["final_lr_ratio"]
    sweep["decay_val_improvement"] = (
        sweep["probe_start_val_loss"] - sweep["probe_final_val_loss"]
    )

    best = sweep.loc[sweep["probe_final_val_loss"].idxmin()]
    print(f"Run: {run_dir}")
    print("\nBest decay amount:")
    print(
        best[
            [
                "probe_start_step",
                "decay_length",
                "final_lr_ratio",
                "probe_start_val_loss",
                "probe_final_val_loss",
                "decay_val_improvement",
            ]
        ].to_string()
    )

    ranking_cols = [
        "probe_start_step",
        "decay_length",
        "final_lr_ratio",
        "probe_start_val_loss",
        "probe_final_val_loss",
        "decay_val_improvement",
    ]
    ranking = sweep[ranking_cols].sort_values("probe_final_val_loss")
    corr = correlation_table(sweep)

    print("\nRanking by final validation loss:")
    print(ranking.to_string(index=False))
    print("\nCorrelations with final validation loss:")
    if corr.empty:
        print("No non-constant metrics with at least 3 finite values.")
    else:
        print(corr.to_string(index=False))

    ranking.to_csv(run_dir / "decay_amount_ranking.csv", index=False)
    corr.to_csv(run_dir / "decay_amount_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        sweep["final_lr_ratio"],
        sweep["probe_final_val_loss"],
        marker="o",
        label="final val loss",
    )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("final LR ratio after 100-step decay")
    ax.set_ylabel("probe final validation loss")
    ax.set_title("Decay amount sweep")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "decay_amount_sweep.png", dpi=160)

    if not trajectory.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for final_lr_ratio, group in trajectory.groupby("final_lr_ratio"):
            group = group.sort_values("probe_decay_step")
            ax.plot(
                group["probe_decay_step"],
                group["probe_validation_loss"],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=f"{final_lr_ratio:g}",
            )
        ax.set_xlabel("decay probe step")
        ax.set_ylabel("validation loss")
        ax.set_title("Validation loss during 100-step decay probes")
        ax.grid(alpha=0.2)
        ax.legend(title="final LR ratio", ncols=2)
        plt.tight_layout()
        fig.savefig(run_dir / "decay_amount_trajectories.png", dpi=160)


if __name__ == "__main__":
    main()
