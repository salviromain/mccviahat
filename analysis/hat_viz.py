#!/usr/bin/env python3
"""
hat_viz.py
----------
Two quick visualisations:

  1. RAW TIME SERIES  — loads perf_stat.csv + hat_interrupts.csv from one
                        emotional and one neutral trial directory and plots
                        the active Layer 1 indicators over time.

  2. PCA SCATTER      — loads an engineered feature CSV (output of
                        extract_features.py), drops NaN-heavy columns,
                        runs PCA to 2 components, and plots emotional vs
                        neutral trials in that space.

Usage:
    python hat_viz.py \
        --emotional_trial runs/emotional/p0000 \
        --neutral_trial   runs/neutral/p0001 \
        --features_csv    data/emotional7.csv data/neutral7.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_perf(trial_dir: Path) -> pd.DataFrame | None:
    p = trial_dir / "perf_stat.csv"
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    return None


def load_hat(trial_dir: Path) -> pd.DataFrame | None:
    p = trial_dir / "hat_interrupts.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(p)
    df["t_s"] = (df["timestamp_ns"] - df["timestamp_ns"].iloc[0]) / 1e9
    # difference cumulative counters
    irq_cols = [
        c for c in df.columns
        if c not in ("timestamp_ns", "t_s")
        and not c.endswith("_freq_khz")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in irq_cols:
        df[c] = df[c].diff().clip(lower=0)
    return df.iloc[1:].reset_index(drop=True)


def print_raw_samples(emotional_dir: Path, neutral_dir: Path, max_points: int) -> None:
    trials = [
        (emotional_dir, "emotional"),
        (neutral_dir, "neutral"),
    ]
    perf_indicators = ["tlb:tlb_flush", "core_power.throttle"]
    hat_indicators = ["TLB"]

    print(f"\nSample raw time-series points (up to {max_points} rows per signal):")
    for trial_dir, label in trials:
        print(f"\n[{label}] {trial_dir}")
        perf = load_perf(trial_dir)
        hat = load_hat(trial_dir)

        if perf is not None:
            t_perf = perf["t_s"] if "t_s" in perf.columns else pd.Series(np.arange(len(perf)))
            for indicator in perf_indicators:
                if indicator in perf.columns:
                    sample = pd.DataFrame({
                        "t_s": t_perf.values,
                        "value": perf[indicator].values,
                    }).head(max_points)
                    print(f"  perf::{indicator}")
                    print(sample.to_string(index=False))
                else:
                    print(f"  perf::{indicator} not found")
        else:
            print("  perf_stat.csv not found")

        if hat is not None:
            for indicator in hat_indicators:
                if indicator in hat.columns:
                    sample = pd.DataFrame({
                        "t_s": hat["t_s"].values,
                        "value": hat[indicator].values,
                    }).head(max_points)
                    print(f"  hat::{indicator}")
                    print(sample.to_string(index=False))
                else:
                    print(f"  hat::{indicator} not found")
        else:
            print("  hat_interrupts.csv not found")


# ── Plot 1: raw time series ───────────────────────────────────────────────────

def plot_raw(emotional_dir: Path, neutral_dir: Path, print_raw_points: int = 0) -> None:
    """
    Plot the two active Layer 1 indicators (TLB shootdowns and power throttle)
    for one emotional and one neutral trial side by side.
    """

    PERF_INDICATORS = ["tlb:tlb_flush", "core_power.throttle"]
    HAT_INDICATORS  = ["TLB"]   # column name inside hat_interrupts.csv

    if print_raw_points > 0:
        print_raw_samples(emotional_dir, neutral_dir, print_raw_points)

    fig, axes = plt.subplots(
        nrows=len(PERF_INDICATORS) + len(HAT_INDICATORS),
        ncols=2,
        figsize=(13, 3 * (len(PERF_INDICATORS) + len(HAT_INDICATORS))),
        sharey="row",
    )
    fig.suptitle("Layer 1 raw signals — one emotional vs one neutral trial",
                 fontsize=13, y=1.01)

    trials = [
        (emotional_dir, "Emotional", "tab:red"),
        (neutral_dir,   "Neutral",   "tab:blue"),
    ]

    for col_idx, (trial_dir, label, color) in enumerate(trials):
        perf = load_perf(trial_dir)
        hat  = load_hat(trial_dir)
        row  = 0

        # perf-based indicators
        for evt in PERF_INDICATORS:
            ax = axes[row][col_idx]
            if perf is not None and evt in perf.columns:
                t = perf["t_s"] if "t_s" in perf.columns else np.arange(len(perf))
                ax.plot(t, perf[evt].values, color=color, linewidth=0.7)
                ax.set_ylabel(evt, fontsize=8)
            else:
                ax.text(0.5, 0.5, "not found", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            ax.set_title(f"{label} — {evt}", fontsize=9)
            ax.set_xlabel("time (s)", fontsize=8)
            row += 1

        # /proc/interrupts-based indicators
        for irq in HAT_INDICATORS:
            ax = axes[row][col_idx]
            if hat is not None and irq in hat.columns:
                ax.plot(hat["t_s"].values, hat[irq].values,
                        color=color, linewidth=0.7)
                ax.set_ylabel(f"hat_{irq} (diff)", fontsize=8)
            else:
                ax.text(0.5, 0.5, "not found", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            ax.set_title(f"{label} — hat_{irq}", fontsize=9)
            ax.set_xlabel("time (s)", fontsize=8)
            row += 1

    plt.tight_layout()
    out = Path("hat_raw_timeseries.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ── Plot 2: PCA scatter ───────────────────────────────────────────────────────

def plot_pca(csv_paths: list[str], print_pca_points: int = 0) -> None:
    """
    Load one or more engineered feature CSVs, concatenate them,
    drop metadata columns, impute or drop NaN columns, scale,
    run PCA to 2 components, and scatter emotional vs neutral.
    """

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(df)} trials from {len(csv_paths)} CSV(s).")
    print(f"Condition counts:\n{df['condition'].value_counts()}\n")

    META_COLS = {"condition", "prompt_index", "elapsed_ms", "dur_s"}
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feature_cols].copy()

    # Drop columns that are entirely NaN (e.g. iat_cv for zero-count indicators)
    before = X.shape[1]
    X = X.dropna(axis=1, how="all")
    print(f"Dropped {before - X.shape[1]} all-NaN columns.")

    # For remaining NaNs, fill with column median
    X = X.fillna(X.median(numeric_only=True))

    # Drop any columns with zero variance (constant across all trials)
    X = X.loc[:, X.std() > 0]
    print(f"Using {X.shape[1]} features for PCA.")

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100
    print(f"PC1: {var_exp[0]:.1f}%   PC2: {var_exp[1]:.1f}%")

    if print_pca_points > 0:
        pca_points = pd.DataFrame({
            "condition": df["condition"].values,
            "prompt_index": df["prompt_index"].values if "prompt_index" in df.columns else np.nan,
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
        })
        print(f"\nSample PCA points (first {print_pca_points}):")
        print(pca_points.head(print_pca_points).to_string(index=False))

    labels    = df["condition"].values
    colors    = {"emotional": "tab:red", "neutral": "tab:blue"}
    fig, ax   = plt.subplots(figsize=(7, 6))

    for cond, color in colors.items():
        mask = labels == cond
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=cond, alpha=0.6, edgecolors="none", s=30,
        )

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=10)
    ax.set_title("PCA of engineered HAT features — emotional vs neutral", fontsize=11)
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.5)

    out = Path("hat_pca.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="HAT quick visualisation.")
    ap.add_argument("--emotional_trial", type=Path, default=None,
                    help="Path to one emotional trial directory (e.g. runs/emotional/p0000).")
    ap.add_argument("--neutral_trial", type=Path, default=None,
                    help="Path to one neutral trial directory.")
    ap.add_argument("--features_csv", nargs="+", default=None,
                    help="One or more engineered feature CSVs to plot in PCA space.")
    ap.add_argument("--print_raw_points", type=int, default=0,
                    help="Print up to N sample raw points per signal before plotting.")
    ap.add_argument("--print_pca_points", type=int, default=0,
                    help="Print first N PCA 2D points from engineered features.")
    args = ap.parse_args()

    if args.emotional_trial and args.neutral_trial:
        print("=== Plot 1: raw time series ===")
        plot_raw(args.emotional_trial, args.neutral_trial, print_raw_points=args.print_raw_points)
    else:
        print("Skipping raw plot (--emotional_trial and --neutral_trial not both provided).")

    if args.features_csv:
        print("\n=== Plot 2: PCA scatter ===")
        plot_pca(args.features_csv, print_pca_points=args.print_pca_points)
    else:
        print("Skipping PCA plot (--features_csv not provided).")


if __name__ == "__main__":
    main()
