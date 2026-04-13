#!/usr/bin/env python3
"""
hat_viz.py
----------
Plots core_power.throttle raw time series for one emotional and one neutral trial.

Usage:
    python hat_viz.py
    python hat_viz.py --emotional_trial runs/70b/emotional7/p0000 --neutral_trial runs/70b/neutral7/p0000
"""

import argparse
from pathlib import Path

import numpy as np          
import pandas as pd 
import matplotlib.pyplot as plt  


# ── Defaults ──────────────────────────────────────────────────────────────────
REPO_ROOT         = Path(__file__).resolve().parent.parent.parent
DEFAULT_EMOTIONAL = REPO_ROOT / 'mccviahat'/'runs' / '70b' / 'emotional7' / 'p0000'
DEFAULT_NEUTRAL   = REPO_ROOT / 'mccviahat'/'runs' / '70b' / 'neutral7'   / 'p0005'


def load_perf(trial_dir: Path) -> pd.DataFrame | None:
    p = trial_dir / "perf_stat.csv"
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    return None


def plot_throttle(emotional_dir: Path, neutral_dir: Path) -> None:
    INDICATOR = "core_power.throttle"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    fig.suptitle("core_power.throttle — one emotional vs one neutral trial",
                 fontsize=14)

    trials = [
        (emotional_dir, "Emotional", "tab:red"),
        (neutral_dir,   "Neutral",   "tab:blue"),
    ]

    for ax, (trial_dir, label, color) in zip(axes, trials):
        perf = load_perf(trial_dir)
        if perf is not None and INDICATOR in perf.columns:
            t = perf["t_s"] if "t_s" in perf.columns else np.arange(len(perf))
            ax.plot(t, perf[INDICATOR].values, color=color, linewidth=0.7)
            ax.set_ylabel("Throttle events per 1 ms bucket", fontsize=13)
        else:
            ax.text(0.5, 0.5, f"not found in\n{trial_dir}", ha="center",
                    va="center", transform=ax.transAxes, color="grey", fontsize=8)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = Path("core_power_throttle_raw.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotional_trial", type=Path, default=DEFAULT_EMOTIONAL,
                    help=f"Path to emotional trial dir (default: {DEFAULT_EMOTIONAL})")
    ap.add_argument("--neutral_trial",   type=Path, default=DEFAULT_NEUTRAL,
                    help=f"Path to neutral trial dir (default: {DEFAULT_NEUTRAL})")
    args = ap.parse_args()

    print(f"Emotional : {args.emotional_trial}")
    print(f"Neutral   : {args.neutral_trial}")

    for d in [args.emotional_trial, args.neutral_trial]:
        if not d.exists():
            print(f"  WARNING: directory not found: {d}")

    plot_throttle(args.emotional_trial, args.neutral_trial)


if __name__ == "__main__":
    main()