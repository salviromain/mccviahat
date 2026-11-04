"""Matched E/N cross pairs for the 70B isolated (per-prompt) trials.

Inputs are read-only: data/features/recomputed/isolated/70b/all_trials.csv and
the raw trial metadata it points to. Each session (`run`) ran one fixed
40-trial schedule with E and N trials interleaved, restarting the server before
every trial. Within each session every E trial is compared with every N trial,
as the fulltrace notebooks compare runs within a node/date.

The schedule is identical in every session, so the trial at a given position
uses the same prompt text each time. Cross pairs are therefore linked by shared
trials (within a session) and by shared prompt texts (across sessions), on top
of session and date terms. No node is recorded for these trials, so there is
no node term.
"""
import hashlib
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fulltrace_matched import SPECS, display_effect

FEATURES_FILE = "data/features/recomputed/isolated/70b/all_trials.csv"
RAW_ROOT = "data/raw/isolated/70b"
RATIO = "core_power.throttle__per_cycle"
EFFECTS = ("block", "day", "E_item", "N_item")


def load_trials(root, timezone="America/Chicago", features=None):
    """One row per trial with session, date and prompt identity. Returns (trials, sources)."""
    specs = [s for s in SPECS if features is None or s[0] in features]
    path = root / FEATURES_FILE
    trials = pd.read_csv(path).copy()
    trials[RATIO] = trials["core_power.throttle__mean_rate"] / trials["cycles__mean_rate"]
    zone = ZoneInfo(timezone)
    sources = [dict(path=FEATURES_FILE, sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
    starts = []
    for row in trials.itertuples(index=False):
        trial_dir = root / RAW_ROOT / row.source_trial.split("/", 2)[2]
        meta_path = trial_dir / "trial_meta.json"
        meta = json.loads(meta_path.read_text())
        if (meta["label"] != row.condition or meta["prompt_index"] != row.prompt_index
                or meta.get("model_size") != "70b" or not meta.get("ok")):
            raise ValueError(f"Feature/metadata mismatch or failed trial: {trial_dir}")
        starts.append(pd.Timestamp(meta["t_request_start_ns"], unit="ns", tz="UTC").tz_convert(zone))
        sources.append(dict(path=str(meta_path.relative_to(root)),
                            sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest()))
    trials = trials.assign(started_local=[s.isoformat() for s in starts],
                           day=[s.date().isoformat() for s in starts], session=trials["run"],
                           trial=trials.source_trial.str.rsplit("/", n=1).str[1])
    if trials.groupby("session").day.nunique().gt(1).any():
        raise ValueError("A session spans two dates")
    schedules = trials.groupby("session").apply(
        lambda s: tuple(zip(s.prompt_index, s.condition)), include_groups=False)
    if schedules.nunique() != 1:
        raise ValueError("Sessions ran different trial schedules; prompt identity is ambiguous")
    if trials.duplicated(["session", "prompt_index"]).any():
        raise ValueError("Duplicate trial positions within a session")
    for feature, label, scale, unit in specs:
        values = trials[feature].to_numpy(float)
        if not np.isfinite(values).all() or (scale == "log_ratio" and (values <= 0).any()):
            raise ValueError(f"{feature} must be finite (and positive for log scale)")
    keep = ["session", "day", "condition", "prompt_index", "trial", "source_trial",
            "started_local", "elapsed_ms", "dur_s"] + [s[0] for s in specs]
    return trials[keep].sort_values(["day", "session", "prompt_index"]).reset_index(drop=True), \
        pd.DataFrame(sources)


def prepare_trials(trials, specs, first="emotional", second="neutral"):
    """Every E trial against every N trial within each session.

    Pairs with the session's first E trial or first N trial span the session;
    the others follow exactly from them (model_point). Returns (blocks, contrasts)."""
    blocks, contrasts = [], []
    for (day, session), cell in trials.groupby(["day", "session"], sort=True):
        e = cell.loc[cell.condition == first].sort_values("prompt_index")
        n = cell.loc[cell.condition == second].sort_values("prompt_index")
        if e.empty or n.empty:
            raise ValueError(f"Session {session} lacks one condition")
        block = dict(day=day, node="not recorded", session=session, block=f"{day} / {session}",
                     batch="isolated_70b", n_E=len(e), n_N=len(n))
        blocks.append(block)
        for feature, label, scale, unit in specs:
            for i, er in enumerate(e.to_dict("records")):
                for j, nr in enumerate(n.to_dict("records")):
                    ev, nv = float(er[feature]), float(nr[feature])
                    contrast = np.log(ev / nv) if scale == "log_ratio" else ev - nv
                    contrasts.append(dict(**block, feature=feature, label=label, scale=scale, unit=unit,
                                          E_run=f"{session}/{er['trial']}", N_run=f"{session}/{nr['trial']}",
                                          E_item=f"E{er['prompt_index']}", N_item=f"N{nr['prompt_index']}",
                                          E_value=ev, N_value=nv, contrast=contrast,
                                          effect=float(display_effect(contrast, scale)),
                                          model_point=(i == 0 or j == 0)))
    return pd.DataFrame(blocks), pd.DataFrame(contrasts)
