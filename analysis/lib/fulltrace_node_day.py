"""Read-only inputs and crossed node/date variance models for the companion notebook.

Each condition is fitted separately: y = intercept + node + date + error.
The three contributions are independent Gaussian terms. Variance components
are fitted by maximum likelihood; tests simulate under the reduced model and
refit both models, including legitimate zero-variance boundary solutions.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import beta
from statsmodels.stats.multitest import multipletests

CONDITIONS = ("emotional", "neutral")
METRICS = ("mean_rate", "variance", "spectral_entropy", "slope")
RATIO = "core_power.throttle__per_cycle"


def load_recordings(root: Path, timezone: str, conditions=CONDITIONS):
    """Join individual feature rows to raw metadata; never use aggregate CSVs."""
    zone = ZoneInfo(timezone)
    raw_root = root / "data/raw/fulltrace"
    feature_root = root / "data/features/recomputed/fulltrace"
    records, sources, exclusions = [], [], []
    for meta_path in sorted(raw_root.glob("batch_*/*/trace_meta.json")):
        meta = json.loads(meta_path.read_text())
        batch, run = meta_path.parts[-3:-1]
        if meta.get("label") not in conditions or meta.get("model_size") != "70b":
            exclusions.append(dict(batch=batch, run_label=run,
                                   condition=meta.get("label"), reason="not 70B or other condition"))
            continue
        if meta.get("run_label") != run or not meta.get("all_ok"):
            raise ValueError(f"Unsuccessful trace or ambiguous run label: {meta_path}")
        if meta.get("mode") != "fulltrace":
            raise ValueError(f"Not a fulltrace recording: {meta_path}")
        paths = list((feature_root / batch).rglob(f"{run}_whole.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected one feature file for {batch}/{run}, found {paths}")
        feature_path = paths[0]
        frame = pd.read_csv(feature_path, dtype={"run_label": str, "condition": str})
        if len(frame) != 1:
            raise ValueError(f"Expected one whole-trace row: {feature_path}")
        row = frame.iloc[0].to_dict()
        if (row["run_label"] != run or row["condition"] != meta["label"]
                or row["mode"] != "whole_trace"):
            raise ValueError(f"Raw/feature identity mismatch: {feature_path}")
        for name in ("n_prompts", "total_ms"):
            expected = meta["n_prompts" if name == "n_prompts" else "total_trace_ms"]
            if not np.isclose(float(row[name]), float(expected), rtol=1e-8):
                raise ValueError(f"Raw/feature {name} mismatch: {feature_path}")
        match = re.match(r"^(\d+)(?=[_eEnN])", run)
        if not match:
            raise ValueError(f"Cannot determine node ID from folder name: {run}")
        start = pd.Timestamp(meta["t_trace_start_ns"], unit="ns", tz="UTC")
        end = pd.Timestamp(meta["t_trace_end_ns"], unit="ns", tz="UTC")
        row.update(
            batch=batch, node=match.group(1), day=start.tz_convert(zone).date().isoformat(),
            started_utc=start.isoformat(), started_local=start.tz_convert(zone).isoformat(),
            ended_local=end.tz_convert(zone).isoformat(),
            crosses_midnight=start.tz_convert(zone).date() != end.tz_convert(zone).date(),
            model_size=meta["model_size"], n_predict=meta["n_predict"],
            inter_prompt_pause_s=meta["inter_prompt_pause_s"],
            feature_file=str(feature_path.relative_to(root)),
            metadata_file=str(meta_path.relative_to(root)),
        )
        records.append(row)
        for path in (feature_path, meta_path):
            sources.append(dict(path=str(path.relative_to(root)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not records:
        raise ValueError("No eligible 70B E/N recordings found")
    df = pd.DataFrame(records).sort_values(["condition", "node", "day", "run_label"])
    df = df.reset_index(drop=True)
    if df.duplicated(["batch", "run_label"]).any():
        raise ValueError("Duplicate recording identities")
    if df.started_utc.duplicated().any():
        raise ValueError("Duplicate start timestamps: check for copied recordings")
    if set(df.condition) != set(conditions):
        raise ValueError(f"Recordings required for every condition: {', '.join(conditions)}")
    if (df["cycles__mean_rate"] <= 0).any():
        raise ValueError("Throttle-per-cycle ratio requires positive cycle rates")
    df[RATIO] = df["core_power.throttle__mean_rate"] / df["cycles__mean_rate"]
    indicators = sorted({c.rsplit("__", 1)[0] for c in df
                         if "__" in c and c.rsplit("__", 1)[1] in METRICS})
    features = [f"{indicator}__{metric}" for indicator in indicators for metric in METRICS]
    features.append(RATIO)
    for feature in features:
        if feature not in df or not np.isfinite(df[feature].to_numpy(float)).all():
            raise ValueError(f"Missing/nonfinite feature: {feature}; no rows are imputed or dropped")
    return df, features, pd.DataFrame(sources), pd.DataFrame(exclusions)


def model_values(values, feature):
    """Predeclared scale: log positive rates/variances/ratio; raw entropy/slope."""
    values = np.asarray(values, dtype=float)
    metric = feature.rsplit("__", 1)[1]
    logged = metric in ("mean_rate", "variance", "per_cycle")
    if logged and np.any(values <= 0):
        raise ValueError(f"Log scale requires strictly positive values: {feature}")
    return (np.log(values) if logged else values.copy()), ("log" if logged else "original")


class VarianceModel:
    """Profiled Gaussian ML with crossed random intercepts and analytic gradients.

    V = residual_variance * (I + r_node*K_node + r_day*K_day).
    The intercept and residual variance are profiled out. Optimization permits
    r=0, so a zero variance is a valid estimate, not an automatic fit failure.
    Standardizing y improves numerical conditioning without changing tests or
    variance fractions. Crossed grouping is essential; days are not nested in nodes.
    """

    def __init__(self, frame):
        self.n = len(frame)
        self.eye = np.eye(self.n)
        self.ones = np.ones(self.n)
        self.codes = {key: pd.factorize(frame[key], sort=True)[0] for key in ("node", "day")}
        self.kernels = {key: (codes[:, None] == codes[None, :]).astype(float)
                        for key, codes in self.codes.items()}

    def fit(self, y, effects=("node", "day"), initial=None, thorough=False):
        y = np.asarray(y, dtype=float)
        effects = tuple(effects)
        kernels = [self.kernels[key] for key in effects]
        rhs = np.column_stack((y, self.ones))

        def objective(ratios, details=False):
            covariance = self.eye.copy()
            for ratio, kernel in zip(ratios, kernels):
                covariance += ratio * kernel
            cf = cho_factor(covariance, lower=True, check_finite=False)
            solved = cho_solve(cf, rhs, check_finite=False)
            mean = solved[:, 0].sum() / solved[:, 1].sum()
            residual = y - mean
            alpha = solved[:, 0] - mean * solved[:, 1]
            ss = residual @ alpha
            if ss <= 0 or not np.isfinite(ss):
                raise ValueError("No positive residual variance; feature may be constant")
            # Constant n*(1+log(2*pi))/2 cancels in every likelihood comparison.
            nll = (2 * np.log(np.diag(cf[0])).sum() + self.n * np.log(ss / self.n)) / 2
            if details:
                variances = {"residual": ss / self.n, "node": 0.0, "day": 0.0}
                variances.update({key: ratio * ss / self.n for key, ratio in zip(effects, ratios)})
                return dict(nll=float(nll), mean=float(mean), variances=variances,
                            ratios=dict(zip(effects, ratios)), covariance=covariance,
                            alpha=alpha, effects=effects)
            inverse = cho_solve(cf, self.eye, check_finite=False)
            gradient = np.array([
                (np.sum(inverse * kernel) - self.n * (alpha @ kernel @ alpha) / ss) / 2
                for kernel in kernels
            ])
            return nll, gradient

        if not effects:
            return objective([], details=True)
        starts = [np.ones(len(effects)) if initial is None else np.asarray(initial, dtype=float)]
        if thorough:
            starts.extend([np.zeros(len(effects)), np.ones(len(effects)),
                           np.full(len(effects), 10.0)])
        unique_starts = []
        for start in starts:
            if not any(np.array_equal(start, other) for other in unique_starts):
                unique_starts.append(start)
        candidates = []
        for start in unique_starts:
            result = minimize(objective, start, jac=True, method="L-BFGS-B",
                              bounds=[(0, None)] * len(effects),
                              options=dict(ftol=1e-11, gtol=1e-7, maxiter=300))
            # A line search can stop at floating-point precision despite an
            # optimal boundary solution. Verify the constrained first-order
            # condition before accepting such a result; never ignore a real
            # failed fit or discard its bootstrap sample.
            _, gradient = objective(result.x)
            projected = np.where((result.x <= 1e-10) & (gradient > 0), 0, gradient)
            stationary = np.max(np.abs(projected)) < 1e-5
            if (result.success or stationary) and np.isfinite(result.fun):
                candidates.append(result)
        if not candidates and not thorough:
            return self.fit(y, effects, initial=initial, thorough=True)
        if not candidates:
            raise RuntimeError("Variance optimization failed after multiple starting values")
        best = min(candidates, key=lambda candidate: candidate.fun)
        fitted = objective(best.x, details=True)
        if np.linalg.cond(fitted["covariance"]) > 1e12:
            raise RuntimeError("Near-singular fitted observation covariance")
        return fitted

    def simulate(self, fit, rng):
        variances = fit["variances"]
        y = fit["mean"] + np.sqrt(variances["residual"]) * rng.normal(size=self.n)
        for key in ("node", "day"):
            codes = self.codes[key]
            effects = rng.normal(size=codes.max() + 1)
            y += np.sqrt(variances[key]) * effects[codes]
        return y


def fractions(fit):
    variances = fit["variances"]
    total = sum(variances.values())
    return np.array([variances[key] / total for key in ("node", "day", "residual")])


def bootstrap_test(model, y, full, effect, repeats, rng):
    """Test removing one variance component while retaining the other component."""
    remaining = tuple(key for key in ("node", "day") if key != effect)
    reduced = model.fit(y, remaining, thorough=True)
    observed = max(0.0, 2 * (reduced["nll"] - full["nll"]))
    if full["nll"] > reduced["nll"] + 1e-6:
        raise RuntimeError("Full model has worse likelihood than its nested reduced model")
    if observed < 1e-8:
        return dict(lrt=observed, p=1.0, p_mc_low=1.0, p_mc_high=1.0,
                    bootstrap_repeats=0, exceedances=0)
    exceedances = 0
    for _ in range(repeats):
        simulated = model.simulate(reduced, rng)
        null_fit = model.fit(simulated, remaining, initial=list(reduced["ratios"].values()), thorough=True)
        initial = [null_fit["ratios"].get(key, 0.0) for key in ("node", "day")]
        full_fit = model.fit(simulated, initial=initial, thorough=True)
        if full_fit["nll"] > null_fit["nll"] + 1e-6:
            raise RuntimeError("Bootstrap model nesting failed")
        statistic = max(0.0, 2 * (null_fit["nll"] - full_fit["nll"]))
        exceedances += statistic >= observed - 1e-10
    # Add-one correction prevents a p-value of zero. Clopper-Pearson bounds
    # describe Monte Carlo precision, not uncertainty in the effect estimate.
    lower = 0.0 if exceedances == 0 else beta.ppf(.025, exceedances, repeats-exceedances+1)
    upper = 1.0 if exceedances == repeats else beta.ppf(.975, exceedances+1, repeats-exceedances)
    return dict(lrt=observed, p=(exceedances + 1) / (repeats + 1),
                p_mc_low=lower, p_mc_high=upper,
                bootstrap_repeats=repeats, exceedances=exceedances)


def _analyze_feature(job):
    """One reproducibly seeded task; module-level for notebook-safe workers."""
    condition, frame, feature, seeds, test_repeats, interval_repeats = job
    rng_ci, rng_node, rng_day = [np.random.default_rng(seed) for seed in seeds]
    model = VarianceModel(frame)
    values, scale = model_values(frame[feature], feature)
    sd = values.std(ddof=0)
    if sd == 0:
        row = dict(condition=condition, feature=feature, scale=scale, status="constant")
        tests = [dict(condition=condition, feature=feature, effect=effect,
                      p=1.0, status="constant; not testable", bootstrap_repeats=0)
                 for effect in ("node", "day")]
        return row, tests, []
    y = (values - values.mean()) / sd
    full = model.fit(y, thorough=True)
    point = fractions(full)
    samples = np.empty((interval_repeats, 3))
    for b in range(interval_repeats):
        simulated = model.simulate(full, rng_ci)
        samples[b] = fractions(model.fit(
            simulated, initial=list(full["ratios"].values()), thorough=True))
    lower, upper = np.quantile(samples, [.025, .975], axis=0)
    row = dict(condition=condition, feature=feature, scale=scale, status="fitted",
               n_recordings=len(frame), n_nodes=frame.node.nunique(), n_days=frame.day.nunique())
    for i, effect in enumerate(("node", "day", "residual")):
        row[f"{effect}_fraction"] = point[i]
        row[f"{effect}_low"] = lower[i]
        row[f"{effect}_high"] = upper[i]
    row["boundary_estimate"] = bool(np.any(point[:2] < 1e-6))
    # alpha uses (I + r_node*K_node + r_day*K_day)^-1, so it already
    # equals y - fitted intercept - BLUP node/day offsets. Multiplying by
    # residual variance again would incorrectly shrink these diagnostics.
    conditional = full["alpha"]
    standardized = conditional / np.sqrt(full["variances"]["residual"])
    row["max_abs_residual_sd"] = float(np.max(np.abs(standardized)))
    residuals = [dict(condition=condition, feature=feature,
                      run_label=frame.run_label.iloc[i], residual_sd=value)
                 for i, value in enumerate(standardized)]
    tests = []
    for effect, rng in (("node", rng_node), ("day", rng_day)):
        result = bootstrap_test(model, y, full, effect, test_repeats, rng)
        tests.append(dict(condition=condition, feature=feature, effect=effect,
                          status="fitted", **result))
    return row, tests, residuals


def analyze(df, features, test_repeats=5999, interval_repeats=999, seed=42, workers=4):
    """Return estimates, all corrected tests, and conditional residual diagnostics.

    The family is every planned feature x condition x node/day hypothesis.
    Constant-within-condition features are explicitly retained with p=1 and no
    variance estimates. Bootstrap fit failures abort instead of being discarded.
    """
    seed_sequences = iter(np.random.SeedSequence(seed).spawn(len(CONDITIONS) * len(features) * 3))
    jobs = []
    for condition in CONDITIONS:
        frame = df.loc[df.condition == condition].reset_index(drop=True)
        for feature in features:
            seeds = [next(seed_sequences) for _ in range(3)]
            jobs.append((condition, frame, feature, seeds, test_repeats, interval_repeats))
    estimates, tests, residuals = [], [], []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = executor.map(_analyze_feature, jobs) if executor else map(_analyze_feature, jobs)
        for index, (row, feature_tests, feature_residuals) in enumerate(results, 1):
            estimates.append(row)
            tests.extend(feature_tests)
            residuals.extend(feature_residuals)
            print(f"{index}/{len(jobs)} — {row['condition']}: {row['feature']}", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    tests = pd.DataFrame(tests)
    tests["p_holm"] = multipletests(tests.p, method="holm")[1]
    tests["supported"] = tests.p_holm < .05
    tests["family_size"] = len(tests)
    # Sensitivity diagnostic using pointwise Monte Carlo bounds. These are
    # not simultaneous confidence bounds and are not a second set of p-values.
    for bound in ("low", "high"):
        tests[f"p_holm_mc_{bound}"] = multipletests(tests[f"p_mc_{bound}"].fillna(1), method="holm")[1]
    tests["mc_decision_uncertain"] = (tests.p_holm_mc_low < .05) & (tests.p_holm_mc_high >= .05)
    return pd.DataFrame(estimates), tests, pd.DataFrame(residuals)
