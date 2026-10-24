"""Matched E/N contrasts with crossed node/date dependence.

Inputs and previous analyses are read-only. The original four PTE measurements
and PTE/cycles are averaged within each condition x node x calendar-date cell.
Each matched cell then contributes one contrast, regardless of repeat count.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import beta
from statsmodels.stats.multitest import multipletests

from fulltrace_node_day import VarianceModel, load_recordings

SPECS = [
    ("core_power.throttle__mean_rate", "Mean throttle rate", "log_ratio", "% vs neutral"),
    ("core_power.throttle__variance", "Throttle variance", "log_ratio", "% vs neutral"),
    ("core_power.throttle__spectral_entropy", "Spectral entropy", "difference", "entropy units"),
    ("core_power.throttle__slope", "Throttle slope", "difference", "slope units"),
    ("core_power.throttle__per_cycle", "PTE / cycles", "log_ratio", "% vs neutral"),
]


def display_effect(values, scale):
    values = np.asarray(values)
    return 100 * np.expm1(values) if scale == "log_ratio" else values


def prepare(root, timezone="America/Chicago", features=None):
    specs = [s for s in SPECS if features is None or s[0] in features]
    if not specs or (features is not None and set(features) != {s[0] for s in specs}):
        raise ValueError("Select at least one known feature")
    traces, _, sources, excluded = load_recordings(root, timezone)
    for column in ["n_prompts", "n_predict", "inter_prompt_pause_s", "model_size"]:
        if traces[column].nunique() != 1:
            raise ValueError(f"Mixed collection settings: {column}")
    if traces.groupby(["node", "day"]).batch.nunique().gt(1).any():
        raise ValueError("A node/date spans batches; review collection settings before matching")
    counts = traces.groupby(["node", "day", "condition"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["emotional", "neutral"], fill_value=0)
    missing = counts.loc[(counts.emotional == 0) | (counts.neutral == 0)].reset_index()
    matched_keys = counts.loc[(counts.emotional > 0) & (counts.neutral > 0)].reset_index()
    blocks, contrasts = [], []
    for key in matched_keys.sort_values(["day", "node"]).itertuples(index=False):
        cell = traces.loc[(traces.node == key.node) & (traces.day == key.day)]
        e = cell.loc[cell.condition == "emotional"]
        n = cell.loc[cell.condition == "neutral"]
        block = dict(node=key.node, day=key.day, block=f"{key.day} / {key.node}",
                     batch=cell.batch.iloc[0], n_E=len(e), n_N=len(n),
                     runs_E=", ".join(e.run_label), runs_N=", ".join(n.run_label))
        blocks.append(block)
        for feature, label, scale, unit in specs:
            e_mean, n_mean = float(e[feature].mean()), float(n[feature].mean())
            if scale == "log_ratio" and min(e_mean, n_mean) <= 0:
                raise ValueError(f"Positive cell means required: {block['block']}: {feature}")
            delta = e_mean - n_mean
            contrast = np.log(e_mean / n_mean) if scale == "log_ratio" else delta
            contrasts.append(dict(**block, feature=feature, label=label, scale=scale, unit=unit,
                                  E_mean=e_mean, N_mean=n_mean, raw_difference=delta,
                                  percentage_difference=100 * delta / n_mean if n_mean > 0 else np.nan,
                                  contrast=contrast, effect=float(display_effect(contrast, scale))))
    return traces, pd.DataFrame(blocks), pd.DataFrame(contrasts), sources, excluded, missing


class ContrastModel(VarianceModel):
    """The parent supplies free-intercept ML; this adds the zero-intercept null.

    Both models retain the same crossed node/date random effects. Random
    contributions describe heterogeneity of E-N contrasts, not raw baselines.
    """

    def fit_null(self, y, initial=None):
        y = np.asarray(y, float)
        kernels = [self.kernels[key] for key in ("node", "day")]

        def objective(ratios, details=False):
            covariance = self.eye + sum(r * k for r, k in zip(ratios, kernels))
            cf = cho_factor(covariance, lower=True, check_finite=False)
            alpha = cho_solve(cf, y, check_finite=False)
            ss = y @ alpha
            if ss <= 0:
                raise ValueError("No residual variation under the null")
            nll = (2 * np.log(np.diag(cf[0])).sum() + self.n * np.log(ss / self.n)) / 2
            if details:
                variances = dict(residual=ss / self.n, node=ratios[0] * ss / self.n,
                                 day=ratios[1] * ss / self.n)
                return dict(nll=float(nll), mean=0.0, variances=variances,
                            ratios=dict(zip(("node", "day"), ratios)),
                            covariance=covariance, alpha=alpha)
            inverse = cho_solve(cf, self.eye, check_finite=False)
            gradient = np.array([.5 * (np.sum(inverse * k) - self.n * (alpha @ k @ alpha) / ss)
                                 for k in kernels])
            return nll, gradient

        starts = [np.zeros(2), np.ones(2), np.full(2, 10.0)]
        if initial is not None:
            starts.append(np.asarray(initial, float))
        candidates = []
        for start in starts:
            fitted = minimize(objective, start, jac=True, method="L-BFGS-B",
                              bounds=[(0, None)] * 2,
                              options=dict(ftol=1e-11, gtol=1e-7, maxiter=300))
            _, gradient = objective(fitted.x)
            projected = np.where((fitted.x <= 1e-10) & (gradient > 0), 0, gradient)
            if np.isfinite(fitted.fun) and (fitted.success or np.max(np.abs(projected)) < 1e-5):
                candidates.append(fitted)
        if not candidates:
            raise RuntimeError("Zero-intercept ML failed from multiple starts")
        best = min(candidates, key=lambda result: result.fun)
        return objective(best.x, details=True)


def mean_se(model, fitted):
    inverse_one = cho_solve(cho_factor(fitted["covariance"], lower=True), model.ones)
    return np.sqrt(fitted["variances"]["residual"] / inverse_one.sum())


def _analyze_feature(job):
    frame, seed, test_repeats, interval_repeats = job
    feature, label, scale, unit = frame.iloc[0][["feature", "label", "scale", "unit"]]
    values = frame.contrast.to_numpy(float)
    normalization = values.std(ddof=0)  # Do not subtract the mean: the null is zero.
    if normalization <= 0:
        raise ValueError(f"Constant contrasts require separate treatment: {feature}")
    y = values / normalization
    model = ContrastModel(frame)
    full = model.fit(y, thorough=True)
    null = model.fit_null(y, initial=list(full["ratios"].values()))
    if full["nll"] > null["nll"] + 1e-6:
        full = model.fit(y, initial=list(null["ratios"].values()), thorough=True)
    if full["nll"] > null["nll"] + 1e-6:
        raise RuntimeError("Observed models violate likelihood nesting")
    observed = max(0.0, 2 * (null["nll"] - full["nll"]))
    rng_null, rng_interval = [np.random.default_rng(s) for s in seed.spawn(2)]
    exceedances = 0
    for _ in range(test_repeats):
        simulated = model.simulate(null, rng_null)
        reduced = model.fit_null(simulated, initial=list(null["ratios"].values()))
        unrestricted = model.fit(simulated, initial=list(reduced["ratios"].values()), thorough=True)
        if unrestricted["nll"] > reduced["nll"] + 1e-6:
            raise RuntimeError("Bootstrap models violate likelihood nesting")
        statistic = max(0.0, 2 * (reduced["nll"] - unrestricted["nll"]))
        exceedances += statistic >= observed - 1e-10
    p = (exceedances + 1) / (test_repeats + 1)
    p_low = 0 if exceedances == 0 else beta.ppf(.025, exceedances, test_repeats - exceedances + 1)
    p_high = 1 if exceedances == test_repeats else beta.ppf(.975, exceedances + 1, test_repeats - exceedances)
    se = mean_se(model, full)
    t_boot, future = [], []
    bootstrap_means = []
    for _ in range(interval_repeats):
        simulated = model.simulate(full, rng_interval)
        fitted = model.fit(simulated, initial=list(full["ratios"].values()), thorough=True)
        standardized_error = (fitted["mean"] - full["mean"]) / mean_se(model, fitted)
        t_boot.append(standardized_error)
        bootstrap_means.append(fitted["mean"] * normalization)
        # Approximate predictive mixture for a new node AND new date, with a
        # matched-cell contrast measured under a comparable recording design.
        mean_draw = full["mean"] - standardized_error * se
        future.extend(mean_draw + np.sqrt(sum(fitted["variances"].values())) * rng_interval.normal(size=20))
    q_low, q_high = np.quantile(t_boot, [.025, .975])
    interval = normalization * np.array([full["mean"] - q_high * se, full["mean"] - q_low * se])
    prediction = normalization * np.quantile(future, [.025, .975])
    estimate = full["mean"] * normalization
    total_variance = sum(full["variances"].values())
    row = dict(feature=feature, label=label, scale=scale, unit=unit,
               estimate=estimate, ci_low=interval[0], ci_high=interval[1],
               prediction_low=prediction[0], prediction_high=prediction[1],
               mean_se=se * normalization, n_blocks=len(frame),
               n_positive=int((values > 0).sum()), n_negative=int((values < 0).sum()),
               n_zero=int((values == 0).sum()), lrt=observed, p=p,
               p_mc_low=p_low, p_mc_high=p_high, exceedances=exceedances,
               test_repeats=test_repeats, interval_repeats=interval_repeats,
               boundary=any(v < 1e-6 * total_variance for k, v in full["variances"].items() if k != "residual"))
    for key in ("estimate", "ci_low", "ci_high", "prediction_low", "prediction_high"):
        row["display_" + key] = float(display_effect(row[key], scale))
    for key, variance in full["variances"].items():
        row[key + "_fraction"] = variance / total_variance
    whitening = solve_triangular(np.linalg.cholesky(full["covariance"]),
                                 y - full["mean"], lower=True) / np.sqrt(full["variances"]["residual"])
    # The parent alpha already equals the conditional residual (shape covariance).
    offsets = sum(full["ratios"][key] * (model.kernels[key] @ full["alpha"]) for key in ("node", "day"))
    np.testing.assert_allclose(y - full["mean"] - offsets, full["alpha"], atol=1e-9)
    residuals = frame[["block", "node", "day", "feature"]].copy()
    residuals["whitened_residual"] = whitening
    residuals["conditional_residual_sd"] = full["alpha"] / np.sqrt(full["variances"]["residual"])
    omitted = []
    for grouping in ("day", "node"):
        for group in sorted(frame[grouping].unique()):
            keep = (frame[grouping] != group).to_numpy()
            submodel = ContrastModel(frame.loc[keep])
            fitted = submodel.fit(y[keep], thorough=True)
            value = fitted["mean"] * normalization
            omitted.append(dict(feature=feature, label=label, scale=scale, unit=unit,
                                omitted_type=grouping, omitted=str(group), n_remaining=int(keep.sum()),
                                estimate=value, display_estimate=float(display_effect(value, scale)),
                                same_sign=bool(np.sign(value) == np.sign(estimate))))
    return row, omitted, residuals, bootstrap_means


def analyze(contrasts, test_repeats=19999, interval_repeats=1999, seed=42, workers=4):
    seeds = np.random.SeedSequence(seed).spawn(len(SPECS))
    # Keep each feature's random stream unchanged when the test family is reduced.
    jobs = [(contrasts.loc[contrasts.feature == feature].reset_index(drop=True),
             seeds[i], test_repeats, interval_repeats) for i, (feature, *_rest) in enumerate(SPECS)
            if feature in set(contrasts.feature)]
    if not jobs or not set(contrasts.feature).issubset({s[0] for s in SPECS}):
        raise ValueError("Contrasts must contain known features")
    estimates, sensitivity, residuals, bootstraps = [], [], [], []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        iterator = executor.map(_analyze_feature, jobs) if executor else map(_analyze_feature, jobs)
        for i, (row, omitted, residual, means) in enumerate(iterator, 1):
            estimates.append(row)
            sensitivity.extend(omitted)
            residuals.append(residual)
            bootstraps.extend(dict(feature=row["feature"], replicate=j, mean=value)
                              for j, value in enumerate(means))
            print(f"Completed {i}/{len(jobs)}: {row['label']}", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    estimates = pd.DataFrame(estimates)
    estimates["p_bonferroni"] = multipletests(estimates.p, method="bonferroni")[1]
    for bound in ("low", "high"):
        estimates["p_bonferroni_mc_" + bound] = multipletests(estimates["p_mc_" + bound], method="bonferroni")[1]
    estimates["supported"] = estimates.p_bonferroni < .05
    estimates["mc_sensitive"] = (estimates.p_bonferroni_mc_low < .05) & (estimates.p_bonferroni_mc_high >= .05)
    estimates["prediction_crosses_zero"] = (estimates.prediction_low <= 0) & (estimates.prediction_high >= 0)
    return estimates, pd.DataFrame(sensitivity), pd.concat(residuals, ignore_index=True), pd.DataFrame(bootstraps)
