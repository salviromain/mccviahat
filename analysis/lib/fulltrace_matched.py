"""Matched E/N cross-pair contrasts with shared-run and crossed node/date dependence.

Inputs and previous analyses are read-only. Within each node x calendar-date
cell, every E run is compared with every N run (E1-N1, E1-N2, E2-N1, E2-N2).
Cross pairs reuse runs, so the model links pairs that share an E run, an N
run, a node/date cell, a node or a date. Only non-redundant pairs enter the
likelihood: in a 2+2 cell, E2-N2 = E2-N1 + E1-N2 - E1-N1 exactly.
"""
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import beta
from statsmodels.stats.multitest import multipletests

from fulltrace_node_day import load_recordings

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


def run_number(run_label):
    """Repeat number within a node/date, e.g. 244_2e, 222E2 and 232_E_2 -> 2."""
    match = re.match(r"^\d+\D*?(\d+)", run_label)
    if not match:
        raise ValueError(f"Cannot determine run number: {run_label}")
    return int(match.group(1))


def prepare(root, timezone="America/Chicago", features=None, first="emotional", second="neutral",
            exclude=()):
    """Cross pairs of `first` vs `second` runs. Columns named E_*/N_* hold the
    first/second condition (emotional/neutral by default). `exclude` lists
    (node, day) cells to drop before matching; they are reported in `excluded`."""
    specs = [s for s in SPECS if features is None or s[0] in features]
    if not specs or (features is not None and set(features) != {s[0] for s in specs}):
        raise ValueError("Select at least one known feature")
    traces, _, sources, excluded = load_recordings(root, timezone, conditions=(first, second))
    dropped = traces[[(node, day) in set(exclude) for node, day in zip(traces.node, traces.day)]]
    traces = traces.drop(dropped.index).reset_index(drop=True)
    excluded = pd.concat([excluded, dropped[["batch", "run_label", "condition"]].assign(
        reason="excluded node/date")], ignore_index=True)
    sources = sources.loc[sources.path.isin(set(traces.feature_file) | set(traces.metadata_file))]
    for column in ["n_prompts", "n_predict", "inter_prompt_pause_s", "model_size"]:
        if traces[column].nunique() != 1:
            raise ValueError(f"Mixed collection settings: {column}")
    if traces.groupby(["node", "day"]).batch.nunique().gt(1).any():
        raise ValueError("A node/date spans batches; review collection settings before matching")
    counts = traces.groupby(["node", "day", "condition"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[first, second], fill_value=0)
    missing = counts.loc[(counts[first] == 0) | (counts[second] == 0)].reset_index()
    matched_keys = counts.loc[(counts[first] > 0) & (counts[second] > 0)].reset_index()
    blocks, contrasts = [], []
    for key in matched_keys.sort_values(["day", "node"]).itertuples(index=False):
        cell = traces.loc[(traces.node == key.node) & (traces.day == key.day)]
        e = cell.loc[cell.condition == first]
        n = cell.loc[cell.condition == second]
        e = e.assign(run=e.run_label.map(run_number)).sort_values("run")
        n = n.assign(run=n.run_label.map(run_number)).sort_values("run")
        block = dict(node=key.node, day=key.day, block=f"{key.day} / {key.node}",
                     batch=cell.batch.iloc[0], n_E=len(e), n_N=len(n),
                     runs_E=", ".join(e.run_label), runs_N=", ".join(n.run_label))
        blocks.append(block)
        for feature, label, scale, unit in specs:
            if scale == "log_ratio" and min(e[feature].min(), n[feature].min()) <= 0:
                raise ValueError(f"Positive run values required: {block['block']}: {feature}")
            # Every E run against every N run. Pairs with the first E run or
            # the first N run span the cell; the others follow exactly from them.
            for i, (e_label, e_value) in enumerate(zip(e.run_label, e[feature].astype(float))):
                for j, (n_label, n_value) in enumerate(zip(n.run_label, n[feature].astype(float))):
                    contrast = (np.log(e_value / n_value) if scale == "log_ratio"
                                else e_value - n_value)
                    contrasts.append(dict(**block, feature=feature, label=label, scale=scale,
                                          unit=unit, E_run=e_label, N_run=n_label,
                                          E_value=e_value, N_value=n_value, contrast=contrast,
                                          effect=float(display_effect(contrast, scale)),
                                          model_point=(i == 0 or j == 0)))
    return traces, pd.DataFrame(blocks), pd.DataFrame(contrasts), sources, excluded, missing


def block_summary(contrasts):
    """One descriptive row per node/date and feature: the mean of its cross pairs.

    On the log scale this equals mean log(E runs) - mean log(N runs).
    """
    grouped = contrasts.groupby(["block", "node", "day", "feature", "label", "scale", "unit"],
                                sort=False)
    summary = grouped.agg(n_pairs=("contrast", "size"), contrast=("contrast", "mean"),
                          E_mean=("E_value", lambda v: v.drop_duplicates().mean()),
                          N_mean=("N_value", lambda v: v.drop_duplicates().mean())).reset_index()
    summary["effect"] = [float(display_effect(c, s)) for c, s in zip(summary.contrast, summary.scale)]
    return summary


class CrossPairModel:
    """Profiled Gaussian ML for cross-pair contrasts.

    Cov(y) = s2 * (K_E + K_N + r_block K_block + r_node K_node + r_day K_day).
    K_E (K_N) links pairs sharing an E (N) run, so s2 is the run-to-run
    variance, assumed equal for E and N runs. Block, node and date terms are
    random contributions to the E-N gap. The intercept and s2 are profiled
    out; r=0 is a valid boundary estimate.
    """

    EFFECTS = ("block", "node", "day")

    def __init__(self, frame, effects=None):
        self.effects = tuple(effects or self.EFFECTS)
        self.n = len(frame)
        self.eye = np.eye(self.n)
        self.ones = np.ones(self.n)

        def same(values):
            values = np.asarray(values)
            return (values[:, None] == values[None, :]).astype(float)

        batch = frame.batch.astype(str) + "/"
        self.base = same(batch + frame.E_run) + same(batch + frame.N_run)
        self.kernels = {key: same(frame[key].astype(str)) for key in self.effects}

    def fit(self, y, initial=None, zero_mean=False):
        y = np.asarray(y, float)
        kernels = [self.kernels[key] for key in self.effects]
        rhs = np.column_stack((y, self.ones))

        def objective(ratios, details=False):
            shape = self.base + sum(r * k for r, k in zip(ratios, kernels))
            cf = cho_factor(shape, lower=True, check_finite=False)
            solved = cho_solve(cf, rhs, check_finite=False)
            mean = 0.0 if zero_mean else solved[:, 0].sum() / solved[:, 1].sum()
            alpha = solved[:, 0] - mean * solved[:, 1]
            ss = (y - mean) @ alpha
            if ss <= 0 or not np.isfinite(ss):
                raise ValueError("No positive residual variance; feature may be constant")
            nll = (2 * np.log(np.diag(cf[0])).sum() + self.n * np.log(ss / self.n)) / 2
            if details:
                variances = dict(residual=ss / self.n,
                                 **{key: r * ss / self.n for key, r in zip(self.effects, ratios)})
                return dict(nll=float(nll), mean=float(mean), variances=variances,
                            ratios=dict(zip(self.effects, ratios)), covariance=shape, alpha=alpha)
            inverse = cho_solve(cf, self.eye, check_finite=False)
            gradient = np.array([.5 * (np.sum(inverse * k) - self.n * (alpha @ k @ alpha) / ss)
                                 for k in kernels])
            return nll, gradient

        size = len(self.effects)
        starts = [np.zeros(size), np.ones(size), np.full(size, 10.0)]
        if initial is not None:
            starts.append(np.asarray(initial, float))
        candidates = []
        for start in starts:
            fitted = minimize(objective, start, jac=True, method="L-BFGS-B",
                              bounds=[(0, None)] * size,
                              options=dict(ftol=1e-11, gtol=1e-7, maxiter=300))
            # Accept boundary optima that stop at floating-point precision only
            # when the constrained first-order condition holds.
            _, gradient = objective(fitted.x)
            projected = np.where((fitted.x <= 1e-10) & (gradient > 0), 0, gradient)
            if np.isfinite(fitted.fun) and (fitted.success or np.max(np.abs(projected)) < 1e-5):
                candidates.append(fitted)
        if not candidates:
            raise RuntimeError("Variance optimization failed after multiple starting values")
        best = min(candidates, key=lambda result: result.fun)
        result = objective(best.x, details=True)
        if np.linalg.cond(result["covariance"]) > 1e12:
            raise RuntimeError("Near-singular fitted observation covariance")
        return result

    def fit_null(self, y, initial=None):
        return self.fit(y, initial=initial, zero_mean=True)

    def simulate(self, fit, rng):
        chol = np.linalg.cholesky(fit["covariance"])
        return fit["mean"] + np.sqrt(fit["variances"]["residual"]) * (chol @ rng.normal(size=self.n))


def pair_variance(variances):
    """Variance of one new cross pair: two runs plus every random term."""
    return 2 * variances["residual"] + sum(v for key, v in variances.items() if key != "residual")


def mean_se(model, fitted):
    inverse_one = cho_solve(cho_factor(fitted["covariance"], lower=True), model.ones)
    return np.sqrt(fitted["variances"]["residual"] / inverse_one.sum())


def _analyze_feature(job):
    frame, seed, test_repeats, interval_repeats, effects, omit = job
    feature, label, scale, unit = frame.iloc[0][["feature", "label", "scale", "unit"]]
    block_means = frame.groupby("block").contrast.mean()
    points = frame.loc[frame.model_point].reset_index(drop=True)
    values = points.contrast.to_numpy(float)
    normalization = values.std(ddof=0)  # Do not subtract the mean: the null is zero.
    if normalization <= 0:
        raise ValueError(f"Constant contrasts require separate treatment: {feature}")
    y = values / normalization
    ratios = lambda fitted: list(fitted["ratios"].values())
    model = CrossPairModel(points, effects)
    full = model.fit(y)
    null = model.fit_null(y, initial=ratios(full))
    if full["nll"] > null["nll"] + 1e-6:
        full = model.fit(y, initial=ratios(null))
    if full["nll"] > null["nll"] + 1e-6:
        raise RuntimeError("Observed models violate likelihood nesting")
    observed = max(0.0, 2 * (null["nll"] - full["nll"]))
    rng_null, rng_interval = [np.random.default_rng(s) for s in seed.spawn(2)]
    exceedances = 0
    for _ in range(test_repeats):
        simulated = model.simulate(null, rng_null)
        reduced = model.fit_null(simulated, initial=ratios(null))
        unrestricted = model.fit(simulated, initial=ratios(reduced))
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
        fitted = model.fit(simulated, initial=ratios(full))
        standardized_error = (fitted["mean"] - full["mean"]) / mean_se(model, fitted)
        t_boot.append(standardized_error)
        bootstrap_means.append(fitted["mean"] * normalization)
        # Approximate predictive mixture for one cross pair on a new node AND
        # new date, under a comparable recording design.
        mean_draw = full["mean"] - standardized_error * se
        future.extend(mean_draw + np.sqrt(pair_variance(fitted["variances"])) * rng_interval.normal(size=20))
    q_low, q_high = np.quantile(t_boot, [.025, .975])
    interval = normalization * np.array([full["mean"] - q_high * se, full["mean"] - q_low * se])
    prediction = normalization * np.quantile(future, [.025, .975])
    estimate = full["mean"] * normalization
    total_variance = pair_variance(full["variances"])
    row = dict(feature=feature, label=label, scale=scale, unit=unit,
               estimate=estimate, ci_low=interval[0], ci_high=interval[1],
               prediction_low=prediction[0], prediction_high=prediction[1],
               mean_se=se * normalization, n_blocks=len(block_means), n_pairs=len(frame),
               n_model_pairs=len(points),
               n_positive=int((block_means > 0).sum()), n_negative=int((block_means < 0).sum()),
               n_zero=int((block_means == 0).sum()),
               n_pairs_positive=int((frame.contrast > 0).sum()), lrt=observed, p=p,
               p_mc_low=p_low, p_mc_high=p_high, exceedances=exceedances,
               test_repeats=test_repeats, interval_repeats=interval_repeats,
               boundary=any(full["variances"][k] < 1e-6 * total_variance for k in model.effects))
    for key in ("estimate", "ci_low", "ci_high", "prediction_low", "prediction_high"):
        row["display_" + key] = float(display_effect(row[key], scale))
    row["residual_fraction"] = 2 * full["variances"]["residual"] / total_variance
    for key in model.effects:
        row[key + "_fraction"] = full["variances"][key] / total_variance
    whitening = solve_triangular(np.linalg.cholesky(full["covariance"]),
                                 y - full["mean"], lower=True) / np.sqrt(full["variances"]["residual"])
    residuals = points[["block", "node", "day", "feature", "E_run", "N_run"]].copy()
    residuals["whitened_residual"] = whitening
    # Run-to-run part of each pair's deviation (BLUP), in units of the run SD.
    residuals["run_residual_sd"] = (model.base @ full["alpha"]) / np.sqrt(full["variances"]["residual"])
    omitted = []
    for grouping in omit:
        groups = sorted(points[grouping].unique())
        if len(groups) < 2:  # omitting the only date/node would leave no data
            continue
        for group in groups:
            keep = (points[grouping] != group).to_numpy()
            fitted = CrossPairModel(points.loc[keep], effects).fit(y[keep])
            value = fitted["mean"] * normalization
            omitted.append(dict(feature=feature, label=label, scale=scale, unit=unit,
                                omitted_type=grouping, omitted=str(group),
                                n_remaining=int(points.loc[keep].block.nunique()),
                                estimate=value, display_estimate=float(display_effect(value, scale)),
                                same_sign=bool(np.sign(value) == np.sign(estimate))))
    return row, omitted, residuals, bootstrap_means


def analyze(contrasts, test_repeats=19999, interval_repeats=1999, seed=42, workers=4,
            effects=None, omit=("day", "node")):
    """`effects`: random terms on the gap (columns of `contrasts`), default
    CrossPairModel.EFFECTS. `omit`: groupings for the leave-one-group-out refits."""
    seeds = np.random.SeedSequence(seed).spawn(len(SPECS))
    # Keep each feature's random stream unchanged when the test family is reduced.
    jobs = [(contrasts.loc[contrasts.feature == feature].reset_index(drop=True),
             seeds[i], test_repeats, interval_repeats, effects, omit) for i, (feature, *_rest) in enumerate(SPECS)
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


# ---------------------------------------------------------------------------
# Probability of superiority (theta): non-parametric within-group comparison.
#
# theta(g) = share of the m(g) x k(g) first-vs-second run pairs in node/date g
# where the first-condition run is higher (ties count 1/2): the group's
# Mann-Whitney U / (m k), i.e. its ROC AUC. theta = mean over groups (each
# group weighted equally); theta_pair = sum U / sum m k (each pair weighted
# equally). 0.5 = the measure carries no information about the condition.
# ---------------------------------------------------------------------------
from itertools import combinations
from math import lcm


def _wins(first, second):
    first, second = np.asarray(first, float)[:, None], np.asarray(second, float)[None, :]
    return float((first > second).sum() + .5 * (first == second).sum())


def superiority_groups(traces, feature, first="emotional", second="neutral"):
    """One row per node/date with both conditions: m, k, wins (U) and theta."""
    rows = []
    for (node, day), cell in traces.groupby(["node", "day"], sort=True):
        a = cell.loc[cell.condition == first, feature].to_numpy(float)
        b = cell.loc[cell.condition == second, feature].to_numpy(float)
        if len(a) and len(b):
            rows.append(dict(node=node, day=day, block=f"{day} / {node}", batch=cell.batch.iloc[0],
                             feature=feature, m=len(a), k=len(b), pairs=len(a) * len(b),
                             wins=_wins(a, b), theta=_wins(a, b) / (len(a) * len(b)),
                             values=np.concatenate([a, b])))
    return pd.DataFrame(rows).sort_values(["day", "node"], key=lambda s: s.astype(str)).reset_index(drop=True)


def theta_estimates(groups):
    return float(groups.theta.mean()), float(groups.wins.sum() / groups.pairs.sum())


def theta_bootstrap(groups, cluster="block", repeats=20000, seed=42):
    """Percentile 95% intervals resampling whole clusters (block, node or day)
    with replacement; all groups of a drawn node/date are taken together."""
    rng = np.random.default_rng(seed)
    per_cluster = groups.groupby(groups[cluster].astype(str)).agg(
        theta_sum=("theta", "sum"), n=("theta", "size"), wins=("wins", "sum"), pairs=("pairs", "sum"))
    c = len(per_cluster)
    if c < 5:
        return dict(cluster=cluster, n_clusters=c, group_low=np.nan, group_high=np.nan,
                    pair_low=np.nan, pair_high=np.nan)
    draws = rng.integers(0, c, size=(repeats, c))
    take = lambda column: per_cluster[column].to_numpy(float)[draws].sum(axis=1)
    by_group = take("theta_sum") / take("n")
    by_pair = take("wins") / take("pairs")
    (gl, gh), (pl, ph) = np.quantile(by_group, [.025, .975]), np.quantile(by_pair, [.025, .975])
    return dict(cluster=cluster, n_clusters=c, group_low=gl, group_high=gh, pair_low=pl, pair_high=ph)


def _group_null(values, m):
    """Wins for every reassignment of m first-condition labels among a group's runs
    (all reassignments are equally likely under the null)."""
    idx = range(len(values))
    out = []
    for chosen in combinations(idx, m):
        rest = [i for i in idx if i not in chosen]
        out.append(_wins(values[list(chosen)], values[rest]))
    return np.array(out)


def theta_permutation(groups, alternative="greater", repeats=200000, seed=42, max_grid=5_000_000):
    """Stratified permutation test of theta = 0.5: labels are reassigned only
    within each node/date, keeping m and k fixed. Groups are independent under
    the null, so the exact null distribution of each statistic is the
    convolution of the per-group distributions (on an integer grid). Falls
    back to Monte Carlo when the grid would be too large.
    Returns p per group and per pair, and whether it is exact."""
    nulls = [_group_null(v, m) for v, m in zip(groups["values"], groups.m)]
    pairs = groups.pairs.to_numpy(int)
    observed = theta_estimates(groups)
    G, P = len(groups), pairs.sum()

    def tail(support, mass, obs):
        tol = 1e-9 * max(1.0, abs(obs))
        if alternative == "greater":
            hit = support >= obs - tol
        elif alternative == "less":
            hit = support <= obs + tol
        else:
            hit = np.abs(support - .5) >= abs(obs - .5) - tol
        return float(mass[hit].sum())

    # Integer scaling: per-group statistic = sum_g wins_g / pairs_g; wins are half-integers.
    scale_group = 2 * lcm(*map(int, pairs))
    results, exact = [], True
    for scale, weights, obs, divisor in [(scale_group, scale_group // pairs, observed[0], G * scale_group),
                                         (2, np.full(G, 2), observed[1], 2 * P)]:
        grid = int(sum(w * p for w, p in zip(weights, pairs))) + 1
        if grid > max_grid:
            exact = False
            break
        dist = np.array([1.0])
        for null, w in zip(nulls, weights):
            values = np.rint(null * w).astype(int)
            local = np.bincount(values, minlength=values.max() + 1) / len(values)
            dist = np.convolve(dist, local)
        support = np.arange(len(dist)) / divisor
        results.append(tail(support, dist, obs))
    if exact:
        return dict(p_group=results[0], p_pair=results[1], exact=True,
                    n_reassignments=float(np.prod([float(len(n)) for n in nulls])))
    rng = np.random.default_rng(seed)
    draws = np.column_stack([rng.choice(n, size=repeats) for n in nulls])
    by_group = (draws / pairs).mean(axis=1)
    by_pair = draws.sum(axis=1) / P
    p = [(1 + (tail(s, np.ones(repeats), o) )) / (repeats + 1) for s, o in [(by_group, observed[0]), (by_pair, observed[1])]]
    return dict(p_group=p[0], p_pair=p[1], exact=False, n_reassignments=float(repeats))


def theta_report(traces, feature, subsets, first="emotional", second="neutral",
                 alternative="greater", repeats=20000, seed=42):
    """theta, cluster-bootstrap intervals and stratified permutation p for each
    named subset. `subsets` maps a name to a boolean function of the group table."""
    groups = superiority_groups(traces, feature, first, second)
    rows = []
    for name, keep in subsets.items():
        sub = groups.loc[keep(groups)].reset_index(drop=True)
        if sub.empty:
            continue
        t_group, t_pair = theta_estimates(sub)
        boot = theta_bootstrap(sub, "block", repeats, seed)
        test = theta_permutation(sub, alternative, seed=seed)
        rows.append(dict(subset=name, feature=feature, n_groups=len(sub), n_pairs=int(sub.pairs.sum()),
                         n_nodes=sub.node.nunique(), n_dates=sub.day.nunique(),
                         theta_group=t_group, group_low=boot["group_low"], group_high=boot["group_high"],
                         theta_pair=t_pair, pair_low=boot["pair_low"], pair_high=boot["pair_high"],
                         p_group=test["p_group"], p_pair=test["p_pair"], exact=test["exact"],
                         groups_all_first=int((sub.theta == 1).sum()),
                         groups_all_second=int((sub.theta == 0).sum())))
    return groups.drop(columns="values"), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run order: does time of day (or position in a node's sequence) explain the
# condition differences on campaigns whose condition order was rotated?
# ---------------------------------------------------------------------------
def run_order_model(traces, feature, day, reference="neutral", late_hour=18.0):
    """OLS of log(feature) on node + condition (+ a time term) for one date.

    Uses the recorded start times (no slot assumption). Time terms:
    'none'; 'linear' (clock hours); 'late' (start at/after `late_hour` vs
    earlier); 'position' (k-th run on the node, as categories).
    Returns (time_effects, condition_effects): percentages = 100 (e^b - 1)."""
    import statsmodels.formula.api as smf
    d = traces.loc[traces.day.astype(str) == str(day)].copy()
    start = pd.to_datetime(d.started_local.str[:19])
    d["hours"] = start.dt.hour + start.dt.minute / 60 + start.dt.second / 3600
    d["late"] = (d.hours >= late_hour).astype(int)
    d["position"] = d.groupby("node").hours.rank(method="first").astype(int)
    d["y"] = np.log(d[feature].astype(float))
    d["node"] = d.node.astype(str)
    conditions = [reference] + sorted(set(d.condition) - {reference})
    d["condition"] = pd.Categorical(d.condition, categories=conditions)
    base = "y ~ C(node) + C(condition, Treatment('%s'))" % reference
    terms = {"none": "", "linear": " + hours", "late": " + late", "position": " + C(position)"}
    pct = lambda b: 100 * np.expm1(b)
    time_rows, cond_rows = [], []
    for name, extra in terms.items():
        fit = smf.ols(base + extra, data=d).fit()
        if name == "linear":
            b, (lo, hi) = fit.params["hours"], fit.conf_int().loc["hours"]
            time_rows.append(dict(day=day, feature=feature, term="per hour (linear)", effect=pct(b),
                                  ci_low=pct(lo), ci_high=pct(hi), p=fit.pvalues["hours"]))
        elif name == "late":
            b, (lo, hi) = fit.params["late"], fit.conf_int().loc["late"]
            time_rows.append(dict(day=day, feature=feature, term=f"late (≥{late_hour:g}:00) vs early",
                                  effect=pct(b), ci_low=pct(lo), ci_high=pct(hi), p=fit.pvalues["late"],
                                  n_late=int(d.late.sum())))
        elif name == "position":
            reduced = smf.ols(base, data=d).fit()
            f, p, df = fit.compare_f_test(reduced)
            time_rows.append(dict(day=day, feature=feature, term="position in sequence (F test)", df=int(df),
                                  effect=np.nan, ci_low=np.nan, ci_high=np.nan, p=p))
        names = {c: f"C(condition, Treatment('{reference}'))[T.{c}]" for c in conditions[1:]}
        contrasts = [(f"{c} vs {reference}", {names[c]: 1}) for c in conditions[1:]]
        others = conditions[1:]
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                contrasts.append((f"{others[i]} vs {others[j]}", {names[others[i]]: 1, names[others[j]]: -1}))
        for label, weights in contrasts:
            vector = np.zeros(len(fit.params))
            for key, w in weights.items():
                vector[list(fit.params.index).index(key)] = w
            test = fit.t_test(vector)
            b = float(np.squeeze(test.effect)); lo, hi = np.squeeze(test.conf_int())
            cond_rows.append(dict(day=day, feature=feature, time_term=name, contrast=label,
                                  effect=pct(b), ci_low=pct(lo), ci_high=pct(hi),
                                  p=float(np.squeeze(test.pvalue)), n_runs=len(d), df_resid=fit.df_resid))
    return pd.DataFrame(time_rows), pd.DataFrame(cond_rows)
