"""Prompt-length control for full-trace throttle comparisons.

Every recording of a condition runs the same 20 prompts, so at the run level
input length is a constant per condition and cannot be adjusted statistically.
It can be controlled at the prompt level, where the 20 prompts of a condition
differ in length:

1. Regression adjustment: estimate how per-prompt throttle changes with input
   tokens *within* recordings (prompt-to-prompt variation of one condition),
   then shift each recording's value by slope x (its condition's mean tokens -
   neutral's mean tokens).
2. Length-matched subsets: recompute each recording's totals from a subset of
   prompts chosen so both conditions have the same mean length (no slope needed).

Per-prompt counter totals: data/features/recomputed/fulltrace_prompt_totals.csv
(scripts/run/extract_prompt_totals_fulltrace.py). Token counts: Llama 3.1 70B
tokenizer, BOS included: prompts/20base/token_counts_llama31_70b_all.csv.
"""
import numpy as np
import pandas as pd

RUN_KEYS = ["batch", "run_label", "condition", "node", "day"]


def load_prompt_frame(root):
    prompts = pd.read_csv(root / "data/features/recomputed/fulltrace_prompt_totals.csv")
    tokens = pd.read_csv(root / "prompts/20base/token_counts_llama31_70b_all.csv")
    frame = prompts.merge(tokens[["condition", "prompt_index", "n_tokens"]],
                          on=["condition", "prompt_index"], how="left", validate="m:1")
    if frame.n_tokens.isna().any():
        raise ValueError("Prompts without token counts")
    frame["node"] = frame.node.astype(str)
    frame["day"] = frame.day.astype(str)
    frame["run"] = frame.batch + "/" + frame.run_label
    frame["throttle_rate"] = frame["core_power.throttle"] / frame.elapsed_s
    return frame, tokens


def length_slopes(frame, outcomes):
    """Log-scale change per 100 input tokens, from prompt-to-prompt variation
    within recordings (run fixed effects), with and without a linear trend in
    prompt position. SEs clustered by prompt (condition x prompt index)."""
    import statsmodels.formula.api as smf
    d = frame.assign(t100=frame.n_tokens / 100,
                     cluster=pd.factorize(frame.condition + "/" + frame.prompt_index.astype(str))[0])
    rows = []
    for name, column in outcomes.items():
        d["y"] = np.log(d[column])
        for spec, formula, data in [("within recording", "y ~ C(run) + t100", d),
                                    ("within recording + position trend", "y ~ C(run) + t100 + prompt_index", d)] + [
                (f"{c} only (+ position trend)", "y ~ C(run) + t100 + prompt_index", d.loc[d.condition == c])
                for c in sorted(d.condition.unique())]:
            fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data.cluster})
            b, (lo, hi) = fit.params["t100"], fit.conf_int().loc["t100"]
            rows.append(dict(outcome=name, model=spec, slope_log=b, slope_low=lo, slope_high=hi,
                             pct_per_100=100 * np.expm1(b), pct_low=100 * np.expm1(lo),
                             pct_high=100 * np.expm1(hi), p=fit.pvalues["t100"], n_prompts=len(data)))
    return pd.DataFrame(rows)


def adjusted_runs(traces, column, slope_log, mean_tokens, reference="neutral"):
    """Recording values shifted to the reference condition's mean prompt length."""
    shift = traces.condition.map(mean_tokens) - mean_tokens[reference]
    if shift.isna().any():
        raise ValueError("Missing mean token count for a condition")
    return traces.assign(**{column + "__length_adj": traces[column] * np.exp(-slope_log * shift / 100)})


def matched_prompts(tokens, shorter, longer):
    """Drop the shortest prompts of `shorter` and the longest of `longer`
    (alternately, keeping the sets as large as possible) until the mean
    length of `shorter` is no longer below that of `longer`."""
    a = tokens.loc[tokens.condition == shorter].set_index("prompt_index").n_tokens.sort_values()
    b = tokens.loc[tokens.condition == longer].set_index("prompt_index").n_tokens.sort_values()
    keep_a, keep_b = list(a.index), list(b.index)
    while a.loc[keep_a].mean() < b.loc[keep_b].mean():
        if len(keep_a) >= len(keep_b):
            keep_a = keep_a[1:]
        else:
            keep_b = keep_b[:-1]
    return keep_a, keep_b, float(a.loc[keep_a].mean()), float(b.loc[keep_b].mean())


def subset_runs(frame, keep):
    """Recording totals over selected prompts: {condition: prompt indices}."""
    mask = [i in keep.get(c, ()) for c, i in zip(frame.condition, frame.prompt_index)]
    sums = frame.loc[mask].groupby(RUN_KEYS)[["core_power.throttle", "cycles", "elapsed_s"]].sum().reset_index()
    sums["per_cycle"] = sums["core_power.throttle"] / sums.cycles
    sums["mean_rate"] = sums["core_power.throttle"] / sums.elapsed_s
    return sums
