#!/usr/bin/env python3
"""
run_prompts_isolated.py

Per-prompt isolated runner for HAT substrate analysis.

Timeouts and model-specific parameters are read automatically from
scripts/model/model_config.sh (written by model_fetch.sh).
All values can still be overridden via CLI flags.

For each prompt:
  1. Reset the LLM server (scripts/reset_server.sh)
  2. Drop hardware caches (via sysctl / drop_caches)
  3. Sleep a short stabilisation baseline
  4. Start substrate_collector.py (scoped to this prompt only)
  5. Send the prompt via curl
  6. SIGTERM the collector immediately after curl returns
  7. Write minimal metadata (prompt index, label, timing, ok)

Directory layout:
  runs/<label>/p<N>/
      perf_stat.txt
      perf_stat.csv
      hat_interrupts.csv
      proc_sample.csv
      kernel_log.txt
      collector_meta.json
      trial_meta.json
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Model config loader
# ─────────────────────────────────────────────────────────────────────────────

# Defaults matching 7b — used if model_config.sh is not found
_MODEL_DEFAULTS = {
    "MODEL_SIZE":       "7b",
    "REQUEST_TIMEOUT":  "600",
    "RESET_TIMEOUT":    "90",
    "BASELINE_S":       "2",
    "PROC_INTERVAL_S":  "0.1",
}

def load_model_config(config_path: str) -> dict:
    """
    Parse model_config.sh (KEY=value lines) and return a dict.
    Falls back to _MODEL_DEFAULTS if the file does not exist.
    """
    p = Path(config_path)
    if not p.exists():
        print(f"  ⚠  {config_path} not found — using built-in 7b defaults.")
        print("     Run: bash scripts/model/model_fetch.sh [7b|70b]  to configure.")
        return dict(_MODEL_DEFAULTS)

    cfg = dict(_MODEL_DEFAULTS)
    pattern = re.compile(r'^([A-Z_]+)=["\']?([^"\'#\n]*)["\']?')
    for line in p.read_text().splitlines():
        m = pattern.match(line.strip())
        if m:
            cfg[m.group(1)] = m.group(2).strip()

    print(f"  Loaded model config: {config_path}  (MODEL_SIZE={cfg.get('MODEL_SIZE')})")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sh(cmd: list[str], check: bool = True, timeout: float = 30.0) -> str:
    return subprocess.check_output(cmd, text=True, timeout=timeout).strip() if check else ""


def get_container_pid(container: str) -> int:
    raw = sh(["docker", "inspect", "-f", "{{.State.Pid}}", container])
    if not raw.isdigit() or int(raw) <= 0:
        raise SystemExit(
            f"Could not resolve host PID for container '{container}': got '{raw}'"
        )
    return int(raw)


def reset_server(reset_script: str, timeout: float = 90.0) -> None:
    result = subprocess.run(
        ["bash", reset_script],
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"reset_server.sh failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )


def drop_caches() -> None:
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            check=True, capture_output=True, timeout=10.0,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ⚠  drop_caches failed: {e.stderr.strip()}")


def start_collector(
    out_dir: Path,
    pid: int,
    duration_s: float,
    perf_interval_ms: int,
    proc_interval_s: float,
    collector_path: str,
    llm_cpus: str = "",
    perf_cpu: str = "",
) -> tuple[subprocess.Popen, object]:
    log_path = out_dir / "collector_stdout.log"
    log_f = open(log_path, "w", encoding="utf-8")
    cmd = [
        "python3", collector_path,
        "--out_dir", str(out_dir),
        "--pid", str(pid),
        "--duration_s", str(duration_s),
        "--perf_interval_ms", str(perf_interval_ms),
        "--proc_interval_s", str(proc_interval_s),
    ]
    if llm_cpus and perf_cpu:
        cmd += ["--llm_cpus", llm_cpus, "--perf_cpu", perf_cpu]
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def stop_collector(
    proc: subprocess.Popen, log_f, kill_timeout: float = 20.0
) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_f.close()


def send_prompt(
    prompt_text: str,
    n_predict: int,
    server_url: str,
    timeout_s: float,
) -> tuple[bool, str]:
    payload = json.dumps({
        "prompt": prompt_text,
        "n_predict": n_predict,
        "ignore_eos": True,
        "logit_bias": [[2, -100.0]],
    })
    try:
        out = subprocess.check_output(
            [
                "curl", "-sS",
                "--max-time", str(int(timeout_s)),
                server_url,
                "-H", "Content-Type: application/json",
                "-d", payload,
            ],
            text=True,
            timeout=timeout_s + 5.0,
        )
        return True, out
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    # ── Load model config first so we can use it as argument defaults ─────────
    # This script lives in scripts/run/
    # model_config.sh and reset_server.sh live in scripts/model/
    _script_dir  = Path(__file__).resolve().parent          # scripts/run/
    _model_dir   = _script_dir.parent / "model"             # scripts/model/
    _default_config = str(_model_dir / "model_config.sh")

    # Pre-parse just --model_config so we can load it before full argparse
    import sys
    _cfg_path = _default_config
    for i, a in enumerate(sys.argv[1:]):
        if a == "--model_config" and i + 1 < len(sys.argv) - 1:
            _cfg_path = sys.argv[i + 2]

    cfg = load_model_config(_cfg_path)

    # ── Argument parser ───────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(
        description="Per-prompt isolated HAT substrate runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--json", required=True,
                    help="Path to prompts JSON (list with 'instructions' fields).")
    ap.add_argument("--label", default=None,
                    help="Condition label ('emotional'/'neutral'). "
                         "If omitted, per-prompt 'condition' field is used.")
    ap.add_argument("--model_config",
                    default=_default_config,
                    help="Path to model_config.sh (auto-located by default).")
    ap.add_argument("--container", default="mccviahat-llama",
                    help="Docker container name.")
    ap.add_argument("--reset_script",
                    default=str(_model_dir / "reset_server.sh"),
                    help="Path to reset_server.sh.")
    ap.add_argument("--collector",
                    default=str(
                        _script_dir.parent.parent / "collectors" / "substrate_collector.py"
                    ),
                    help="Path to substrate_collector.py.")
    ap.add_argument("--server_url", default="http://localhost:8000/completion",
                    help="llama.cpp completion endpoint.")
    ap.add_argument("--n_predict", type=int, default=50,
                    help="Tokens to generate per prompt.")

    # ── Model-aware defaults (from model_config.sh) ───────────────────────────
    ap.add_argument("--baseline_s", type=float,
                    default=float(cfg["BASELINE_S"]),
                    help="Stabilisation sleep after reset (s).")
    ap.add_argument("--per_request_timeout_s", type=float,
                    default=float(cfg["REQUEST_TIMEOUT"]),
                    help="Hard timeout for each curl request (s).")
    ap.add_argument("--reset_timeout_s", type=float,
                    default=float(cfg["RESET_TIMEOUT"]),
                    help="Max time to wait for reset_server.sh (s).")
    ap.add_argument("--proc_interval_s", type=float,
                    default=float(cfg["PROC_INTERVAL_S"]),
                    help="/proc sampling interval for collector (s).")

    # ── Fixed perf / CPU settings ─────────────────────────────────────────────
    ap.add_argument("--perf_interval_ms", type=int, default=1,
                    help="perf stat bucket interval (ms).")
    ap.add_argument("--from", dest="start_index", type=int, default=0,
                    help="Prompt index to start from (0-based).")
    ap.add_argument("--n_prompts", type=int, default=None,
                    help="Number of prompts to run. Default: all remaining.")
    ap.add_argument("--llm_cpus", type=str, default="",
                    help="CPU cores for LLM container (enables taskset).")
    ap.add_argument("--perf_cpu", type=str, default="",
                    help="CPU core for perf stat (enables taskset).")
    args = ap.parse_args()

    # ── Print active config so the user can verify ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model              : {cfg['MODEL_SIZE']}")
    print(f"  baseline_s         : {args.baseline_s}s")
    print(f"  per_request_timeout: {args.per_request_timeout_s}s")
    print(f"  reset_timeout      : {args.reset_timeout_s}s")
    print(f"  proc_interval_s    : {args.proc_interval_s}s")
    print(f"{'='*60}\n")

    # ── Load prompts ──────────────────────────────────────────────────────────
    prompts: list[dict] = json.load(open(args.json, "r", encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise SystemExit("JSON must be a non-empty list of objects.")

    n_total = len(prompts)

    if args.label is None:
        missing = [i for i, p in enumerate(prompts) if not p.get("condition")]
        if missing:
            raise SystemExit(
                f"--label not provided and {len(missing)} prompt(s) lack a "
                f"'condition' field (indices: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}). "
                "Pass --label or tag prompts with mix_prompts.py first."
            )
        print(f"Loaded {n_total} prompts  (mixed-condition, per-prompt labels)")
    else:
        print(f"Loaded {n_total} prompts  label={args.label}")

    if args.start_index < 0 or args.start_index >= n_total:
        raise SystemExit(
            f"--from {args.start_index} out of range (0–{n_total - 1})"
        )

    prompts = prompts[args.start_index:]
    if args.n_prompts is not None:
        prompts = prompts[: args.n_prompts]
    print(f"Starting from index {args.start_index} → {len(prompts)} prompt(s) to run\n")

    # ── Per-prompt loop ───────────────────────────────────────────────────────
    for i, prompt_obj in enumerate(prompts):
        idx = args.start_index + i
        label = args.label if args.label is not None else prompt_obj["condition"]
        file_idx = prompt_obj.get("mixed_index", idx)

        instr = prompt_obj.get("instructions", "")
        if not isinstance(instr, str) or not instr.strip():
            raise SystemExit(f"Prompt {idx}: missing or empty 'instructions' field.")

        out_dir = Path("runs") / label / f"p{file_idx:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{idx+1}/{n_total}] prompt={idx}  label={label}  out={out_dir}",
            flush=True,
        )

        # 1. Reset server
        print("  → resetting server ...", flush=True)
        try:
            reset_server(args.reset_script, timeout=args.reset_timeout_s)
        except RuntimeError as e:
            print(f"  ✗ {e}")
            (out_dir / "trial_meta.json").write_text(
                json.dumps(
                    {"prompt_index": idx, "label": label, "ok": False,
                     "error": "reset_failed"},
                    indent=2,
                )
            )
            continue

        # 2. Drop hardware caches
        print("  → dropping caches ...", flush=True)
        drop_caches()

        # 3. Stabilisation baseline
        print(f"  → baseline sleep {args.baseline_s}s ...", flush=True)
        time.sleep(args.baseline_s)

        # 4. Resolve container PID (after reset)
        pid = get_container_pid(args.container)
        print(f"  Container '{args.container}' → host PID {pid}")

        # 5. Start collector (runs for the full request timeout + tail)
        collector_duration_s = args.per_request_timeout_s + 5.0
        print("  → starting collector ...", flush=True)
        collector_proc, collector_log_f = start_collector(
            out_dir=out_dir,
            pid=pid,
            duration_s=collector_duration_s,
            perf_interval_ms=args.perf_interval_ms,
            proc_interval_s=args.proc_interval_s,
            collector_path=args.collector,
            llm_cpus=args.llm_cpus,
            perf_cpu=args.perf_cpu,
        )

        # 6. Send prompt
        print("  → sending prompt ...", flush=True)
        t_req_start = time.time_ns()
        ok, response_raw = send_prompt(
            prompt_text=instr,
            n_predict=args.n_predict,
            server_url=args.server_url,
            timeout_s=args.per_request_timeout_s,
        )
        t_req_end = time.time_ns()
        elapsed_ms = (t_req_end - t_req_start) / 1e6
        print(f"  → done  ok={ok}  elapsed={elapsed_ms:.0f}ms", flush=True)

        # 7. Stop collector immediately after curl returns
        print("  → stopping collector ...", flush=True)
        stop_collector(collector_proc, collector_log_f)

        # 8. Write trial metadata
        trial_meta = {
            "prompt_index":    idx,
            "label":           label,
            "model_size":      cfg["MODEL_SIZE"],
            "ok":              ok,
            "t_request_start_ns": t_req_start,
            "t_request_end_ns":   t_req_end,
            "elapsed_ms":      elapsed_ms,
        }
        (out_dir / "trial_meta.json").write_text(
            json.dumps(trial_meta, indent=2), encoding="utf-8"
        )

        if not ok:
            (out_dir / "response_error.txt").write_text(
                response_raw, encoding="utf-8"
            )

    dest = args.label if args.label else "<per-prompt-condition>"
    print(f"\n✓ Done. Results in runs/{dest}/")


if __name__ == "__main__":
    run()
