#!/usr/bin/env python3
"""
run_prompts_isolated.py

Per-prompt isolated runner for HAT substrate analysis.

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
      trial_meta.json   ← only metadata we actually need
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sh(cmd: list[str], check: bool = True, timeout: float = 30.0) -> str:
    """Run a command, return stdout stripped."""
    return subprocess.check_output(cmd, text=True, timeout=timeout).strip() if check else ""


def get_container_pid(container: str) -> int:
    """Resolve the host-side PID of a running Docker container."""
    raw = sh(["docker", "inspect", "-f", "{{.State.Pid}}", container])
    if not raw.isdigit() or int(raw) <= 0:
        raise SystemExit(f"Could not resolve host PID for container '{container}': got '{raw}'")
    return int(raw)


def reset_server(reset_script: str, timeout: float = 60.0) -> None:
    """Call the server reset script and wait for it to finish."""
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
    """
    Drop the kernel page cache, dentries, and inodes.
    Requires passwordless sudo for 'tee /proc/sys/vm/drop_caches'.
    Also tries perf/LLC flush via sysctl where available.
    """
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
    """Launch substrate_collector.py as a background process."""
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


def stop_collector(proc: subprocess.Popen, log_f, kill_timeout: float = 20.0) -> None:
    """SIGTERM the collector and wait for it to flush perf output."""
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
    """POST one prompt to the llama.cpp /completion endpoint."""
    payload = json.dumps({
        "prompt": prompt_text,
        "n_predict": n_predict,
        "ignore_eos": True,
        "logit_bias": [[2, -100.0]],   # suppress EOS (Llama-2 token 2)
    })
    try:
        out = subprocess.check_output(
            [
                "curl", "-sS",
                "--max-time", str(timeout_s),
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
    ap = argparse.ArgumentParser(
        description="Per-prompt isolated HAT substrate runner."
    )
    ap.add_argument("--json", required=True,
                    help="Path to prompts JSON (list of objects with 'instructions').")
    ap.add_argument("--label", required=True,
                    help="Condition label: 'emotional' or 'neutral'.")
    ap.add_argument("--container", default="mccviahat-llama",
                    help="Docker container name (for host PID resolution).")
    ap.add_argument("--reset_script", default="scripts/model/reset_server.sh",
                    help="Path to the server reset script.")
    ap.add_argument("--collector", default="collectors/substrate_collector.py",
                    help="Path to substrate_collector.py.")
    ap.add_argument("--server_url", default="http://localhost:8000/completion",
                    help="llama.cpp server completion endpoint.")
    ap.add_argument("--n_predict", type=int, default=50,
                    help="Tokens to generate per prompt.")
    ap.add_argument("--baseline_s", type=float, default=2.0,
                    help="Stabilisation sleep after reset, before collector starts (s).")
    ap.add_argument("--per_request_timeout_s", type=float, default=60.0,
                    help="Hard timeout for the curl request (s).")
    ap.add_argument("--perf_interval_ms", type=int, default=1,
                    help="perf stat bucket interval in ms.")
    ap.add_argument("--proc_interval_s", type=float, default=0.1,
                    help="/proc sampling interval (s). Default: 0.1.")
    ap.add_argument("--reset_timeout_s", type=float, default=90.0,
                    help="Max time to wait for server reset script (s).")
    ap.add_argument("--start_index", type=int, default=0,
                    help="Resume from this prompt index (0-based, inclusive).")
    ap.add_argument("--llm_cpus", type=str, default="",
                    help="CPU cores reserved for the LLM container, e.g. '0-11'. "
                         "Enables taskset isolation when combined with --perf_cpu.")
    ap.add_argument("--perf_cpu", type=str, default="",
                    help="CPU core(s) for the perf stat process, e.g. '12'. "
                         "Enables taskset isolation when combined with --llm_cpus.")
    args = ap.parse_args()

    # ── Load prompts ──────────────────────────────────────────────────────────
    prompts: list[dict] = json.load(open(args.json, "r", encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise SystemExit("JSON must be a non-empty list of objects.")

    n_total = len(prompts)
    print(f"Loaded {n_total} prompts from {args.json}  label={args.label}")

    # ── Output root ───────────────────────────────────────────────────────────
    run_root = Path("runs") / args.label
    run_root.mkdir(parents=True, exist_ok=True)


    # ── Per-prompt loop ───────────────────────────────────────────────────────
    for idx, prompt_obj in enumerate(prompts):
        if idx < args.start_index:
            continue  # resume support

        instr = prompt_obj.get("instructions", "")
        if not isinstance(instr, str) or not instr.strip():
            raise SystemExit(f"Prompt {idx}: missing or empty 'instructions' field.")

        out_dir = run_root / f"p{idx:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx+1}/{n_total}] prompt={idx}  out={out_dir}", flush=True)

        # 1. Reset server
        print("  → resetting server …", flush=True)
        try:
            reset_server(args.reset_script, timeout=args.reset_timeout_s)
        except RuntimeError as e:
            print(f"  ✗ {e}")
            (out_dir / "trial_meta.json").write_text(
                json.dumps({"prompt_index": idx, "label": args.label, "ok": False,
                            "error": "reset_failed"}, indent=2)
            )
            continue

        # 2. Drop hardware caches
        print("  → dropping caches …", flush=True)
        drop_caches()

        # 3. Stabilisation baseline (let the server idle cleanly before we measure)
        print(f"  → baseline sleep {args.baseline_s}s …", flush=True)
        time.sleep(args.baseline_s)

        # 3.5. Resolve container PID (after reset)
        pid = get_container_pid(args.container)
        print(f"  Container '{args.container}' → host PID {pid}")

        # 4. Start collector
        #    Ceiling duration: baseline already elapsed; give the full request
        #    timeout plus a small tail so perf doesn't cut off mid-inference.
        collector_duration_s = args.per_request_timeout_s + 5.0
        print("  → starting collector …", flush=True)
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

        # 5. Send prompt
        print("  → sending prompt …", flush=True)
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

        # 6. Stop collector immediately after curl returns
        print("  → stopping collector …", flush=True)
        stop_collector(collector_proc, collector_log_f)

        # 7. Write minimal trial metadata
        trial_meta = {
            "prompt_index": idx,
            "label": args.label,
            "ok": ok,
            "t_request_start_ns": t_req_start,
            "t_request_end_ns": t_req_end,
            "elapsed_ms": elapsed_ms,
        }
        (out_dir / "trial_meta.json").write_text(
            json.dumps(trial_meta, indent=2), encoding="utf-8"
        )

        # Optionally: save raw response for debugging (comment out to save disk)
        if not ok:
            (out_dir / "response_error.txt").write_text(response_raw, encoding="utf-8")

    print(f"\n✓ All done. Results in runs/{args.label}/")


if __name__ == "__main__":
    run()
