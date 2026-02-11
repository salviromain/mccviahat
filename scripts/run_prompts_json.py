#!/usr/bin/env python3
import argparse, json, os, subprocess, time
from pathlib import Path

def sh(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

def now_ns() -> int:
    return time.time_ns()

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to prompts JSON (list of objects with 'instructions').")
    ap.add_argument("--label", required=True, help="Run label, e.g. neutral or emotional.")
    ap.add_argument("--container", default="mccviahat-llama", help="Docker container name.")
    ap.add_argument("--n_predict", type=int, default=50)
    ap.add_argument("--baseline_s", type=float, default=3.0)
    ap.add_argument("--tail_s", type=float, default=3.0)
    ap.add_argument("--budget_s", type=float, default=None,
                    help="Fixed time budget (s). Default: auto = n_prompts × per_request_timeout_s.")
    ap.add_argument("--interval_s", type=float, default=0.01, help="Sampling interval for proc sampler.")
    ap.add_argument("--per_request_timeout_s", type=float, default=25.0, help="Hard timeout per request.")
    args = ap.parse_args()

    prompts = json.load(open(args.json, "r", encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise SystemExit("JSON must be a non-empty list.")
    if len(prompts) > 20:
        print(f"Capping prompts from {len(prompts)} → 20")
        prompts = prompts[:20]

    # Auto-compute budget: n_prompts × per_request_timeout_s (worst-case estimate).
    # --budget_s overrides if given explicitly.
    if args.budget_s is None:
        args.budget_s = len(prompts) * args.per_request_timeout_s
    print(f"Prompts: {len(prompts)},  budget: {args.budget_s:.0f}s "
          f"({args.budget_s/60:.1f} min),  per-request timeout: {args.per_request_timeout_s}s")

    # Resolve container PID on host
    pid = sh(["docker", "inspect", "-f", "{{.State.Pid}}", args.container])
    if not pid.isdigit() or int(pid) <= 0:
        raise SystemExit(f"Could not resolve PID for container {args.container}: {pid}")

    run_id = time.strftime("%Y-%m-%dT%H-%M-%S") + f"_{args.label}"
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"
    resp_path = out_dir / "responses.jsonl"
    csv_path  = out_dir / "proc_sample.csv"

    total_s = args.baseline_s + args.budget_s + args.tail_s
    t0_ns = now_ns()
    t_end_target_ns = t0_ns + int(total_s * 1e9)

    # Start substrate collector for the full fixed window.
    # Redirect its stdout/stderr to a log file so it doesn't
    # interleave with our own output or the shell prompt.
    collector_log_path = out_dir / "collector_stdout.log"
    collector_log = open(collector_log_path, "w", encoding="utf-8")
    collector_proc = subprocess.Popen(
        ["python3", "collectors/substrate_collector.py",
         "--out_dir", str(out_dir),
         "--pid", str(pid),
         "--duration_s", str(total_s),
         "--perf_interval_ms", "1",
         "--proc_interval_s", str(args.interval_s),
         "--collect_kernel_log"],
        stdout=collector_log,
        stderr=subprocess.STDOUT,
    )


    # Baseline
    time.sleep(args.baseline_s)

    with open(resp_path, "w", encoding="utf-8") as rf:
        req_events = []
        for p in prompts:
            instr = p.get("instructions", "")
            if not isinstance(instr, str) or not instr.strip():
                raise SystemExit("Each prompt must have a non-empty 'instructions' string.")

            # Stop early if we’ve hit the fixed budget window (strict equality)
            if now_ns() >= (t0_ns + int((args.baseline_s + args.budget_s) * 1e9)):
                break

            payload = json.dumps({"prompt": instr, "n_predict": args.n_predict, "ignore_eos": True})
            t_req_start = now_ns()
            try:
                out = subprocess.check_output(
                    ["curl", "-sS", "--max-time", str(args.per_request_timeout_s),
                     "http://localhost:8000/completion",
                     "-H", "Content-Type: application/json",
                     "-d", payload],
                    text=True
                )
                ok = True
            except subprocess.CalledProcessError as e:
                out = str(e)
                ok = False
            t_req_end = now_ns()

            rec = {
                "id": p.get("id"),
                "title": p.get("title"),
                "ok": ok,
                "t_request_start_ns": t_req_start,
                "t_request_end_ns": t_req_end,
                "response_raw": out,
            }
            rf.write(json.dumps(rec) + "\n")
            req_events.append(rec)

    # Tail / pad until fixed end time (so total ms is identical)
    remaining_ns = t_end_target_ns - now_ns()
    if remaining_ns > 0:
        time.sleep(remaining_ns / 1e9)


    meta = {
        "run_id": run_id,
        "label": args.label,
        "container": args.container,
        "pid": int(pid),
        "n_prompts": len(prompts),
        "n_predict": args.n_predict,
        "baseline_s": args.baseline_s,
        "budget_s": args.budget_s,
        "tail_s": args.tail_s,
        "interval_s": args.interval_s,
        "total_s": total_s,
        "t0_ns": t0_ns,
        "t_end_target_ns": t_end_target_ns,
        "json_source": os.path.abspath(args.json),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Wait for the substrate collector to finish before printing anything.
    print("Waiting for substrate collector to finish ...", flush=True)
    collector_proc.wait()
    collector_log.close()

    # Print collector log so the user sees what it wrote.
    print("\n--- substrate_collector output ---")
    print(collector_log_path.read_text(encoding="utf-8").rstrip())
    print("--- end ---\n")

    print(f"Wrote: {out_dir}/")
    print(f"  {csv_path}")
    print(f"  {resp_path}")
    print(f"  {meta_path}")

if __name__ == "__main__":
    run()
