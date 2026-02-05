#!/usr/bin/env python3
"""
collectors/substrate_collector.py

Goal: one "collector" process you can start before prompt execution and stop after,
compatible with your fixed-window runner (run_prompts_json).

This collector captures (Option 1: use sudo for kernel tracepoints):
- 1ms-bucket perf stats for system-wide kernel tracepoints and PMU/power signals
  (irq/softirq activity, tlb_flush tracepoint, thermal/power/throttle PMU events)
- a low-overhead /proc time series for the target host PID (utime/stime/rss) + /proc/stat CPU totals
- kernel log slice for the collection window (for rare events like MCE), saved as text

Outputs into a run directory:
- perf_stat.txt
- proc_sample.csv
- kernel_log.txt
- collector_meta.json

Notes:
- Tracefs is root-only on your nodes; perf tracepoints require sudo.
- `sudo -n` is used so it fails fast if passwordless sudo is not available.
"""

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional


# -----------------------------
# /proc sampling (same logic as your existing proc_sampler.py)
# -----------------------------

def read_proc_stat_cpu() -> Tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as f:
        first = f.readline().strip()

    parts = first.split()
    if not parts or parts[0] != "cpu":
        raise RuntimeError("Unexpected /proc/stat format (first line not 'cpu ...')")

    values = list(map(int, parts[1:]))
    total = sum(values)

    idle = values[3]
    if len(values) > 4:
        idle += values[4]

    return total, idle


def read_proc_pid_stat(pid: int) -> Tuple[int, int, int]:
    path = f"/proc/{pid}/stat"
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()

    rparen = s.rfind(")")
    if rparen == -1:
        raise RuntimeError(f"Unexpected format in {path}: missing ')'")
    after = s[rparen + 2 :]  # skip ") "
    fields = after.split()

    utime = int(fields[11])
    stime = int(fields[12])
    rss_pages = int(fields[21])

    return utime, stime, rss_pages


def pid_exists(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


# -----------------------------
# perf + kernel log orchestration
# -----------------------------

@dataclass
class PerfPlan:
    interval_ms: int
    events: List[str]

    def command(self, duration_s: float, out_path: str) -> List[str]:
        # System-wide (-a) is what you want for irq/tlb tracepoints and energy/throttle events.
        # Use `-- sleep <duration>` as a stable wall-clock window.
        # Output is text; we keep it as-is.
        return [
            "sudo", "-n", "perf", "stat",
            "-a",
            "-I", str(self.interval_ms),
            "-e", ",".join(self.events),
            "-o", out_path,
            "--",
            "sleep", str(duration_s),
        ]


def write_kernel_log_slice(out_path: str, t_start_epoch: float, t_end_epoch: float) -> None:
    # Use kernel log slice for rare events like MCE.
    # journalctl generally requires root for full kernel logs on many systems.
    # We use `--since @<epoch>` / `--until @<epoch>`.
    since_arg = f"@{t_start_epoch:.3f}"
    until_arg = f"@{t_end_epoch:.3f}"
    cmd = ["sudo", "-n", "journalctl", "-k", "--since", since_arg, "--until", until_arg, "--no-pager"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = f"[kernel_log_error]\ncommand: {' '.join(cmd)}\nexit_code: {e.returncode}\noutput:\n{e.output}\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fail_fast_sudo() -> None:
    # Ensure we don't hang waiting for a password prompt.
    # If this fails, user must configure sudo or run collector as root.
    cmd = ["sudo", "-n", "true"]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        raise SystemExit(
            "sudo is required for perf tracepoints (tracefs is root-only). "
            "Configure passwordless sudo for this node/user, or run collector as root."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Run output directory (e.g., runs/<run_id>).")
    ap.add_argument("--pid", type=int, required=True, help="Target host PID (e.g., docker inspect .State.Pid).")
    ap.add_argument("--duration_s", type=float, required=True, help="Total wall-clock duration to collect.")
    ap.add_argument("--perf_interval_ms", type=int, default=1, help="perf stat bucket interval in ms (default: 1).")
    ap.add_argument("--proc_interval_s", type=float, default=0.2, help="/proc sampling interval seconds (default: 0.2).")
    ap.add_argument("--collect_kernel_log", action="store_true", help="Also dump kernel log slice for window.")
    ap.add_argument("--events", type=str, default="", help="Comma-separated perf events override (optional).")
    args = ap.parse_args()

    if args.duration_s <= 0:
        raise SystemExit("--duration_s must be > 0")
    if args.perf_interval_ms <= 0:
        raise SystemExit("--perf_interval_ms must be > 0")
    if args.proc_interval_s <= 0:
        raise SystemExit("--proc_interval_s must be > 0")

    pid = args.pid
    if not pid_exists(pid):
        raise SystemExit(f"PID {pid} does not exist on this host. Did you pass the host PID?")

    ensure_dir(args.out_dir)

    # Fail fast if sudo is not usable non-interactively
    fail_fast_sudo()

    # Default event set aligned to your stated needs and what your node listed:
    # - irq tracepoints (interrupts/softirqs/tasklets)
    # - tlb:tlb_flush (proxy for shootdown/flush activity)
    # - thermal/power/throttle PMU signals (as available on your system)
    default_events = [
        "irq:irq_handler_entry",
        "irq:irq_handler_exit",
        "irq:softirq_entry",
        "irq:softirq_exit",
        "irq:softirq_raise",
        "irq:tasklet_entry",
        "irq:tasklet_exit",
        "tlb:tlb_flush",
        "core_power.throttle",
        "msr/cpu_thermal_margin/",
        "power/energy-pkg/",
        "power/energy-ram/",
    ]

    events = [e.strip() for e in args.events.split(",") if e.strip()] if args.events else default_events
    plan = PerfPlan(interval_ms=args.perf_interval_ms, events=events)

    perf_out = os.path.join(args.out_dir, "perf_stat.txt")
    proc_out = os.path.join(args.out_dir, "proc_sample.csv")
    klog_out = os.path.join(args.out_dir, "kernel_log.txt")
    meta_out = os.path.join(args.out_dir, "collector_meta.json")

    # Write meta now (so you can see configuration even if interrupted)
    t0_epoch = time.time()
    t0_ns = time.time_ns()
    meta = {
        "t0_epoch": t0_epoch,
        "t0_ns": t0_ns,
        "duration_s": args.duration_s,
        "perf_interval_ms": args.perf_interval_ms,
        "proc_interval_s": args.proc_interval_s,
        "pid": pid,
        "perf_events": events,
        "perf_command": plan.command(args.duration_s, perf_out),
        "collect_kernel_log": bool(args.collect_kernel_log),
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Start perf stat in the background (system-wide; fixed window via sleep)
    perf_cmd = plan.command(args.duration_s, perf_out)
    perf_proc = subprocess.Popen(perf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # /proc sampler loop for the same duration window
    header = [
        "timestamp_ns",
        "cpu_total_jiffies",
        "cpu_idle_jiffies",
        "pid",
        "proc_utime_jiffies",
        "proc_stime_jiffies",
        "proc_rss_pages",
    ]

    t_end = time.time() + args.duration_s
    with open(proc_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        while time.time() < t_end:
            if not pid_exists(pid):
                break

            ts = time.time_ns()
            cpu_total, cpu_idle = read_proc_stat_cpu()
            utime, stime, rss_pages = read_proc_pid_stat(pid)
            w.writerow([ts, cpu_total, cpu_idle, pid, utime, stime, rss_pages])

            # keep it simple: flush each row so partial runs are still usable
            f.flush()
            time.sleep(args.proc_interval_s)

    # Wait for perf to finish (it should, because it runs "sleep duration")
    try:
        perf_proc.wait(timeout=max(5.0, args.duration_s + 5.0))
    except subprocess.TimeoutExpired:
        # If something went wrong, kill it so you don't leave perf running
        perf_proc.kill()
        perf_proc.wait(timeout=5.0)

    # Kernel log slice (optional, done after to capture full window)
    t1_epoch = time.time()
    if args.collect_kernel_log:
        write_kernel_log_slice(klog_out, t0_epoch, t1_epoch)

    # Update meta with end times
    meta["t1_epoch"] = t1_epoch
    meta["t1_ns"] = time.time_ns()
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {args.out_dir}/")
    print(f"  perf: {perf_out}")
    print(f"  proc: {proc_out}")
    if args.collect_kernel_log:
        print(f"  klog: {klog_out}")
    print(f"  meta: {meta_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
