#!/usr/bin/env python3
"""
collectors/substrate_collector.py

HAT-focused collector: captures the Hardware Anomaly Trace (HAT) and supporting
substrate metrics for an LLM inference server.

The HAT (§4.2.2 of the thesis) comprises two layers:

  Layer 1 — Hardware Anomaly Interrupts:
    Spurious interrupts, Machine Check Exceptions (MCE), Performance Counter
    Overflow (NMI/PMI), Thermal/Power anomalies, TLB Shootdowns.

  Layer 2 — Continuous performance metrics that indicate hardware anomalies:
    Cache hit rates, power consumption variations, thermal sensor readings,
    memory access pattern entropy, clock cycle variations.

This collector captures both layers via three mechanisms:

  1. `perf stat -I 1` (1ms buckets, system-wide):
     - HAT Layer 1 tracepoints: IRQ handlers (including spurious), softirq,
       TLB flushes, thermal throttle
     - HAT Layer 2 PMU counters: cache misses/references, LLC misses, branch
       mispredictions, instructions, cycles, pipeline stalls, dTLB misses
     - HAT Layer 2 power/energy: RAPL energy (pkg + ram), thermal margin
     - Supporting events: context-switches, cpu-migrations, page-faults, cpu-clock

  2. `/proc` + `/sys` polling (200ms):
     - /proc/interrupts: SPU (spurious), NMI/PMI (perf counter overflow),
       MCE (machine check), TLB (shootdown), LOC, RES, CAL — all named HAT events
     - /proc/softirqs: deferred interrupt processing by type
     - /sys/.../scaling_cur_freq: CPU frequency per core (DVFS confounder)
     - /proc/stat + /proc/<pid>/stat: CPU utilisation (confounder)

  3. Kernel log slice (always collected):
     - journalctl -k: captures MCE records, thermal events, hardware errors

Removed (not HAT-relevant):
  - /proc/net/dev — no network in inference hot path
  - /proc/diskstats — model is mmap'd and cached
  - /proc/pressure/* — OS scheduler accounting, not hardware anomalies
  - irq:tasklet_entry/exit — near-zero on modern kernels, not a named HAT type

Outputs:
  perf_stat.txt       — raw perf stat output
  perf_stat.csv       — wide-format (1 row/ms, 1 col/event)
  proc_sample.csv     — per-process CPU + RSS (200ms)
  hat_interrupts.csv  — /proc/interrupts + /proc/softirqs + CPU freq (200ms)
  kernel_log.txt      — dmesg slice (MCE, thermal, hardware errors)
  collector_meta.json — configuration + timestamps
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────────────────
# /proc sampling
# ─────────────────────────────────────────────────────────

def read_proc_stat_cpu() -> Tuple[int, int]:
    """Return (total_jiffies, idle_jiffies) from /proc/stat."""
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
    """Return (utime, stime, rss_pages) for a given PID."""
    path = f"/proc/{pid}/stat"
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()

    rparen = s.rfind(")")
    if rparen == -1:
        raise RuntimeError(f"Unexpected format in {path}: missing ')'")
    after = s[rparen + 2:]  # skip ") "
    fields = after.split()

    utime = int(fields[11])
    stime = int(fields[12])
    rss_pages = int(fields[21])

    return utime, stime, rss_pages


def pid_exists(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


# ─────────────────────────────────────────────────────────
# HAT-relevant /proc and /sys reads
# ─────────────────────────────────────────────────────────

def read_proc_interrupts() -> dict:
    """Parse /proc/interrupts -> {irq_name: total_count_across_cpus}.

    Key HAT rows: SPU (spurious), NMI, PMI (perf counter overflow),
    MCE (machine check), TLB (shootdown), LOC (local APIC timer),
    RES (rescheduling IPI), CAL (function call IPI).
    """
    try:
        with open("/proc/interrupts", "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = {}
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            irq_name = parts[0].rstrip(":")
            counts = [int(x) for x in parts[1:] if x.isdigit()]
            data[irq_name] = sum(counts)
        return data
    except Exception:
        return {}


def read_proc_softirqs() -> dict:
    """Parse /proc/softirqs -> {type: total_count_across_cpus}.

    Types: HI, TIMER, NET_TX, NET_RX, BLOCK, TASKLET, SCHED, HRTIMER, RCU.
    """
    try:
        with open("/proc/softirqs", "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = {}
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            sirq_name = parts[0].rstrip(":")
            counts = [int(x) for x in parts[1:] if x.isdigit()]
            data[sirq_name] = sum(counts)
        return data
    except Exception:
        return {}


def read_cpu_frequencies() -> dict:
    """Read per-core CPU frequency from sysfs (kHz)."""
    try:
        import glob
        data = {}
        freq_files = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")
        for fpath in sorted(freq_files):
            cpu_num = fpath.split("/cpu")[1].split("/")[0]
            with open(fpath, "r", encoding="utf-8") as f:
                freq_khz = int(f.read().strip())
                data[f"cpu{cpu_num}_freq_khz"] = freq_khz
        return data
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────
# perf + kernel log
# ─────────────────────────────────────────────────────────

@dataclass
class PerfPlan:
    interval_ms: int
    events: List[str]

    def command(self, duration_s: float, out_path: str) -> List[str]:
        return [
            "sudo", "-n", "perf", "stat",
            "-a",
            "-x", ",",
            "-I", str(self.interval_ms),
            "-e", ",".join(self.events),
            "-o", out_path,
            "--",
            "sleep", str(duration_s),
        ]


def write_kernel_log_slice(out_path: str, t_start_epoch: float, t_end_epoch: float) -> None:
    """Dump kernel log (dmesg) for the collection window.

    Captures MCE records, thermal events, and other hardware errors --
    these are HAT Layer 1 anomalies that don't always generate perf events.
    """
    since_arg = f"@{t_start_epoch:.3f}"
    until_arg = f"@{t_end_epoch:.3f}"
    cmd = ["sudo", "-n", "journalctl", "-k", "--since", since_arg,
           "--until", until_arg, "--no-pager"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = (f"[kernel_log_error]\ncommand: {' '.join(cmd)}\n"
               f"exit_code: {e.returncode}\noutput:\n{e.output}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fail_fast_sudo() -> None:
    cmd = ["sudo", "-n", "true"]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        raise SystemExit(
            "sudo is required for perf tracepoints (tracefs is root-only). "
            "Configure passwordless sudo for this node/user, or run collector as root."
        )


def _probe_event(event: str) -> bool:
    """Return True if perf can open this event on the running kernel."""
    try:
        subprocess.check_call(
            ["sudo", "-n", "perf", "stat", "-e", event, "--", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _postprocess_perf_csv(raw_path: str, out_path: str) -> None:
    """Convert perf stat -x ',' raw output into a clean wide-format CSV.

    Input:  timestamp,value,unit,event,runtime,pct_running,...
    Output: t_s, event_1, event_2, ...
    """
    from collections import OrderedDict

    rows_by_ts: dict = OrderedDict()
    events_seen: list = []

    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            ts_str = parts[0].strip()
            val_str = parts[1].strip()
            event = parts[3].strip()
            if not event:
                continue
            try:
                ts = float(ts_str)
            except ValueError:
                continue
            if val_str.startswith("<") or val_str == "":
                val = float("nan")
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    val = float("nan")

            if event not in events_seen:
                events_seen.append(event)
            if ts not in rows_by_ts:
                rows_by_ts[ts] = {}
            rows_by_ts[ts][event] = val

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["t_s"] + events_seen
        w.writerow(header)
        for ts in rows_by_ts:
            row = [ts]
            for evt in events_seen:
                row.append(rows_by_ts[ts].get(evt, ""))
            w.writerow(row)

    print(f"  perf_stat.csv: {len(rows_by_ts)} rows x {len(events_seen)} events")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HAT-focused substrate collector for LLM inference experiments."
    )
    ap.add_argument("--out_dir", required=True,
                    help="Run output directory (e.g., runs/<run_id>).")
    ap.add_argument("--pid", type=int, required=True,
                    help="Target host PID (e.g., docker inspect .State.Pid).")
    ap.add_argument("--duration_s", type=float, required=True,
                    help="Total wall-clock duration to collect.")
    ap.add_argument("--perf_interval_ms", type=int, default=1,
                    help="perf stat bucket interval in ms (default: 1).")
    ap.add_argument("--proc_interval_s", type=float, default=0.2,
                    help="/proc sampling interval seconds (default: 0.2).")
    ap.add_argument("--events", type=str, default="",
                    help="Comma-separated perf events override (optional).")
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
    fail_fast_sudo()

    # ================================================================
    # HAT event set
    # ================================================================
    #
    # HAT Layer 1 -- Hardware Anomaly Interrupts (tracepoints):
    #   irq:irq_handler_entry/exit -- ALL hardware IRQs (includes spurious)
    #   irq:softirq_entry/exit/raise -- deferred interrupt processing
    #   tlb:tlb_flush -- TLB Shootdown (named HAT anomaly)
    #   core_power.throttle -- Thermal/Power anomaly (named HAT anomaly)
    #   mce:mce_record -- Machine Check Exception (named HAT anomaly, probed)
    #
    # HAT Layer 2 -- Continuous performance metrics (PMU counters):
    #   cache-misses, cache-references -- cache hit rate per time unit
    #   LLC-load-misses -- last-level cache pressure
    #   branch-misses, branch-instructions -- branch prediction accuracy
    #   instructions, cycles -- IPC / clock cycle variations
    #   stalled-cycles-frontend, stalled-cycles-backend -- pipeline stalls
    #   dTLB-load-misses -- memory access pattern indicator
    #
    # HAT Layer 2 -- Power/thermal:
    #   power/energy-pkg/, power/energy-ram/ -- power consumption variation
    #   msr/cpu_thermal_margin/ -- thermal sensor reading
    #
    # Supporting (confounders/context, not HAT itself):
    #   context-switches, cpu-migrations -- scheduling context
    #   page-faults -- memory management context
    #   cpu-clock -- CPU time normaliser

    hat_events = [
        # -- HAT Layer 1: hardware anomaly tracepoints --
        "irq:irq_handler_entry",
        "irq:irq_handler_exit",
        "irq:softirq_entry",
        "irq:softirq_exit",
        "irq:softirq_raise",
        "tlb:tlb_flush",                # TLB Shootdown
        "core_power.throttle",          # Thermal/Power Anomaly
        # -- HAT Layer 2: PMU hardware counters --
        "cache-misses",
        "cache-references",
        "LLC-load-misses",
        "branch-misses",
        "branch-instructions",
        "instructions",
        "cycles",
        "stalled-cycles-frontend",
        "stalled-cycles-backend",
        "dTLB-load-misses",
        # -- HAT Layer 2: power / thermal --
        "msr/cpu_thermal_margin/",
        "power/energy-pkg/",
        "power/energy-ram/",
        # -- Supporting (confounders) --
        "context-switches",
        "cpu-migrations",
        "page-faults",
        "cpu-clock",
    ]

    # Probe optional HAT Layer 1 events that depend on kernel config
    probed_events: List[str] = []
    optional_hat_events = [
        "mce:mce_record",               # Machine Check Exception
    ]
    for evt in optional_hat_events:
        if _probe_event(evt):
            probed_events.append(evt)
            print(f"  + {evt}")
        else:
            print(f"  x {evt} (not available on this kernel -- skipped)")

    events = ([e.strip() for e in args.events.split(",") if e.strip()]
              if args.events else hat_events)
    events = events + probed_events
    plan = PerfPlan(interval_ms=args.perf_interval_ms, events=events)

    # Output paths
    perf_out = os.path.join(args.out_dir, "perf_stat.txt")
    proc_out = os.path.join(args.out_dir, "proc_sample.csv")
    hat_irq_out = os.path.join(args.out_dir, "hat_interrupts.csv")
    klog_out = os.path.join(args.out_dir, "kernel_log.txt")
    meta_out = os.path.join(args.out_dir, "collector_meta.json")

    # Write meta now (visible even if interrupted)
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
        "probed_events": probed_events,
        "perf_command": plan.command(args.duration_s, perf_out),
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Start perf stat in background
    perf_cmd = plan.command(args.duration_s, perf_out)
    perf_log = os.path.join(args.out_dir, "perf_stderr.log")
    with open(perf_log, "w") as perf_log_f:
        perf_proc = subprocess.Popen(perf_cmd, stdout=subprocess.DEVNULL,
                                     stderr=perf_log_f)

    # -- CSV headers --
    proc_header = [
        "timestamp_ns",
        "cpu_total_jiffies",
        "cpu_idle_jiffies",
        "pid",
        "proc_utime_jiffies",
        "proc_stime_jiffies",
        "proc_rss_pages",
    ]

    # Build HAT interrupt CSV header from initial samples
    sample_interrupts = read_proc_interrupts()
    sample_softirqs = read_proc_softirqs()
    sample_freq = read_cpu_frequencies()

    hat_header = ["timestamp_ns"]
    hat_header += sorted(sample_interrupts.keys())
    hat_header += sorted(sample_softirqs.keys())
    hat_header += sorted(sample_freq.keys())

    t_end = time.time() + args.duration_s

    # SIGTERM handler for graceful shutdown
    _stop_requested = False

    def _handle_sigterm(signum, frame):
        nonlocal _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # -- Sampling loop --
    with open(proc_out, "w", newline="", encoding="utf-8") as f_proc, \
         open(hat_irq_out, "w", newline="", encoding="utf-8") as f_hat:

        w_proc = csv.writer(f_proc)
        w_proc.writerow(proc_header)

        w_hat = csv.writer(f_hat)
        w_hat.writerow(hat_header)

        while time.time() < t_end and not _stop_requested:
            if not pid_exists(pid):
                break

            ts = time.time_ns()

            # Process + system CPU
            cpu_total, cpu_idle = read_proc_stat_cpu()
            utime, stime, rss_pages = read_proc_pid_stat(pid)
            w_proc.writerow([ts, cpu_total, cpu_idle, pid, utime, stime, rss_pages])
            f_proc.flush()

            # HAT interrupt counts + softirqs + CPU frequency
            interrupts = read_proc_interrupts()
            softirqs = read_proc_softirqs()
            freq = read_cpu_frequencies()

            all_metrics = {}
            all_metrics.update(interrupts)
            all_metrics.update(softirqs)
            all_metrics.update(freq)

            hat_row = [ts]
            for key in hat_header[1:]:
                val = all_metrics.get(key)
                hat_row.append(val if val is not None else "")

            w_hat.writerow(hat_row)
            f_hat.flush()

            time.sleep(args.proc_interval_s)

    # Wait for perf to finish
    if _stop_requested:
        perf_proc.terminate()
    try:
        perf_proc.wait(timeout=max(5.0, args.duration_s + 5.0))
    except subprocess.TimeoutExpired:
        perf_proc.kill()
        perf_proc.wait(timeout=5.0)

    # Post-process perf CSV
    perf_csv_out = os.path.join(args.out_dir, "perf_stat.csv")
    _postprocess_perf_csv(perf_out, perf_csv_out)

    # Kernel log slice -- always collected (MCE, thermal, hardware errors)
    t1_epoch = time.time()
    write_kernel_log_slice(klog_out, t0_epoch, t1_epoch)

    # Update meta with end times
    meta["t1_epoch"] = t1_epoch
    meta["t1_ns"] = time.time_ns()
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {args.out_dir}/")
    print(f"  perf raw:       {perf_out}")
    print(f"  perf csv:       {perf_csv_out}")
    print(f"  proc:           {proc_out}")
    print(f"  hat_interrupts: {hat_irq_out}")
    print(f"  kernel log:     {klog_out}")
    print(f"  meta:           {meta_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
