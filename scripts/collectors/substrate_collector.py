#!/usr/bin/env python3
"""
collectors/substrate_collector_v2.py

Streamlined HAT collector for LLM inference experiments.
Collects only the indicators needed for current and next-step analysis:

  HAT Layer 1 (consciousness-relevant):
    core_power.throttle  — power throttle events (probed, 1ms via perf)
    TLB shootdowns       — tlb:tlb_flush (perf, 1ms) + hat_TLB (/proc/interrupts, 100ms)

  Workload confound controls (new):
    instructions         — retired instruction count (1ms via perf)
    cycles               — CPU cycles consumed (1ms via perf)
    → IPC = instructions / cycles is the primary workload normaliser

  /proc polling (100ms):
    /proc/interrupts     — TLB shootdown counts (cross-CPU coordination)
    /sys/cpufreq         — per-core frequency (thermal state check)
    /proc/<pid>/stat     — per-process CPU utilisation

  Post-trial:
    kernel_log.txt       — dmesg slice for MCE / thermal events

Removed vs v1 (not needed for current analysis):
    cache-misses, cache-references, LLC-load-misses, dTLB-load-misses
    branch-misses, branch-instructions
    stalled-cycles-frontend, stalled-cycles-backend
    uncore_imc/cas_count_read/, cas_count_write/
    power/energy-pkg/, power/energy-ram/
    msr/cpu_thermal_margin/
    context-switches, cpu-migrations, page-faults, cpu-clock
    mce:mce_record, uncore_imc/UNC_M_ECC_CORRECTABLE_ERRORS/
    sched:sched_stat_wait

Usage:
    python substrate_collector_v2.py --out_dir runs/trial_001 --pid 12345 --duration_s 60
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Tuple, List
from collections import OrderedDict


# ─────────────────────────────────────────────────────────
# /proc sampling
# ─────────────────────────────────────────────────────────

def read_proc_stat_cpu() -> Tuple[int, int]:
    """Return (total_jiffies, idle_jiffies) from /proc/stat."""
    with open("/proc/stat", "r", encoding="utf-8") as f:
        first = f.readline().strip()
    parts = first.split()
    if not parts or parts[0] != "cpu":
        raise RuntimeError("Unexpected /proc/stat format")
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
        raise RuntimeError(f"Unexpected format in {path}")
    fields = s[rparen + 2:].split()
    return int(fields[11]), int(fields[12]), int(fields[21])


def pid_exists(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


# ─────────────────────────────────────────────────────────
# /proc/interrupts and /sys/cpufreq
# ─────────────────────────────────────────────────────────

def read_proc_interrupts() -> dict:
    """Parse /proc/interrupts -> {irq_name: total_count_across_cpus}."""
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


def read_cpu_frequencies() -> dict:
    """Read per-core CPU frequency from sysfs (kHz)."""
    try:
        import glob
        data = {}
        for fpath in sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")):
            cpu_num = fpath.split("/cpu")[1].split("/")[0]
            with open(fpath, "r", encoding="utf-8") as f:
                data[f"cpu{cpu_num}_freq_khz"] = int(f.read().strip())
        return data
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────
# perf stat + kernel log
# ─────────────────────────────────────────────────────────

@dataclass
class PerfPlan:
    interval_ms: int
    events: List[str]

    def command(self, duration_s: float, out_path: str, cpu: str = "") -> List[str]:
        prefix = ["taskset", "-c", cpu] if cpu else []
        return prefix + [
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
    """Dump kernel log (dmesg) for the collection window."""
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
    try:
        subprocess.check_call(["sudo", "-n", "true"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        raise SystemExit(
            "sudo is required for perf tracepoints. "
            "Configure passwordless sudo or run as root."
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
    """Convert perf stat -x ',' raw output into a clean wide-format CSV."""
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


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Streamlined HAT collector: throttle + TLB + workload confound controls."
    )
    ap.add_argument("--out_dir", required=True,
                    help="Run output directory (e.g., runs/<run_id>).")
    ap.add_argument("--pid", type=int, required=True,
                    help="Target host PID (e.g., docker inspect .State.Pid).")
    ap.add_argument("--duration_s", type=float, required=True,
                    help="Total wall-clock duration to collect.")
    ap.add_argument("--perf_interval_ms", type=int, default=1,
                    help="perf stat bucket interval in ms (default: 1).")
    ap.add_argument("--proc_interval_s", type=float, default=0.1,
                    help="/proc sampling interval seconds (default: 0.1).")
    ap.add_argument("--events", type=str, default="",
                    help="Comma-separated perf events override (optional).")
    ap.add_argument("--llm_cpus", type=str, default="",
                    help="CPU cores reserved for the LLM, e.g. '0-11'.")
    ap.add_argument("--perf_cpu", type=str, default="",
                    help="CPU core(s) for perf stat itself, e.g. '12'.")
    args = ap.parse_args()

    if args.duration_s <= 0:
        raise SystemExit("--duration_s must be > 0")
    if args.perf_interval_ms <= 0:
        raise SystemExit("--perf_interval_ms must be > 0")
    if args.proc_interval_s <= 0:
        raise SystemExit("--proc_interval_s must be > 0")

    pid = args.pid
    if not pid_exists(pid):
        raise SystemExit(f"PID {pid} does not exist on this host.")

    ensure_dir(args.out_dir)
    fail_fast_sudo()

    # ── CPU pinning ────────────────────────────────────────────────────────
    llm_cpus = args.llm_cpus.strip()
    perf_cpu = args.perf_cpu.strip()
    use_taskset = bool(llm_cpus and perf_cpu)

    if use_taskset:
        print(f"  taskset: LLM pid {pid} → cpus [{llm_cpus}],  perf → cpu [{perf_cpu}]")
        try:
            subprocess.check_call(
                ["sudo", "-n", "taskset", "-acp", llm_cpus, str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"taskset for LLM PID {pid} failed: {e}")
    elif llm_cpus or perf_cpu:
        print("  ⚠  Both --llm_cpus and --perf_cpu must be set to enable taskset. Skipping.")

    # ================================================================
    # Event set — streamlined for current analysis needs
    # ================================================================
    #
    # HAT Layer 1:
    #   tlb:tlb_flush        — TLB shootdown tracepoint (1ms resolution)
    #   core_power.throttle  — power/thermal throttle (probed, 1ms resolution)
    #
    # Workload confound controls:
    #   instructions         — retired instructions per 1ms bucket
    #   cycles               — CPU cycles per 1ms bucket
    #   → IPC = instructions/cycles measures computational intensity
    #   → If emotional/neutral differ in IPC, HAT differences may be
    #     explained by workload rather than emotional content

    hat_events = [
        # HAT Layer 1 — TLB shootdown tracepoint
        "tlb:tlb_flush",

        # Workload confound controls — instructions and cycles for IPC
        "instructions",
        "cycles",
    ]

    # Probe optional events
    probed_events: List[str] = []
    optional_events = [
        ("core_power.throttle",
         "Thermal/Power Anomaly — PROCHOT/power-limit throttle event"),
    ]

    for evt, description in optional_events:
        if _probe_event(evt):
            probed_events.append(evt)
            print(f"  + {evt}  ({description})")
        else:
            print(f"  x {evt}  ({description}) — not available, skipped")

    events = ([e.strip() for e in args.events.split(",") if e.strip()]
              if args.events else hat_events)
    events = events + probed_events
    plan = PerfPlan(interval_ms=args.perf_interval_ms, events=events)

    # ── Output paths ───────────────────────────────────────────────────────
    perf_out    = os.path.join(args.out_dir, "perf_stat.txt")
    proc_out    = os.path.join(args.out_dir, "proc_sample.csv")
    hat_irq_out = os.path.join(args.out_dir, "hat_interrupts.csv")
    klog_out    = os.path.join(args.out_dir, "kernel_log.txt")
    meta_out    = os.path.join(args.out_dir, "collector_meta.json")

    # ── Write meta ─────────────────────────────────────────────────────────
    t0_epoch = time.time()
    t0_ns = time.time_ns()
    meta = {
        "t0_epoch": t0_epoch,
        "t0_ns": t0_ns,
        "duration_s": args.duration_s,
        "perf_interval_ms": args.perf_interval_ms,
        "proc_interval_s": args.proc_interval_s,
        "pid": pid,
        "llm_cpus": llm_cpus,
        "perf_cpu": perf_cpu,
        "taskset_enabled": use_taskset,
        "perf_events": events,
        "probed_events": probed_events,
        "perf_command": plan.command(args.duration_s, perf_out,
                                     cpu=perf_cpu if use_taskset else ""),
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ── Start perf stat ────────────────────────────────────────────────────
    perf_cmd = plan.command(args.duration_s, perf_out,
                            cpu=perf_cpu if use_taskset else "")
    perf_log = os.path.join(args.out_dir, "perf_stderr.log")
    with open(perf_log, "w") as perf_log_f:
        perf_proc = subprocess.Popen(perf_cmd, stdout=subprocess.DEVNULL,
                                     stderr=perf_log_f,
                                     start_new_session=True)

    # ── CSV headers ────────────────────────────────────────────────────────
    proc_header = [
        "timestamp_ns",
        "cpu_total_jiffies",
        "cpu_idle_jiffies",
        "pid",
        "proc_utime_jiffies",
        "proc_stime_jiffies",
        "proc_rss_pages",
    ]

    sample_interrupts = read_proc_interrupts()
    sample_freq = read_cpu_frequencies()

    hat_header = ["timestamp_ns"]
    hat_header += sorted(sample_interrupts.keys())
    hat_header += sorted(sample_freq.keys())

    t_end = time.time() + args.duration_s

    # ── SIGTERM handler ────────────────────────────────────────────────────
    _stop_requested = False

    def _handle_sigterm(signum, frame):
        nonlocal _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # ── Sampling loop ──────────────────────────────────────────────────────
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

            # Process + system CPU utilisation
            cpu_total, cpu_idle = read_proc_stat_cpu()
            utime, stime, rss_pages = read_proc_pid_stat(pid)
            w_proc.writerow([ts, cpu_total, cpu_idle, pid, utime, stime, rss_pages])
            f_proc.flush()

            # HAT interrupt counts + CPU frequency
            interrupts = read_proc_interrupts()
            freq = read_cpu_frequencies()

            all_metrics = {}
            all_metrics.update(interrupts)
            all_metrics.update(freq)

            hat_row = [ts]
            for key in hat_header[1:]:
                val = all_metrics.get(key)
                hat_row.append(val if val is not None else "")
            w_hat.writerow(hat_row)
            f_hat.flush()

            time.sleep(args.proc_interval_s)

    # ── Stop perf ──────────────────────────────────────────────────────────
    if _stop_requested:
        try:
            os.killpg(perf_proc.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            perf_proc.terminate()
    try:
        perf_proc.wait(timeout=max(10.0, args.duration_s + 5.0))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(perf_proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            perf_proc.kill()
        perf_proc.wait(timeout=5.0)

    # ── Post-process perf CSV ──────────────────────────────────────────────
    perf_csv_out = os.path.join(args.out_dir, "perf_stat.csv")
    if perf_proc.returncode != 0:
        print(f"  ⚠ perf stat exited with code {perf_proc.returncode}")
        if os.path.isfile(perf_log):
            err_text = open(perf_log, "r", encoding="utf-8",
                            errors="replace").read().strip()
            if err_text:
                for ln in err_text.splitlines()[:20]:
                    print(f"    {ln}")
    if os.path.isfile(perf_out) and os.path.getsize(perf_out) > 0:
        _postprocess_perf_csv(perf_out, perf_csv_out)
    else:
        print(f"  ⚠ {perf_out} not found or empty — perf stat may have failed.")
        print(f"    Check {perf_log} for details.")

    # ── Kernel log ─────────────────────────────────────────────────────────
    t1_epoch = time.time()
    write_kernel_log_slice(klog_out, t0_epoch, t1_epoch)

    # ── Update meta ────────────────────────────────────────────────────────
    meta["t1_epoch"] = t1_epoch
    meta["t1_ns"] = time.time_ns()
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote: {args.out_dir}/")
    print(f"  perf raw:       {perf_out}")
    print(f"  perf csv:       {perf_csv_out}")
    print(f"  proc:           {proc_out}")
    print(f"  hat_interrupts: {hat_irq_out}")
    print(f"  kernel log:     {klog_out}")
    print(f"  meta:           {meta_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
