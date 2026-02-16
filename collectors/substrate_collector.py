#!/usr/bin/env python3
"""
collectors/substrate_collector.py

Goal: one "collector" process you can start before prompt execution and stop after,
compatible with your fixed-window runner (run_prompts_json).

This collector captures:
- 1ms-bucket perf stats for system-wide kernel tracepoints and PMU/power signals
  (irq/softirq activity, tlb_flush tracepoint, thermal/power/throttle PMU events)
  Plus low-overhead software counters (context-switches, cpu-migrations, page-faults)
- a low-overhead /proc time series for the target host PID (utime/stime/rss) + /proc/stat CPU totals
- kernel log slice for the collection window (for rare events like MCE), saved as text
- Additional /proc and /sys metrics sampled at proc_interval_s:
  * /proc/interrupts, /proc/softirqs (interrupt counts per CPU)
  * /proc/pressure/{cpu,memory,io} (PSI metrics)
  * /proc/net/dev (network I/O counters)
  * /proc/diskstats (disk I/O counters)
  * CPU frequency from /sys/.../scaling_cur_freq

Outputs into a run directory:
- perf_stat.txt
- proc_sample.csv (process + system CPU)
- proc_system_sample.csv (interrupts, pressure, network, disk, freq)
- kernel_log.txt
- collector_meta.json

Notes:
- Tracefs is root-only on your nodes; perf tracepoints require sudo.
- `sudo -n` is used so it fails fast if passwordless sudo is not available.
- Most new metrics (/proc, /sys) require no special permissions.

Practical note on CloudLab / bare metal nodes:
Most of the "cheap" adds are /proc and /sys reads (no root needed), plus some extra
tracepoints/software perf events (root often needed for tracepoints; PMU access depends
on perf_event_paranoid and kernel config). In other words: you can add a lot without
making the collector fragile.
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
# Additional /proc and /sys sampling
# -----------------------------

def read_proc_interrupts() -> dict:
    """Parse /proc/interrupts and return total counts per interrupt type."""
    try:
        with open("/proc/interrupts", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # First line is CPU header, skip it
        data = {}
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            irq_name = parts[0].rstrip(":")
            # Sum counts across all CPUs (numeric fields after IRQ name)
            counts = [int(x) for x in parts[1:] if x.isdigit()]
            data[irq_name] = sum(counts)
        return data
    except Exception:
        return {}


def read_proc_softirqs() -> dict:
    """Parse /proc/softirqs and return total counts per softirq type."""
    try:
        with open("/proc/softirqs", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # First line is CPU header, skip it
        data = {}
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            sirq_name = parts[0].rstrip(":")
            # Sum counts across all CPUs
            counts = [int(x) for x in parts[1:] if x.isdigit()]
            data[sirq_name] = sum(counts)
        return data
    except Exception:
        return {}


def read_proc_pressure(resource: str) -> dict:
    """Read /proc/pressure/{cpu,memory,io} and return avg10, avg60, avg300, total."""
    try:
        with open(f"/proc/pressure/{resource}", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        result = {}
        for line in lines:
            if line.startswith("some") or line.startswith("full"):
                prefix = line.split()[0]
                parts = line.split()
                for part in parts[1:]:
                    if "=" in part:
                        k, v = part.split("=")
                        result[f"{prefix}_{k}"] = float(v)
        return result
    except Exception:
        return {}


def read_proc_net_dev() -> dict:
    """Parse /proc/net/dev and return rx/tx bytes and packets for all interfaces."""
    try:
        with open("/proc/net/dev", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        data = {}
        for line in lines[2:]:  # Skip two header lines
            if ":" not in line:
                continue
            iface, stats = line.split(":", 1)
            iface = iface.strip()
            parts = stats.split()
            if len(parts) >= 16:
                data[f"{iface}_rx_bytes"] = int(parts[0])
                data[f"{iface}_rx_packets"] = int(parts[1])
                data[f"{iface}_tx_bytes"] = int(parts[8])
                data[f"{iface}_tx_packets"] = int(parts[9])
        return data
    except Exception:
        return {}


def read_proc_diskstats() -> dict:
    """Parse /proc/diskstats and return read/write counts for major block devices."""
    try:
        with open("/proc/diskstats", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        data = {}
        for line in lines:
            parts = line.split()
            if len(parts) < 14:
                continue
            dev_name = parts[2]
            # Skip loop, ram, and partition devices; focus on whole disks
            if dev_name.startswith("loop") or dev_name.startswith("ram") or any(c.isdigit() for c in dev_name[-1:]):
                continue
            data[f"{dev_name}_reads_completed"] = int(parts[3])
            data[f"{dev_name}_sectors_read"] = int(parts[5])
            data[f"{dev_name}_writes_completed"] = int(parts[7])
            data[f"{dev_name}_sectors_written"] = int(parts[9])
        return data
    except Exception:
        return {}


def read_cpu_frequencies() -> dict:
    """Read current CPU frequencies from /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq."""
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


# -----------------------------
# perf + kernel log orchestration
# -----------------------------

@dataclass
class PerfPlan:
    interval_ms: int
    events: List[str]

    def command(self, duration_s: float, out_path: str) -> List[str]:
        # System-wide (-a) with CSV output (-x ',').
        # -x replaces the human-readable format with machine-parseable CSV.
        # Fields: timestamp, counter-value, unit, event-name, ...
        # -o writes to file (perf stat outputs to stderr by default).
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


def _postprocess_perf_csv(raw_path: str, out_path: str) -> None:
    """Convert perf stat -x ',' raw output into a clean wide-format CSV.
    
    perf stat -x ',' -I <ms> produces lines like:
        <timestamp>,<value>,<unit>,<event>,<runtime>,<pct_running>,...
    or (no unit):
        <timestamp>,<value>,,<event>,<runtime>,<pct_running>,...
    
    We pivot this into one row per timestamp, one column per event.
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
            # parts[2] is unit (may be empty)
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

    # Write wide-format CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["t_s"] + events_seen
        w.writerow(header)
        for ts in rows_by_ts:
            row = [ts]
            for evt in events_seen:
                row.append(rows_by_ts[ts].get(evt, ""))
            w.writerow(row)

    print(f"  perf_stat.csv: {len(rows_by_ts)} rows × {len(events_seen)} events")


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
    # - software counters (context-switches, cpu-migrations, page-faults, cpu-clock)
    default_events = [
        "irq:irq_handler_entry",
        "irq:irq_handler_exit",
        "irq:softirq_entry",
        "irq:softirq_exit",
        "irq:softirq_raise",
        "irq:tasklet_entry",
        "irq:tasklet_exit",
        "tlb:tlb_flush",
        "context-switches",
        "cpu-migrations",
        "page-faults",
        "cpu-clock",
        "core_power.throttle",
        "msr/cpu_thermal_margin/",
        "power/energy-pkg/",
        "power/energy-ram/",
    ]

    events = [e.strip() for e in args.events.split(",") if e.strip()] if args.events else default_events
    plan = PerfPlan(interval_ms=args.perf_interval_ms, events=events)

    perf_out = os.path.join(args.out_dir, "perf_stat.txt")
    proc_out = os.path.join(args.out_dir, "proc_sample.csv")
    proc_sys_out = os.path.join(args.out_dir, "proc_system_sample.csv")
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
    # Redirect any remaining stderr (warnings, etc.) to a log file to keep terminal clean
    perf_log = os.path.join(args.out_dir, "perf_stderr.log")
    with open(perf_log, "w") as perf_log_f:
        perf_proc = subprocess.Popen(perf_cmd, stdout=subprocess.DEVNULL, stderr=perf_log_f)

    # /proc sampler loop for the same duration window
    proc_header = [
        "timestamp_ns",
        "cpu_total_jiffies",
        "cpu_idle_jiffies",
        "pid",
        "proc_utime_jiffies",
        "proc_stime_jiffies",
        "proc_rss_pages",
    ]

    # Collect initial system metrics to determine CSV columns
    sample_interrupts = read_proc_interrupts()
    sample_softirqs = read_proc_softirqs()
    sample_pressure_cpu = read_proc_pressure("cpu")
    sample_pressure_mem = read_proc_pressure("memory")
    sample_pressure_io = read_proc_pressure("io")
    sample_net = read_proc_net_dev()
    sample_disk = read_proc_diskstats()
    sample_freq = read_cpu_frequencies()

    # Build system CSV header dynamically (prefix PSI keys to avoid duplicate column names)
    sys_header = ["timestamp_ns"]
    sys_header += sorted(sample_interrupts.keys())
    sys_header += sorted(sample_softirqs.keys())
    sys_header += [f"psi_cpu_{k}" for k in sorted(sample_pressure_cpu.keys())]
    sys_header += [f"psi_mem_{k}" for k in sorted(sample_pressure_mem.keys())]
    sys_header += [f"psi_io_{k}" for k in sorted(sample_pressure_io.keys())]
    sys_header += sorted(sample_net.keys())
    sys_header += sorted(sample_disk.keys())
    sys_header += sorted(sample_freq.keys())

    t_end = time.time() + args.duration_s

    # Handle SIGTERM gracefully so post-processing runs when the runner
    # kills the collector after the 10-second tail.
    _stop_requested = False

    def _handle_sigterm(signum, frame):
        nonlocal _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    with open(proc_out, "w", newline="", encoding="utf-8") as f_proc, \
         open(proc_sys_out, "w", newline="", encoding="utf-8") as f_sys:
        
        w_proc = csv.writer(f_proc)
        w_proc.writerow(proc_header)
        
        w_sys = csv.writer(f_sys)
        w_sys.writerow(sys_header)

        while time.time() < t_end and not _stop_requested:
            if not pid_exists(pid):
                break

            ts = time.time_ns()
            
            # Process and system CPU metrics
            cpu_total, cpu_idle = read_proc_stat_cpu()
            utime, stime, rss_pages = read_proc_pid_stat(pid)
            w_proc.writerow([ts, cpu_total, cpu_idle, pid, utime, stime, rss_pages])
            f_proc.flush()

            # Additional system metrics
            interrupts = read_proc_interrupts()
            softirqs = read_proc_softirqs()
            pressure_cpu = read_proc_pressure("cpu")
            pressure_mem = read_proc_pressure("memory")
            pressure_io = read_proc_pressure("io")
            net = read_proc_net_dev()
            disk = read_proc_diskstats()
            freq = read_cpu_frequencies()

            # Build row matching header order
            # Merge all dicts into one with proper prefixed PSI keys
            all_metrics = {}
            all_metrics.update(interrupts)
            all_metrics.update(softirqs)
            all_metrics.update({f"psi_cpu_{k}": v for k, v in pressure_cpu.items()})
            all_metrics.update({f"psi_mem_{k}": v for k, v in pressure_mem.items()})
            all_metrics.update({f"psi_io_{k}": v for k, v in pressure_io.items()})
            all_metrics.update(net)
            all_metrics.update(disk)
            all_metrics.update(freq)

            sys_row = [ts]
            for key in sys_header[1:]:  # skip timestamp_ns
                val = all_metrics.get(key)
                sys_row.append(val if val is not None else "")
            
            w_sys.writerow(sys_row)
            f_sys.flush()

            time.sleep(args.proc_interval_s)

    # Wait for perf to finish (it should, because it runs "sleep duration").
    # If we were stopped early via SIGTERM, kill perf so we don't block.
    if _stop_requested:
        perf_proc.terminate()
    try:
        perf_proc.wait(timeout=max(5.0, args.duration_s + 5.0))
    except subprocess.TimeoutExpired:
        # If something went wrong, kill it so you don't leave perf running
        perf_proc.kill()
        perf_proc.wait(timeout=5.0)

    # Post-process perf -x CSV into a clean wide-format CSV.
    # Raw -x output has fields: timestamp,value,unit,event,...
    # We pivot to: t_s, event1, event2, ...
    perf_csv_out = os.path.join(args.out_dir, "perf_stat.csv")
    _postprocess_perf_csv(perf_out, perf_csv_out)

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
    print(f"  perf raw:  {perf_out}")
    print(f"  perf csv:  {perf_csv_out}")
    print(f"  proc:      {proc_out}")
    print(f"  proc_sys:  {proc_sys_out}")
    if args.collect_kernel_log:
        print(f"  klog: {klog_out}")
    print(f"  meta: {meta_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
