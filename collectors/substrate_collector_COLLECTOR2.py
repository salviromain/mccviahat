#!/usr/bin/env python3
"""
collectors/substrate_collector.py

HAT-focused collector: captures the Hardware Anomaly Trace (HAT) and supporting
substrate metrics for an LLM inference server.  Tuned for Intel Xeon (c6420/c6620)
bare-metal nodes on CloudLab.

The HAT (§4.2.2 of the thesis) comprises two layers:

  Layer 1 — Hardware Anomaly Interrupts (named types, priority order):
    TLB Shootdown       → tlb:tlb_flush (perf, 1ms) + hat_TLB (/proc/interrupts)
    Machine Check       → mce:mce_record (probed) + hat_MCE (/proc/interrupts)
                          + full descriptor in kernel_log.txt
    Thermal/Power       → core_power.throttle (probed) + msr/cpu_thermal_margin/
    Spurious            → hat_SPU (/proc/interrupts)
    ECC correctable     → uncore_imc/UNC_M_ECC_CORRECTABLE_ERRORS/ (probed)
                          (uncorrectable ECC → escalates to MCE, already covered)

  Layer 2 — Continuous performance metrics (priority order):
    DRAM bandwidth      → uncore_imc/cas_count_read/ + cas_count_write/  ← NEW
                          (direct memory controller transaction counts;
                           64 bytes/CAS → multiply for bandwidth in bytes)
    Memory pressure     → dTLB-load-misses, LLC-load-misses, cache-misses,
                          cache-references  (CPU-side PMU proxies)
    Power consumption   → power/energy-pkg/, power/energy-ram/ (RAPL)
    Thermal readings    → msr/cpu_thermal_margin/ (continuous, °C below Tj_max)
    Clock variations    → cycles, instructions, stalled-cycles-frontend/backend
    Inference latency   → elapsed_ms in trial_meta.json (written by runner)

  Demoted / removed vs previous version:
    irq:irq_handler_entry/exit   — dropped: aggregate of all IRQ types, dominated
                                   by timer tick, not inference-specific
    irq:softirq_entry/exit/raise — dropped: OS scheduler noise, not HAT-relevant
    /proc/softirqs polling       — dropped: SCHED/RCU/TIMER rows not used in
                                   analysis; saves polling overhead

  Still collected but excluded from analysis (kept for completeness):
    hat_NMI, hat_PMI  — Performance Counter Overflow; counts perf's own activity

Three collection mechanisms:
  1. perf stat -a -I 1ms  (system-wide, 1ms buckets)
  2. /proc/interrupts + /sys/cpufreq polling (100ms) — HAT L1 named interrupts only
  3. journalctl -k slice  (post-trial, full window)

Outputs:
  perf_stat.txt       — raw perf stat output
  perf_stat.csv       — wide-format (1 row/ms, 1 col/event)
  proc_sample.csv     — per-process CPU + RSS (100ms)
  hat_interrupts.csv  — /proc/interrupts named rows + CPU freq (100ms)
                        NOTE: /proc/softirqs no longer included
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

    NOT CALLED IN MAIN LOOP — retained for ad-hoc use only.
    Softirq rows (SCHED, RCU, TIMER, NET_RX, BLOCK) are OS scheduling noise
    that is not used in the HAT analysis and not worth the polling overhead.
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

    def command(self, duration_s: float, out_path: str, cpu: str = "") -> List[str]:
        # If a CPU is specified, pin perf to that core via taskset.
        # perf -a still counts system-wide; taskset only pins the perf
        # *process itself* (its own scheduling + cache footprint).
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
    ap.add_argument("--proc_interval_s", type=float, default=0.1,
                    help="/proc sampling interval seconds (default: 0.1).")
    ap.add_argument("--events", type=str, default="",
                    help="Comma-separated perf events override (optional).")
    ap.add_argument("--llm_cpus", type=str, default="",
                    help="CPU cores reserved for the LLM, e.g. '0-11'. "
                         "If set, perf is pinned to --perf_cpu and the LLM "
                         "container is re-pinned via taskset.")
    ap.add_argument("--perf_cpu", type=str, default="",
                    help="CPU core(s) for perf stat itself, e.g. '12'. "
                         "Only used when --llm_cpus is set.")
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

    # ── CPU pinning (taskset isolation) ──────────────────────────────────────
    # If --llm_cpus is given, we:
    #   (a) re-pin the LLM container process to those cores via taskset
    #   (b) pin the perf stat process to --perf_cpu (a separate core)
    # This keeps perf's cache footprint off the LLM's cores.
    llm_cpus  = args.llm_cpus.strip()
    perf_cpu  = args.perf_cpu.strip()
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
    # HAT event set — tuned for Intel Xeon c6420 (Broadwell-EP)
    # ================================================================
    #
    # HAT Layer 1 — Hardware Anomaly Interrupts (tracepoints)
    #
    # tlb:tlb_flush
    #   TLB Shootdown: fires on every TLB flush (local + cross-CPU shootdown IPI).
    #   1ms resolution enables burst_clustering and lz_complexity on the time series.
    #   Cross-CPU shootdowns also visible in hat_TLB (/proc/interrupts, 100ms).
    #   Source: Linux include/trace/events/tlb.h; arch/x86/mm/tlb.c
    #
    # HAT Layer 2 — PMU hardware counters
    #
    # uncore_imc/cas_count_read/ + cas_count_write/
    #   DRAM read/write transaction counts from the memory controller directly.
    #   Each CAS = one 64-byte cache line transfer → multiply by 64 for bytes.
    #   More direct than LLC-load-misses (CPU-side proxy) or energy-ram (power proxy).
    #   Available on this node: confirmed via `perf list | grep uncore_imc`.
    #   Source: Intel SDM Vol.3B §2.3 uncore IMC PMU; perf list uncore_imc
    #
    # cache-misses / cache-references / LLC-load-misses
    #   CPU-side cache pressure: misses → DRAM fetches from weight tensor access.
    #   Complement the IMC CAS counters (CPU view vs controller view).
    #   Source: Intel SDM Vol.3B §18.3 LAST_LEVEL_CACHE_MISSES/REFERENCES
    #
    # dTLB-load-misses
    #   Data TLB load misses → hardware page table walk.
    #   Maps to DTLB_LOAD_MISSES.MISS_CAUSES_A_WALK.
    #   Weight tensors (~4GB) far exceed TLB capacity; this is directly inference-driven.
    #   Source: Intel SDM Vol.3B §18.3
    #
    # branch-misses / branch-instructions
    #   Pipeline flush on misprediction (~15 cycle penalty).
    #   Shaped by transformer control flow.
    #   Source: Intel SDM Vol.3B §18.3 BR_MISP_RETIRED.ALL_BRANCHES
    #
    # instructions / cycles
    #   IPC = instructions / cycles — direct computational throughput measure.
    #   Source: Intel SDM Vol.3B §18.2
    #
    # stalled-cycles-frontend / stalled-cycles-backend
    #   Frontend = I-cache/branch stalls; backend = memory latency stalls.
    #   Source: Intel SDM Vol.3B §18.3 IDQ_UOPS_NOT_DELIVERED / CYCLE_ACTIVITY
    #
    # HAT Layer 2 — Power / thermal
    #
    # power/energy-pkg/ / power/energy-ram/
    #   RAPL package and DRAM energy in Joules/interval.
    #   Source: Intel SDM Vol.3B §14.9; Linux arch/x86/events/rapl.c
    #
    # msr/cpu_thermal_margin/
    #   °C below Tj_max (throttle threshold) from IA32_THERM_STATUS MSR bits [22:16].
    #   Slow-moving; used as confounder check rather than primary signal.
    #   Source: Intel SDM Vol.3B §14.7.2
    #
    # Supporting (confounders — collected for normalisation, not primary signals)
    #
    # context-switches  — scheduling artefact
    # cpu-migrations    — task moved between CPUs by load balancer
    # page-faults       — minor (mmap remap) + major (page-in)
    # cpu-clock         — CPU time; duration normaliser
    # sched:sched_stat_wait — per-task wait time in run queue (scheduler delay)

    hat_events = [
        # HAT Layer 1 — TLB Shootdown tracepoint
        "tlb:tlb_flush",

        # HAT Layer 2 — DRAM bandwidth (memory controller, direct measurement)
        "uncore_imc/cas_count_read/",
        "uncore_imc/cas_count_write/",

        # HAT Layer 2 — CPU-side memory pressure PMU counters
        "cache-misses",
        "cache-references",
        "LLC-load-misses",
        "dTLB-load-misses",

        # HAT Layer 2 — IPC / pipeline
        "branch-misses",
        "branch-instructions",
        "instructions",
        "cycles",
        "stalled-cycles-frontend",
        "stalled-cycles-backend",

        # HAT Layer 2 — power / thermal
        "msr/cpu_thermal_margin/",
        "power/energy-pkg/",
        "power/energy-ram/",

        # Supporting / confounders
        "context-switches",
        "cpu-migrations",
        "page-faults",
        "cpu-clock",
    ]

    # Probe optional HAT Layer 1 events — depend on kernel config / CPU model
    probed_events: List[str] = []
    optional_hat_events = [
        # Machine Check Exception — named HAT Layer 1 anomaly
        # Full MCE descriptor also in kernel_log.txt (bank, status, address, misc)
        # Source: Linux arch/x86/kernel/cpu/mcheck/mce.c
        ("mce:mce_record",
         "Machine Check Exception — hardware fault record"),

        # Thermal/Power Anomaly — named HAT Layer 1 anomaly
        # Fires when PROCHOT or power limit forces frequency reduction
        # Source: Intel SDM Vol.3B §14.7; Linux arch/x86/events/intel/core.c
        ("core_power.throttle",
         "Thermal/Power Anomaly — PROCHOT/power-limit throttle event"),

        # ECC correctable errors — named HAT anomaly (single-bit DRAM bit-flips
        # corrected silently by ECC hardware; never generate MCE)
        # Available on this node via uncore IMC PMU (confirmed via perf list)
        # Source: Intel SDM Vol.3B §2.3 uncore IMC; UNC_M_ECC_CORRECTABLE_ERRORS
        ("uncore_imc/UNC_M_ECC_CORRECTABLE_ERRORS/",
         "ECC correctable errors — silent single-bit DRAM errors"),

        # Scheduler wait-time tracepoint (deadline/scheduling pressure confounder)
        # Captures time tasks spend waiting in run queue before being scheduled.
        ("sched:sched_stat_wait",
         "Scheduler wait-time tracepoint — run-queue wait delay"),
    ]

    for evt, description in optional_hat_events:
        if _probe_event(evt):
            probed_events.append(evt)
            print(f"  + {evt}  ({description})")
        else:
            print(f"  x {evt}  ({description}) — not available on this kernel, skipped")

    events = ([e.strip() for e in args.events.split(",") if e.strip()]
              if args.events else hat_events)
    events = events + probed_events
    plan = PerfPlan(interval_ms=args.perf_interval_ms, events=events)

    # Output paths
    perf_out = os.path.join(args.out_dir, "perf_stat.txt")
    proc_out = os.path.join(args.out_dir, "proc_sample.csv")
    hat_irq_out = os.path.join(args.out_dir, "hat_interrupts.csv")  # /proc/interrupts + CPU freq only
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
        "llm_cpus": llm_cpus,
        "perf_cpu": perf_cpu,
        "taskset_enabled": use_taskset,
        "perf_events": events,
        "probed_events": probed_events,
        "perf_command": plan.command(args.duration_s, perf_out, cpu=perf_cpu if use_taskset else ""),
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Start perf stat in background.
    # start_new_session=True puts perf in its own process group so that
    # SIGTERM sent to the collector by run_prompts_json.py doesn't kill
    # perf before it can flush its -o output file.
    perf_cmd = plan.command(args.duration_s, perf_out, cpu=perf_cpu if use_taskset else "")
    perf_log = os.path.join(args.out_dir, "perf_stderr.log")
    with open(perf_log, "w") as perf_log_f:
        perf_proc = subprocess.Popen(perf_cmd, stdout=subprocess.DEVNULL,
                                     stderr=perf_log_f,
                                     start_new_session=True)

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
    # /proc/softirqs intentionally excluded — OS scheduling noise, not HAT-relevant
    sample_interrupts = read_proc_interrupts()
    sample_freq       = read_cpu_frequencies()

    hat_header = ["timestamp_ns"]
    hat_header += sorted(sample_interrupts.keys())
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

            # HAT interrupt counts + CPU frequency
            # /proc/softirqs intentionally not polled (OS noise, not HAT-relevant)
            interrupts = read_proc_interrupts()
            freq       = read_cpu_frequencies()

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

    # Stop perf gracefully.  SIGINT makes perf flush its -o output;
    # SIGTERM may cause it to exit without writing the file.
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

    # Post-process perf CSV (only if perf actually produced output)
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