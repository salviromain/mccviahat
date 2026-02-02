#!/usr/bin/env python3
"""
proc_sampler.py

Sample once per interval:
- System CPU totals from /proc/stat
- A target process's CPU times from /proc/<pid>/stat

Writes a CSV time series.

Why jiffies?
- /proc reports CPU times in "jiffies" (kernel ticks), which are stable and cheap to read.
- You can convert to seconds later if needed: seconds = jiffies / USER_HZ
"""

import argparse
import csv
import os
import time
from typing import Tuple


def read_proc_stat_cpu() -> Tuple[int, int]:
    """
    Returns (total_jiffies, idle_jiffies) for overall CPU from /proc/stat line "cpu ...".
    Format: cpu user nice system idle iowait irq softirq steal guest guest_nice
    """
    with open("/proc/stat", "r", encoding="utf-8") as f:
        first = f.readline().strip()

    parts = first.split()
    if not parts or parts[0] != "cpu":
        raise RuntimeError("Unexpected /proc/stat format (first line not 'cpu ...')")

    # Convert all numeric fields after "cpu"
    values = list(map(int, parts[1:]))
    total = sum(values)

    # idle is field 4, iowait is field 5 in standard layout
    idle = values[3]
    if len(values) > 4:
        idle += values[4]

    return total, idle


def read_proc_pid_stat(pid: int) -> Tuple[int, int, int]:
    """
    Returns (utime_jiffies, stime_jiffies, rss_pages) from /proc/<pid>/stat.

    Note: /proc/<pid>/stat has a tricky second field "(comm)" which may contain spaces.
    We handle this by finding the last ')' and splitting after it.
    """
    path = f"/proc/{pid}/stat"
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()

    # Split safely after the comm field
    rparen = s.rfind(")")
    if rparen == -1:
        raise RuntimeError(f"Unexpected format in {path}: missing ')'")
    after = s[rparen + 2 :]  # skip ") "
    fields = after.split()

    # Field positions in "after" (0-indexed) correspond to original stat fields starting at #3.
    # Original fields:
    # 14 utime, 15 stime, 24 rss (in pages)
    # So in "after": utime is (14 - 3) = 11, stime is 12, rss is (24 - 3) = 21
    utime = int(fields[11])
    stime = int(fields[12])
    rss_pages = int(fields[21])

    return utime, stime, rss_pages


def pid_exists(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="Target host PID to sample (e.g., Docker container PID).")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path.")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0).")
    ap.add_argument("--duration", type=float, default=10.0, help="How long to sample in seconds (default: 10).")
    args = ap.parse_args()

    pid = args.pid
    if not pid_exists(pid):
        raise SystemExit(f"PID {pid} does not exist on this host. Did you pass the host PID?")

    t_end = time.time() + args.duration

    # CSV header
    header = [
        "timestamp_ns",
        "cpu_total_jiffies",
        "cpu_idle_jiffies",
        "pid",
        "proc_utime_jiffies",
        "proc_stime_jiffies",
        "proc_rss_pages",
    ]

    # Create parent directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        while time.time() < t_end:
            if not pid_exists(pid):
                # Process ended; stop sampling cleanly
                break

            ts = time.time_ns()
            cpu_total, cpu_idle = read_proc_stat_cpu()
            utime, stime, rss_pages = read_proc_pid_stat(pid)

            w.writerow([ts, cpu_total, cpu_idle, pid, utime, stime, rss_pages])
            f.flush()

            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
