"""Pre-flight safety checks before launching GraspNet training.

This is a read-only gate. It reports per-GPU usage, free disk on the
shared volume, and free RAM, then decides whether it is safe to launch.
It never changes any state. The training launcher runs this first and
aborts if it fails.

The container has no resource isolation and shares its disk with the
host, so these checks exist to avoid harming the host or other users.
"""

import argparse
import shutil
import subprocess
import sys

# Conservative defaults. The disk floor is the single most important
# guard because the volume is shared with the host.
abort_floor_gb = 80
warn_floor_gb = 150
min_ram_gb = 16
# A GPU holding more than this many MiB is treated as "in use by someone".
gpu_busy_mib = 1024


def query_gpus():
    """Return per-GPU usage as a list of dicts via ``nvidia-smi``.

    Returns:
        A list with one dict per GPU holding ``index`` (int),
        ``mem_used_mib`` (int) and ``util_pct`` (int). Empty if
        ``nvidia-smi`` is unavailable.
    """
    fields = "index,memory.used,utilization.gpu"
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        index, mem, util = (part.strip() for part in line.split(","))
        gpus.append(
            {
                "index": int(index),
                "mem_used_mib": int(mem),
                "util_pct": int(util),
            }
        )
    return gpus


def parse_gpu_arg(value, gpus):
    """Resolve the ``--gpu`` argument into a list of GPU indices.

    Args:
        value: Either ``"all"`` or a comma-separated list such as
            ``"0"`` or ``"0,1"``.
        gpus: The list returned by :func:`query_gpus`.

    Returns:
        The selected GPU indices as a list of ints.
    """
    if value == "all":
        return [g["index"] for g in gpus]
    return [int(token) for token in value.split(",") if token != ""]


def free_disk_gb(path):
    """Return the free space at ``path`` in gibibytes."""
    return shutil.disk_usage(path).free / 1024**3


def free_ram_gb():
    """Return available RAM in gibibytes from ``/proc/meminfo``."""
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    return 0.0


def main():
    """Run all checks and exit non-zero if launching is unsafe."""
    parser = argparse.ArgumentParser(description="GraspNet pre-flight check")
    parser.add_argument(
        "--gpu",
        required=True,
        help="GPU index, comma list (e.g. '0' or '0,1'), or 'all'.",
    )
    parser.add_argument("--disk-path", default="/workspace")
    parser.add_argument("--abort-floor-gb", type=float, default=abort_floor_gb)
    parser.add_argument(
        "--allow-busy",
        action="store_true",
        help="Treat an in-use GPU as a warning, not an abort (shared host).",
    )
    args = parser.parse_args()

    problems = []

    gpus = query_gpus()
    print("== GPUs ==")
    if not gpus:
        print("  nvidia-smi unavailable")
        problems.append("nvidia-smi unavailable")
    for gpu in gpus:
        print(
            f"  GPU {gpu['index']}: {gpu['mem_used_mib']} MiB used, "
            f"{gpu['util_pct']}% util"
        )

    selected = parse_gpu_arg(args.gpu, gpus)
    print(f"\nSelected GPUs: {selected}")
    by_index = {g["index"]: g for g in gpus}
    for index in selected:
        gpu = by_index.get(index)
        if gpu is None:
            problems.append(f"GPU {index} not present")
        elif gpu["mem_used_mib"] > gpu_busy_mib:
            msg = f"GPU {index} already in use ({gpu['mem_used_mib']} MiB)"
            if args.allow_busy:
                print(f"  WARN: {msg} (shared use allowed)")
            else:
                problems.append(msg)

    free_gb = free_disk_gb(args.disk_path)
    print(f"\nFree disk on {args.disk_path}: {free_gb:.0f} GB")
    if free_gb < args.abort_floor_gb:
        problems.append(
            f"free disk {free_gb:.0f} GB < floor {args.abort_floor_gb:.0f} GB"
        )
    elif free_gb < warn_floor_gb:
        print(f"  WARN: below {warn_floor_gb} GB soft limit")

    ram_gb = free_ram_gb()
    print(f"Free RAM: {ram_gb:.0f} GB")
    if ram_gb < min_ram_gb:
        problems.append(f"free RAM {ram_gb:.0f} GB < {min_ram_gb} GB")

    print()
    if problems:
        print("ABORT: " + "; ".join(problems))
        sys.exit(1)
    print("SAFE TO LAUNCH")
    sys.exit(0)


if __name__ == "__main__":
    main()
