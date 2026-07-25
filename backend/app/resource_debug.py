import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


NVIDIA_SMI = shutil.which("nvidia-smi")


def _kilobytes_to_megabytes(value: int) -> float:
    return round(value / 1024, 1)


def _proc_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator:
                continue
            parts = raw_value.strip().split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0])
    except OSError:
        return {}
    return values


def _gpu_memory() -> dict[str, Any]:
    if not NVIDIA_SMI:
        return {"available": False}
    try:
        completed = subprocess.run(
            [
                NVIDIA_SMI,
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=1,
        )
        rows = [
            [int(value.strip()) for value in line.split(",")]
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"available": False}
    if not rows:
        return {"available": False}
    return {
        "available": True,
        "used_mb": sum(row[0] for row in rows),
        "total_mb": sum(row[1] for row in rows),
        "device_count": len(rows),
    }


def resource_snapshot() -> dict[str, Any]:
    process = _proc_values(Path("/proc/self/status"))
    system = _proc_values(Path("/proc/meminfo"))
    ram: dict[str, Any] = {}
    if "VmRSS" in process:
        ram["backend_rss_mb"] = _kilobytes_to_megabytes(process["VmRSS"])
    if "VmHWM" in process:
        ram["backend_peak_rss_mb"] = _kilobytes_to_megabytes(process["VmHWM"])
    if "MemTotal" in system:
        total = system["MemTotal"]
        available = system.get("MemAvailable", system.get("MemFree", 0))
        ram.update(
            {
                "system_used_mb": _kilobytes_to_megabytes(total - available),
                "system_available_mb": _kilobytes_to_megabytes(available),
                "system_total_mb": _kilobytes_to_megabytes(total),
            }
        )
    return {
        "backend_pid": os.getpid(),
        "ram": ram,
        "vram": _gpu_memory(),
    }


def resource_delta(start: dict[str, Any], end: dict[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for section, fields in (
        ("ram", ("backend_rss_mb", "system_used_mb")),
        ("vram", ("used_mb",)),
    ):
        start_section = start.get(section, {})
        end_section = end.get(section, {})
        for field in fields:
            start_value = start_section.get(field)
            end_value = end_section.get(field)
            if isinstance(start_value, (int, float)) and isinstance(end_value, (int, float)):
                delta[f"{section}.{field}"] = round(end_value - start_value, 1)
    return delta
