"""Hardware inspection. Records only what is measurable on the actual machine."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _total_ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        if sys.platform == "linux" and Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
            return int(out.stdout.strip())
    except Exception:
        return None
    return None


def _available_ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        if sys.platform == "linux" and Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
            page = 4096
            free = 0
            for line in out.stdout.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split()[0])
                if line.startswith("Pages free") or line.startswith("Pages inactive"):
                    free += int(line.split(":")[1].strip().rstrip(".")) * page
            return free
    except Exception:
        return None
    return None


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=20
            )
            names = [line.strip() for line in out.stdout.splitlines() if line.strip() and line.strip() != "Name"]
            if names:
                return names[0]
        if sys.platform == "linux" and Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=20
            )
            if out.stdout.strip():
                return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _gpu_info() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free = total = None
            try:
                free, total = torch.cuda.mem_get_info(index)
            except Exception:
                pass
            gpus.append(
                {
                    "backend": "cuda",
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(total or props.total_memory),
                    "free_memory_bytes": int(free) if free is not None else None,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "torch_bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    if hasattr(torch.backends, "mps"):
        try:
            if torch.backends.mps.is_available():
                gpus.append({"backend": "mps", "name": "apple-mps", "total_memory_bytes": None,
                             "note": "Apple unified memory; not equivalent to dedicated VRAM"})
        except Exception:
            pass
    return gpus


def inspect_hardware(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or Path.cwd())
    usage = shutil.disk_usage(root)
    return {
        "captured_on": platform.node(),
        "captured_at_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "default_threads": torch.get_num_threads(),
        },
        "cpu": {
            "name": _cpu_name(),
            "logical_threads": os.cpu_count(),
            "high_precision_bf16_cpu": _bf16_cpu_ok(),
        },
        "memory": {
            "total_ram_bytes": _total_ram_bytes(),
            "available_ram_bytes": _available_ram_bytes(),
        },
        "gpus": _gpu_info(),
        "disk": {
            "path": str(root.resolve()),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        },
    }


def peak_rss_bytes() -> int | None:
    """Best-effort peak resident set size of this process."""
    try:
        if sys.platform == "win32":
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            for library, name in (("kernel32", "K32GetProcessMemoryInfo"), ("psapi", "GetProcessMemoryInfo")):
                try:
                    function = getattr(getattr(ctypes.windll, library), name)
                except AttributeError:
                    continue
                if function(process, ctypes.byref(counters), counters.cb):
                    return int(counters.PeakWorkingSetSize)
        else:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return int(usage.ru_maxrss * (1024 if sys.platform != "darwin" else 1))
    except Exception:
        return None
    return None


def _bf16_cpu_ok() -> bool:
    try:
        x = torch.ones(4, dtype=torch.bfloat16)
        y = (x + x).float().sum().item()
        return y == 8.0
    except Exception:
        return False


def resolve_device(requested: str) -> torch.device:
    """Resolve 'auto'/'cpu'/'cuda'/'mps' and refuse unavailable requests loudly."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    if requested not in ("cpu", "mps") and not requested.startswith("cuda"):
        raise RuntimeError(f"unsupported device: {requested}")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> str:
    """Resolve precision and never silently downgrade a request."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        # FP32 on CPU: the project's CPU-first baseline.
        return "fp32"
    if requested in ("fp32", "fp16", "bf16"):
        if requested == "bf16" and device.type == "cpu" and not _bf16_cpu_ok():
            raise RuntimeError("bf16 requested on a CPU without working bf16 support")
        if requested == "fp16" and device.type == "cpu":
            raise RuntimeError("fp16 autocast is not supported on CPU in this project")
        return requested
    raise RuntimeError(f"unsupported precision: {requested}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the actual target hardware")
    parser.add_argument("--json", default=None, help="write report to this path")
    args = parser.parse_args()
    info = inspect_hardware()
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
