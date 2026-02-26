"""Device health probes via nvidia-smi XML — no CUDA init in caller.

Runs ``nvidia-smi -q -x`` to inspect device memory, power and thermal state
without creating a CUDA context in the calling process.
"""

from __future__ import annotations

import subprocess
import sys
import time
import xml.etree.ElementTree as ET

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeviceInfo:
    """Snapshot of one physical accelerator."""

    index: int
    name: str = ""
    uuid: str = ""
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    utilization_pct: int = 0
    temperature_c: int = 0
    pids: List[int] = field(default_factory=list)

    @property
    def memory_used_pct(self) -> float:
        return self.memory_used_mb / self.memory_total_mb * 100.0 if self.memory_total_mb > 0 else 100.0

    @property
    def memory_free_pct(self) -> float:
        return 100.0 - self.memory_used_pct

    def is_ready(self, mem_pct: float = 10.0, power_w: float = 100.0) -> bool:
        """Device is ready (idle) when memory usage AND power draw are both low."""
        return self.memory_used_pct < mem_pct and self.power_draw_w < power_w

    def __repr__(self) -> str:
        status = "ready" if self.is_ready() else "busy"
        return (
            f"DeviceInfo({self.index}, {self.name!r}, "
            f"mem={self.memory_used_mb}/{self.memory_total_mb}MiB "
            f"({self.memory_used_pct:.0f}%), "
            f"pwr={self.power_draw_w:.0f}W, {status})"
        )


# ── XML helpers ──────────────────────────────────────────────────────────

def _int(text: Optional[str], default: int = 0) -> int:
    if not text:
        return default
    try:
        return int(text.strip().split()[0])
    except (ValueError, IndexError):
        return default


def _float(text: Optional[str], default: float = 0.0) -> float:
    if not text:
        return default
    try:
        return float(text.strip().split()[0])
    except (ValueError, IndexError):
        return default


def _txt(parent, *tags) -> Optional[str]:
    node = parent
    for t in tags:
        if node is None:
            return None
        node = node.find(t)
    return node.text if node is not None else None


# ── Main monitor ─────────────────────────────────────────────────────────

class DeviceMonitor:
    """Query accelerator states via ``nvidia-smi``.  Zero CUDA init."""

    def __init__(self, memory_threshold_pct: float = 10.0, power_threshold_w: float = 100.0):
        self.memory_threshold_pct = memory_threshold_pct
        self.power_threshold_w = power_threshold_w

    def query_all(self) -> List[DeviceInfo]:
        try:
            r = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=15)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip())
        except FileNotFoundError:
            raise RuntimeError("nvidia-smi not found")

        root = ET.fromstring(r.stdout)
        devices: List[DeviceInfo] = []

        for i, g in enumerate(root.findall("gpu")):
            mem = g.find("fb_memory_usage")
            pwr = g.find("gpu_power_readings") or g.find("power_readings")
            util = g.find("utilization")
            temp = g.find("temperature")
            procs = g.find("processes")

            pids: List[int] = []
            if procs is not None:
                for pi in procs.findall("process_info"):
                    pt = _txt(pi, "pid")
                    if pt:
                        try:
                            pids.append(int(pt.strip()))
                        except ValueError:
                            pass

            pw_limit = 0.0
            if pwr is not None:
                for tag in ("current_power_limit", "power_limit", "default_power_limit"):
                    v = _float(_txt(pwr, tag))
                    if v > 0:
                        pw_limit = v
                        break

            devices.append(DeviceInfo(
                index=i,
                name=(_txt(g, "product_name") or "").strip(),
                uuid=(_txt(g, "uuid") or "").strip(),
                memory_used_mb=_int(_txt(mem, "used") if mem else None),
                memory_total_mb=_int(_txt(mem, "total") if mem else None),
                memory_free_mb=_int(_txt(mem, "free") if mem else None),
                power_draw_w=_float(_txt(pwr, "power_draw") if pwr else None),
                power_limit_w=pw_limit,
                utilization_pct=_int(_txt(util, "gpu_util") if util else None),
                temperature_c=_int(_txt(temp, "gpu_temp") if temp else None),
                pids=pids,
            ))
        return devices

    def ready_devices(self, devices: Optional[List[DeviceInfo]] = None) -> List[DeviceInfo]:
        if devices is None:
            devices = self.query_all()
        ready = [d for d in devices if d.is_ready(self.memory_threshold_pct, self.power_threshold_w)]
        ready.sort(key=lambda d: d.memory_free_mb, reverse=True)
        return ready

    def wait_until_ready(
        self,
        min_count: int = 1,
        max_count: Optional[int] = None,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None,
        verbose: bool = True,
    ) -> List[DeviceInfo]:
        """Block until *min_count* devices pass readiness probe."""
        if max_count is None:
            max_count = min_count

        start = time.time()
        attempt = 0

        while True:
            attempt += 1
            all_devs = self.query_all()
            ready = self.ready_devices(all_devs)
            elapsed = time.time() - start

            if verbose:
                tag = "✓" if len(ready) >= min_count else f"waiting for {min_count}"
                print(
                    f"[model_worker] health-check #{attempt}: "
                    f"{len(ready)}/{len(all_devs)} devices ready ({tag}) "
                    f"[{elapsed:.0f}s]"
                )

            if len(ready) >= min_count:
                selected = ready[:max_count]
                if verbose:
                    print(f"[model_worker] selected devices: {[d.index for d in selected]}")
                return selected

            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(
                    f"Only {len(ready)} devices ready after {timeout:.0f}s (need {min_count})"
                )

            if verbose:
                print(f"[model_worker] next probe in {poll_interval:.0f}s …")
            time.sleep(poll_interval)

    def summary(self) -> str:
        devices = self.query_all()
        hdr = f"{'Dev':>3}  {'Name':<22}  {'Mem Used':>10}  {'Mem Total':>10}  {'Used%':>6}  {'Power':>7}  {'Util':>5}  {'Status':>6}"
        sep = "─" * len(hdr)
        lines = [hdr, sep]
        for d in devices:
            mark = " READY" if d.is_ready(self.memory_threshold_pct, self.power_threshold_w) else "  BUSY"
            lines.append(
                f"{d.index:>3}  {d.name:<22}  {d.memory_used_mb:>7} MiB  {d.memory_total_mb:>7} MiB  "
                f"{d.memory_used_pct:>5.1f}%  {d.power_draw_w:>5.0f} W  {d.utilization_pct:>4}%  {mark}"
            )
        n_ready = len(self.ready_devices(devices))
        lines.append(sep)
        lines.append(f"Ready: {n_ready}/{len(devices)}")
        return "\n".join(lines)
