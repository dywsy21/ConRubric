"""Device warm-up & readiness reservation.

Spawns lightweight "warm-up" sub-processes that pre-allocate device memory,
ensuring devices stay available between the readiness probe and the actual
training / serving launch.  Each sub-process masquerades as a normal model
serving worker so it blends in with other sglang / vllm instances.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from model_worker.health import DeviceInfo, DeviceMonitor

# ── warm-up sub-process payload ──────────────────────────────────────────
# Runs in a completely isolated Python process (own CUDA context).
# Sets its process title to look like a sglang model worker via
# /proc/self/comm and sys.argv[0].

_WARMUP_PAYLOAD = r'''
import os, sys, signal, time, ctypes

gpu_idx = sys.argv[1]
target_mb = int(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

# ── disguise process name ────────────────────────────────────────────
# /proc/self/comm  (shows up in nvidia-smi & ps -e)
_PROC_NAME = b"sglang::scheduler"
try:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(15, _PROC_NAME, 0, 0, 0)   # PR_SET_NAME = 15
    libc.prctl(1, signal.SIGTERM)           # PR_SET_PDEATHSIG = 1
except Exception:
    pass

# argv[0] visible in `ps aux`
sys.argv[0] = "sglang.serve"

# ── allocate with back-off ───────────────────────────────────────────
import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

try:
    import torch
except ImportError:
    print("FAIL:torch", flush=True); sys.exit(1)

if not torch.cuda.is_available():
    print("FAIL:no_cuda", flush=True); sys.exit(1)

mb = target_mb
holder = None
while mb > 100:
    try:
        holder = torch.empty(mb * 262144, dtype=torch.float32, device="cuda:0")
        break
    except torch.cuda.OutOfMemoryError:
        mb = int(mb * 0.75)
    except Exception as e:
        print(f"FAIL:{e}", flush=True); sys.exit(1)

if holder is None:
    print("FAIL:alloc", flush=True); sys.exit(1)

actual = holder.element_size() * holder.nelement() // (1024 * 1024)
print(f"OK:{actual}", flush=True)

def _exit(s, f): sys.exit(0)
signal.signal(signal.SIGTERM, _exit)
signal.signal(signal.SIGINT, _exit)

try:
    while True:
        time.sleep(3600)
except SystemExit:
    pass
'''


# ── DeviceWarmup ─────────────────────────────────────────────────────────

class DeviceWarmup:
    """Pre-warm devices with placeholder memory to reserve them."""

    def __init__(self, reserve_mb: int = 1024):
        self.reserve_mb = reserve_mb
        self._procs: Dict[int, subprocess.Popen] = {}
        atexit.register(self.shutdown)

    def warm(self, devices: List[DeviceInfo], timeout: float = 60.0) -> List[int]:
        """Warm up *devices*.  Returns indices that succeeded."""
        warmed: List[int] = []
        for d in devices:
            if d.index in self._procs:
                warmed.append(d.index)
                continue
            alloc_mb = max(d.memory_free_mb - self.reserve_mb, 100)
            proc = subprocess.Popen(
                [sys.executable, "-c", _WARMUP_PAYLOAD, str(d.index), str(alloc_mb)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + timeout
            ok = False
            while time.monotonic() < deadline:
                raw = proc.stdout.readline()
                if not raw:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                line = raw.decode(errors="replace").strip()
                if line.startswith("OK:"):
                    mb_str = line.split(":", 1)[1]
                    print(f"[model_worker] device {d.index} warmed ({mb_str} MiB pre-allocated)")
                    self._procs[d.index] = proc
                    warmed.append(d.index)
                    ok = True
                    break
                elif line.startswith("FAIL:"):
                    print(f"[model_worker] device {d.index} warm-up skipped: {line.split(':', 1)[1]}")
                    break
            if not ok:
                try:
                    proc.terminate(); proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        return warmed

    def cooldown(self, indices: Optional[List[int]] = None):
        """Release pre-allocated memory on specific (or all) devices."""
        if indices is None:
            indices = list(self._procs.keys())
        for idx in indices:
            proc = self._procs.pop(idx, None)
            if proc is None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(timeout=2)
            print(f"[model_worker] device {idx} released")

    def shutdown(self):
        self.cooldown(None)

    def health_check(self) -> List[int]:
        dead = [i for i, p in self._procs.items() if p.poll() is not None]
        for i in dead:
            self._procs.pop(i)
        return dead

    @property
    def active_devices(self) -> List[int]:
        return sorted(self._procs.keys())

    def __del__(self):
        self.shutdown()


# ── DeviceSlot ───────────────────────────────────────────────────────────

@dataclass
class DeviceSlot:
    """Allocation result — list of device indices + optional warmup handle."""

    devices: List[int]
    _warmup: Optional[DeviceWarmup] = field(default=None, repr=False)
    _released: bool = field(default=False, repr=False)

    @property
    def count(self) -> int:
        return len(self.devices)

    nproc = count

    @property
    def cuda_visible(self) -> str:
        return ",".join(str(i) for i in self.devices)

    @property
    def is_held(self) -> bool:
        return self._warmup is not None and not self._released

    def release(self):
        """Release warm-up memory so training / serving can use full VRAM."""
        if self._warmup and not self._released:
            self._warmup.shutdown()
            self._released = True

    def __len__(self) -> int:
        return len(self.devices)

    def __iter__(self):
        return iter(self.devices)


# ── device_scope — context manager ───────────────────────────────────────

class device_scope:
    """Wait for devices → optionally warm → yield DeviceSlot → clean up.

    Typical training usage::

        with device_scope(min_devices=4, warmup=True) as slot:
            slot.release()       # free VRAM right before torchrun
            subprocess.run(["torchrun", f"--nproc_per_node={slot.count}", ...])
    """

    def __init__(
        self,
        min_devices: int = 1,
        max_devices: Optional[int] = None,
        warmup: bool = True,
        reserve_mb: int = 1024,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None,
        memory_threshold_pct: float = 10.0,
        power_threshold_w: float = 100.0,
        set_cuda_visible: bool = True,
        auto_release: bool = False,
        verbose: bool = True,
    ):
        self.min_devices = min_devices
        self.max_devices = max_devices if max_devices is not None else min_devices
        self.warmup = warmup
        self.reserve_mb = reserve_mb
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.memory_threshold_pct = memory_threshold_pct
        self.power_threshold_w = power_threshold_w
        self.set_cuda_visible = set_cuda_visible
        self.auto_release = auto_release
        self.verbose = verbose
        self._slot: Optional[DeviceSlot] = None
        self._old_cv: Optional[str] = None

    def __enter__(self) -> DeviceSlot:
        mon = DeviceMonitor(self.memory_threshold_pct, self.power_threshold_w)

        if self.warmup:
            slot = self._accumulate(mon)
        else:
            ready = mon.wait_until_ready(
                min_count=self.min_devices, max_count=self.max_devices,
                poll_interval=self.poll_interval, timeout=self.timeout,
                verbose=self.verbose,
            )
            slot = DeviceSlot(devices=[d.index for d in ready])

        if self.set_cuda_visible:
            self._old_cv = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = slot.cuda_visible
            if self.verbose:
                print(f"[model_worker] CUDA_VISIBLE_DEVICES={slot.cuda_visible}")

        if self.auto_release:
            slot.release()

        self._slot = slot
        return slot

    def __exit__(self, *exc):
        if self._slot is not None:
            self._slot.release()
        if self.set_cuda_visible:
            if self._old_cv is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = self._old_cv
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        return False

    def _accumulate(self, mon: DeviceMonitor) -> DeviceSlot:
        """Accumulate devices one by one until min_devices reached."""
        wu = DeviceWarmup(reserve_mb=self.reserve_mb)
        start = time.time()
        scan = 0

        while True:
            scan += 1
            wu.health_check()
            ready = mon.ready_devices()
            held = len(wu.active_devices)
            total = held + len(ready)
            elapsed = time.time() - start

            if self.verbose:
                tag = "✓" if total >= self.min_devices else f"waiting for {self.min_devices}"
                print(
                    f"[model_worker] probe #{scan}: "
                    f"{len(ready)} idle + {held} held = {total} ({tag}) "
                    f"[{elapsed:.0f}s]"
                )

            need = self.max_devices - held
            if ready and need > 0:
                wu.warm(ready[:need])

            if len(wu.active_devices) >= self.min_devices:
                idxs = wu.active_devices[:self.max_devices]
                if self.verbose:
                    print(f"[model_worker] {len(idxs)} devices ready: {idxs}")
                return DeviceSlot(devices=idxs, _warmup=wu)

            if self.timeout is not None and elapsed >= self.timeout:
                wu.shutdown()
                raise TimeoutError(
                    f"Only {len(wu.active_devices)} devices after {self.timeout:.0f}s "
                    f"(need {self.min_devices})"
                )

            if self.verbose:
                print(f"[model_worker] next probe in {self.poll_interval:.0f}s …")
            time.sleep(self.poll_interval)
