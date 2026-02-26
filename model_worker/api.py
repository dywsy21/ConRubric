"""Public decorators and device-selection helpers."""

from __future__ import annotations

import functools
import time
from typing import Optional

from model_worker.health import DeviceMonitor
from model_worker.warmup import device_scope


def best_device(
    memory_threshold_pct: float = 10.0,
    power_threshold_w: float = 100.0,
    fallback: str = "cuda:0",
) -> str:
    """Return ``"cuda:X"`` for the best idle device, or *fallback*."""
    try:
        mon = DeviceMonitor(memory_threshold_pct, power_threshold_w)
        all_devs = mon.query_all()
    except Exception:
        return fallback

    if not all_devs:
        return "cpu"

    ready = mon.ready_devices(all_devs)
    if ready:
        return f"cuda:{ready[0].index}"

    best = max(all_devs, key=lambda d: d.memory_free_mb)
    if best.memory_free_mb > 2048:
        return f"cuda:{best.index}"
    return fallback


def auto_devices(
    min_devices: int = 1,
    max_devices: Optional[int] = None,
    warmup: bool = True,
    reserve_mb: int = 1024,
    poll_interval: float = 30.0,
    timeout: Optional[float] = None,
    memory_threshold_pct: float = 10.0,
    power_threshold_w: float = 100.0,
    auto_release: bool = False,
    max_retries: int = 0,
    retry_delay: float = 120.0,
    verbose: bool = True,
):
    """Decorator: wait for *min_devices* idle devices then invoke function.

    Injects ``devices``, ``nproc``, ``device_slot`` keyword arguments.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    with device_scope(
                        min_devices=min_devices,
                        max_devices=max_devices,
                        warmup=warmup,
                        reserve_mb=reserve_mb,
                        poll_interval=poll_interval,
                        timeout=timeout,
                        memory_threshold_pct=memory_threshold_pct,
                        power_threshold_w=power_threshold_w,
                        auto_release=auto_release,
                        verbose=verbose,
                    ) as slot:
                        kwargs["devices"] = slot.devices
                        kwargs["nproc"] = slot.count
                        kwargs["device_slot"] = slot
                        return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt < max_retries:
                        print(f"[model_worker] attempt {attempt + 1} failed: {e}")
                        print(f"[model_worker] retrying in {retry_delay:.0f}s …")
                        time.sleep(retry_delay)
                    else:
                        raise
            raise last_err  # type: ignore[misc]
        return wrapper
    return decorator
