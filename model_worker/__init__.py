"""model_worker — Model serving health-check & device scheduling utilities.

Pre-flight device validation, warm-up and readiness probes for distributed
training and vLLM / sglang model serving on multi-GPU nodes.
"""

import sys as _sys

# Ensure all [model_worker] log messages are visible immediately even when
# stdout is piped (e.g. through tee).  Without this, Python uses full
# buffering on pipes and messages only appear once the buffer fills up.
try:
    _sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from model_worker.health import DeviceMonitor, DeviceInfo  # noqa: F401
from model_worker.warmup import DeviceWarmup, DeviceSlot, device_scope  # noqa: F401
from model_worker.api import auto_devices, best_device  # noqa: F401
from model_worker.elastic import elastic_run  # noqa: F401

__all__ = [
    "DeviceMonitor",
    "DeviceInfo",
    "DeviceWarmup",
    "DeviceSlot",
    "device_scope",
    "auto_devices",
    "best_device",
    "elastic_run",
]
