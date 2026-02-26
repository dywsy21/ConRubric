"""Elastic device scaling for distributed training.

Monitors idle devices while training runs.  After each checkpoint save,
checks whether new devices became available.  If so, gracefully restarts
training to utilise the expanded device set.  Resume-from-checkpoint is
handled by the underlying trainer (verl SFT / PPO).

Typical SFT usage::

    from model_worker.elastic import elastic_run

    def launch(nproc, cuda_visible):
        return subprocess.Popen(["torchrun", f"--nproc_per_node={nproc}", ...])

    elastic_run(launch, "out/sft/latest_checkpointed_iteration.txt",
                min_devices=4, max_devices=8)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Callable, List, Optional

from model_worker.health import DeviceMonitor
from model_worker.warmup import device_scope

# Ensure log messages flush immediately when stdout is a pipe.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ── helpers ──────────────────────────────────────────────────────────────

def _file_mtime(path: str) -> float:
    """Return mtime of *path*, or 0.0 if it does not exist."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _read_step(path: str) -> str:
    """Read checkpoint step number from indicator file."""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "?"


# ── expansion monitor ────────────────────────────────────────────────────

def _watch_for_expansion(
    proc: subprocess.Popen,
    current_devices: List[int],
    max_devices: int,
    checkpoint_indicator: str,
    check_interval: float,
    memory_threshold_pct: float,
    power_threshold_w: float,
    verbose: bool,
) -> bool:
    """Monitor running training for GPU expansion opportunities.

    Returns ``True`` if training was terminated for restart with more
    devices.  Returns ``False`` if training exited on its own.
    """
    if len(current_devices) >= max_devices:
        # Already at max capacity — just wait for normal exit.
        proc.wait()
        return False

    mon = DeviceMonitor(memory_threshold_pct, power_threshold_w)
    current_set = set(current_devices)
    last_ckpt_mtime = _file_mtime(checkpoint_indicator)

    while True:
        # Sleep in 1-second ticks so we notice early exits quickly.
        elapsed = 0.0
        while elapsed < check_interval:
            time.sleep(1.0)
            elapsed += 1.0
            if proc.poll() is not None:
                return False  # Training exited

        # ── Training still running — check for expansion ────────────
        new_mtime = _file_mtime(checkpoint_indicator)
        if new_mtime <= last_ckpt_mtime:
            continue  # No new checkpoint since last check
        last_ckpt_mtime = new_mtime
        step = _read_step(checkpoint_indicator)

        # New checkpoint!  Probe for idle devices not used by training.
        try:
            idle_new = [
                d for d in mon.ready_devices()
                if d.index not in current_set
            ]
        except Exception:
            continue

        if not idle_new:
            if verbose:
                print(
                    f"[model_worker] elastic: checkpoint at step {step}, "
                    f"no new idle devices"
                )
            continue

        new_total = min(len(current_devices) + len(idle_new), max_devices)
        if new_total <= len(current_devices):
            continue

        if verbose:
            idle_idx = [d.index for d in idle_new]
            print(
                f"[model_worker] elastic: checkpoint at step {step} — "
                f"{len(idle_new)} new idle devices: {idle_idx}"
            )
            print(
                f"[model_worker] elastic: scaling "
                f"{len(current_devices)} → {new_total} devices, "
                f"restarting from checkpoint …"
            )

        # ── Graceful terminate ──────────────────────────────────────
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        return True


# ── public API ───────────────────────────────────────────────────────────

def elastic_run(
    launch_fn: Callable[[int, str], subprocess.Popen],
    checkpoint_indicator: str,
    min_devices: int = 1,
    max_devices: int = 8,
    check_interval: float = 60.0,
    warmup: bool = True,
    reserve_mb: int = 1024,
    device_poll_interval: float = 30.0,
    memory_threshold_pct: float = 10.0,
    power_threshold_w: float = 100.0,
    max_crashes: int = 3,
    crash_cooldown: float = 30.0,
    verbose: bool = True,
) -> int:
    """Run training with elastic GPU scaling.

    Workflow
    -------
    1. Wait for ≥ *min_devices* idle GPUs (via :func:`device_scope`).
    2. Call ``launch_fn(nproc, cuda_visible_devices_str)`` → ``Popen``.
    3. After each checkpoint (detected by *checkpoint_indicator* mtime
       change), check whether **new** GPUs became idle.
    4. If yes, terminate training and loop back to step 1.
    5. Repeat until training completes or *max_devices* is reached on
       every checkpoint.

    On crashes the loop auto-resumes from the last checkpoint (up to
    *max_crashes* times).

    Parameters
    ----------
    launch_fn : (nproc, cuda_visible) -> Popen
        Factory that starts the training subprocess.  ``nproc`` is the
        device count, ``cuda_visible`` is the comma-separated device
        index string (already set in ``os.environ`` by :func:`device_scope`).
    checkpoint_indicator : str
        Path to a file whose mtime changes whenever a checkpoint is
        saved (e.g. ``latest_checkpointed_iteration.txt``).
    min_devices / max_devices : int
        Device count bounds.
    check_interval : float
        Seconds between expansion probes while training is running.
    max_crashes : int
        Maximum consecutive crash-restarts before giving up.
    crash_cooldown : float
        Seconds to wait after a crash or intentional restart before
        re-acquiring devices (allows GPU memory to fully release).

    Returns
    -------
    int
        Exit code of the final training run (0 = success).
    """
    crashes = 0

    while True:
        # ── Phase 1: acquire devices ────────────────────────────────
        with device_scope(
            min_devices=min_devices,
            max_devices=max_devices,
            warmup=warmup,
            reserve_mb=reserve_mb,
            poll_interval=device_poll_interval,
            memory_threshold_pct=memory_threshold_pct,
            power_threshold_w=power_threshold_w,
            set_cuda_visible=True,
            auto_release=False,
            verbose=verbose,
        ) as slot:
            nproc = slot.count
            devices = list(slot.devices)
            cuda_vis = slot.cuda_visible
            slot.release()  # free VRAM right before training launches

            if verbose:
                print(
                    f"[model_worker] elastic: launching training with "
                    f"{nproc} devices {devices}"
                )

            # ── Phase 2: launch training ────────────────────────────
            proc = launch_fn(nproc, cuda_vis)

            # ── Phase 3: monitor ────────────────────────────────────
            restarting = _watch_for_expansion(
                proc=proc,
                current_devices=devices,
                max_devices=max_devices,
                checkpoint_indicator=checkpoint_indicator,
                check_interval=check_interval,
                memory_threshold_pct=memory_threshold_pct,
                power_threshold_w=power_threshold_w,
                verbose=verbose,
            )

        # ── Out of device_scope context ─────────────────────────────

        if restarting:
            # Intentional restart for GPU expansion — reset crash counter.
            crashes = 0
            if verbose:
                step = _read_step(checkpoint_indicator)
                print(
                    f"[model_worker] elastic: cooldown {crash_cooldown:.0f}s "
                    f"before restart (last ckpt step {step}) …"
                )
            time.sleep(crash_cooldown)
            continue

        # Training exited on its own.
        rc = proc.returncode
        if rc == 0:
            if verbose:
                print("[model_worker] elastic: training completed successfully")
            return 0

        # ── Crash handling ──────────────────────────────────────────
        crashes += 1
        if crashes > max_crashes:
            if verbose:
                print(
                    f"[model_worker] elastic: training crashed {crashes} "
                    f"times (exit {rc}), giving up"
                )
            return rc

        if _file_mtime(checkpoint_indicator) > 0:
            if verbose:
                step = _read_step(checkpoint_indicator)
                print(
                    f"[model_worker] elastic: training crashed (exit {rc}), "
                    f"resuming from step {step} "
                    f"({crashes}/{max_crashes})"
                )
            time.sleep(crash_cooldown)
            continue
        else:
            if verbose:
                print(
                    f"[model_worker] elastic: training crashed (exit {rc}) "
                    f"with no checkpoint — cannot resume"
                )
            return rc
