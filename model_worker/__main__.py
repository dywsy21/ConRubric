"""CLI: ``python -m model_worker [status|wait|hold]``

    python -m model_worker               # device health table
    python -m model_worker wait -n 4     # block until 4 ready
    python -m model_worker hold -n 4     # reserve until Ctrl-C
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from model_worker.health import DeviceMonitor
from model_worker.warmup import DeviceWarmup


def cmd_status(args):
    print(DeviceMonitor(args.mem_threshold, args.power_threshold).summary())


def cmd_wait(args):
    mon = DeviceMonitor(args.mem_threshold, args.power_threshold)
    ready = mon.wait_until_ready(
        min_count=args.n, max_count=args.N or args.n,
        poll_interval=args.interval, timeout=args.timeout,
        verbose=not args.quiet,
    )
    idxs = [d.index for d in ready]
    cv = ",".join(str(i) for i in idxs)
    if args.format == "shell":
        print(f"export CUDA_VISIBLE_DEVICES={cv}; export DEVICE_COUNT={len(idxs)}")
    elif args.format == "json":
        import json
        print(json.dumps({"devices": idxs, "count": len(idxs), "cuda_visible": cv}))
    else:
        print(cv)


def cmd_hold(args):
    mon = DeviceMonitor(args.mem_threshold, args.power_threshold)
    wu = DeviceWarmup(reserve_mb=args.reserve_mb)

    def _cleanup(*_):
        print("\n[model_worker] shutting down …")
        wu.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    max_n = args.N or args.n
    print(f"[model_worker] waiting for {args.n} devices (hold mode) …")

    while True:
        wu.health_check()
        ready = mon.ready_devices()
        need = max_n - len(wu.active_devices)
        if ready and need > 0:
            wu.warm(ready[:need])
        if len(wu.active_devices) >= args.n:
            print(f"[model_worker] holding {len(wu.active_devices)} devices: {wu.active_devices}")
            print("[model_worker] Ctrl-C to release")
            try:
                while True:
                    wu.health_check()
                    if len(wu.active_devices) < args.n:
                        print("[model_worker] lost a device — re-scanning …")
                        break
                    time.sleep(10)
            except KeyboardInterrupt:
                _cleanup()
        else:
            time.sleep(args.interval)


def main():
    p = argparse.ArgumentParser(prog="model_worker", description="Device health & scheduling")
    p.add_argument("--mem-threshold", type=float, default=10.0)
    p.add_argument("--power-threshold", type=float, default=100.0)
    sub = p.add_subparsers(dest="command")

    sub.add_parser("status", help="Device health table")

    w = sub.add_parser("wait", help="Wait for N devices")
    w.add_argument("-n", type=int, required=True)
    w.add_argument("-N", type=int, default=None)
    w.add_argument("--interval", type=float, default=30.0)
    w.add_argument("--timeout", type=float, default=None)
    w.add_argument("--format", choices=["shell", "json", "plain"], default="shell")
    w.add_argument("-q", "--quiet", action="store_true")

    h = sub.add_parser("hold", help="Reserve devices until Ctrl-C")
    h.add_argument("-n", type=int, required=True)
    h.add_argument("-N", type=int, default=None)
    h.add_argument("--interval", type=float, default=30.0)
    h.add_argument("--reserve-mb", type=int, default=1024)

    args = p.parse_args()
    if args.command is None or args.command == "status":
        cmd_status(args)
    elif args.command == "wait":
        cmd_wait(args)
    elif args.command == "hold":
        cmd_hold(args)


if __name__ == "__main__":
    main()
