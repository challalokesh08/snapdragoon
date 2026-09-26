#!/usr/bin/env python3
"""Measure inference latency so the deployment claim is backed by numbers.

This is the script that produces the table in your submission. Run it on your Mac
now, and re-run the identical command on the HP Omnibook when you get one; the
only difference should be the ``--backend`` flag.

    python scripts/benchmark.py --backend onnx --runs 20
    python scripts/benchmark.py --backend qualcomm --runs 50

Output is both human-readable and JSON (``--json out/bench.json``) so you can
paste a real table into the writeup instead of a guess.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from snapdragoon import config  # noqa: E402
from snapdragoon.engines import get_engine  # noqa: E402


def _stats(samples: list[float]) -> dict:
    if not samples:
        return {}
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "mean_ms": round(statistics.mean(ordered), 2),
        "median_ms": round(statistics.median(ordered), 2),
        "p90_ms": round(ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))], 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
        "stdev_ms": round(statistics.stdev(ordered), 2) if len(ordered) > 1 else 0.0,
    }


def host_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }
    # Best-effort CPU model on Windows / Linux without extra dependencies.
    if platform.system() == "Windows":
        try:
            import wmi  # type: ignore

            info["cpu"] = wmi.processor()[0].Name
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        except Exception:  # noqa: BLE001
            pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "onnx", "qualcomm", "demo"])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None, help="also write JSON here")
    ap.add_argument("--asr-only", action="store_true")
    ap.add_argument("--vision-only", action="store_true")
    args = ap.parse_args()

    print(f"\nSnapdragoon benchmark  (runs={args.runs}, warmup={args.warmup})\n")
    for k, v in host_info().items():
        print(f"  {k:10s} {v}")

    try:
        engine = get_engine(args.backend)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Could not initialise backend {args.backend!r}: {exc}\n")
        return 2

    print(f"\n  engine     {engine.name}")
    print(f"  device     {engine.device_description}")
    print(f"  NPU        {'yes' if engine.is_neural_accelerated else 'no'}")

    if engine.name == "demo":
        print(
            "\n  WARNING: the demo engine performs no inference. These numbers\n"
            "           describe nothing and must not appear in your submission.\n"
            "           Install requirements-ml.txt and fetch models first.\n"
        )

    results: dict = {"host": host_info(), "engine": engine.name,
                     "device": engine.device_description,
                     "neural_accelerated": engine.is_neural_accelerated}

    engine.warmup()

    if not args.vision_only:
        # int() is required, not cosmetic: CAPTURE_CHUNK_SECONDS is a float
        # config value, and np.zeros rejects a float length.
        window_samples = int(config.CAPTURE_CHUNK_SECONDS * config.SAMPLE_RATE)
        audio = np.zeros(window_samples, dtype=np.float32)
        samples = []
        for i in range(args.runs + args.warmup):
            t0 = time.perf_counter()
            engine.transcribe(audio, config.SAMPLE_RATE)
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= args.warmup:
                samples.append(dt)
        s = _stats(samples)
        results["asr"] = s
        print(f"\n  ASR  ({config.CAPTURE_CHUNK_SECONDS:g}s window)")
        if s:
            print(f"    mean {s['mean_ms']} ms | median {s['median_ms']} ms | "
                  f"p90 {s['p90_ms']} ms | stdev {s['stdev_ms']} ms")
            rtf = (config.CAPTURE_CHUNK_SECONDS * 1000) / s["median_ms"] if s["median_ms"] else 0
            results["asr"]["realtime_factor"] = round(rtf, 2)
            print(f"    real-time factor {rtf:.2f}x  "
                  f"({'faster' if rtf >= 1 else 'SLOWER'} than real time)")

    if not args.asr_only:
        frame = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
        samples = []
        for i in range(args.runs + args.warmup):
            t0 = time.perf_counter()
            engine.classify(frame)
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= args.warmup:
                samples.append(dt)
        s = _stats(samples)
        results["classifier"] = s
        print("\n  Vision (640x480 frame)")
        if s:
            print(f"    mean {s['mean_ms']} ms | median {s['median_ms']} ms | "
                  f"p90 {s['p90_ms']} ms | stdev {s['stdev_ms']} ms")
            results["classifier"]["fps"] = round(1000.0 / s["median_ms"], 2) if s["median_ms"] else 0
            print(f"    {results['classifier']['fps']} fps")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\n  wrote {args.json}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
