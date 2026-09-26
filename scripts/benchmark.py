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


def math_stack_info() -> dict:
    """Record the numeric stack, because it changes the answer by 2x.

    Two installs of the *same* onnxruntime version, on the same machine, with
    identical session options, were measured at 191 ms and 402 ms for the same
    5 s window. The only differences were numpy 2.5.3 built against Apple's
    Accelerate versus numpy 2.1.3 built against OpenBLAS. So an absolute
    millisecond figure is not a property of this project -- it is a property of
    the environment -- and the honest response is to ship the environment with
    the number so a reader can account for a mismatch instead of wondering.
    """
    info: dict = {}
    try:
        import numpy as np

        info["numpy"] = np.__version__
        deps = np.__config__.show(mode="dicts").get("Build Dependencies", {})
        info["numpy_blas"] = deps.get("blas", {}).get("name", "unknown")
    except Exception:  # noqa: BLE001
        info["numpy"] = "unavailable"
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["providers_available"] = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        info["onnxruntime"] = "unavailable"
    return info


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


class _StageProbe:
    """Times the encoder and decoder ORT calls for the life of a `with` block.

    Reported because the total alone is misleading. On an 8-core Apple Silicon
    laptop the Whisper *encoder* forward measures a flat 125 ms regardless of
    which interpreter, which onnxruntime build, thread count or session-creation
    order is used -- it is a fixed cost of the graph. Everything above that is
    per-token work, whose cost is dominated by re-feeding the encoder KV cache
    (4 layers x 2 tensors x 1x8x1500x64 float32 ~= 25 MB) on every step. That is
    a property of the encoder+decoder ONNX export, not of the hardware.

    The split is the useful part for a Snapdragon port: a faster NPU moves the
    125 ms and nothing else. It also explains an otherwise baffling
    observation -- total ASR latency varied 2x between two Python environments
    on one machine while vision stayed flat at 4.9 ms. The variable part is
    memory traffic, which the allocator and BLAS build affect; a conv forward
    is compute-bound and neither does.

    `make_stage_probe` returns None when the engine has no inspectable runner,
    so a missing breakdown is never presented as a measured one.
    """

    def __init__(self, engine):
        self._totals = {"encoder_ms": 0.0, "decoder_ms": 0.0,
                        "decode_steps": 0}
        runner = getattr(engine, "_asr", None)
        encoder = getattr(runner, "_encoder", None)
        decoder = getattr(runner, "_decoder", None)
        if encoder is None or decoder is None:
            raise TypeError("engine exposes no encoder+decoder pair")
        if not (hasattr(encoder, "run") and hasattr(decoder, "run")):
            raise TypeError("sessions do not expose run()")
        self._encoder, self._decoder = encoder, decoder
        self._enc_run, self._dec_run = encoder.run, decoder.run
        self._armed = False

    def __enter__(self):
        encoder, decoder = self._encoder, self._decoder
        encoder.run = self._wrap(self._enc_run, "encoder_ms")
        decoder.run = self._wrap(self._dec_run, "decoder_ms", steps=True)
        self._armed = True
        return self._totals

    def __exit__(self, *exc):
        if self._armed:
            self._encoder.run = self._enc_run
            self._decoder.run = self._dec_run
            self._armed = False
        return False

    def _wrap(self, fn, key, steps=False):
        def wrapper(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self._totals[key] += (time.perf_counter() - t) * 1000.0
                if steps:
                    self._totals["decode_steps"] += 1
        return wrapper


def _make_stage_probe(engine):
    """Return a probe for ``engine``, or None if its internals are not visible."""
    try:
        return _StageProbe(engine)
    except TypeError:
        return None


def _benchmark_asr(engine, args, results: dict) -> None:
    """Time real transcription of real speech.

    It has to be real audio. An earlier version of this script fed
    ``np.zeros(...)``, which the engine's short-window gate now declines to
    transcribe -- so the benchmark silently started timing the gate instead of
    the model and reported 0.03 ms and a real-time factor of 166,666x. A
    benchmark that measures its own guard rather than the thing it exists to
    measure is worse than no benchmark, because the number looks plausible.

    Rather than synthesise something, this uses the checked-in reference clip:
    its correct transcript is known, so the input is both real and verifiable.
    """
    import numpy as np

    clip = config.PROJECT_ROOT / "assets" / "reference-speech.wav"
    if not clip.is_file():
        print("\n  ASR  skipped: assets/reference-speech.wav is missing.")
        print("       Synthetic input would be measured, but a synthetic input")
        print("       cannot be transcribed and this benchmark would be fiction.")
        return

    # A missing model makes `transcribe` return an error string immediately, so
    # the loop below would "measure" a function call that does no inference and
    # report a spectacular 0.03 ms. Check up front and say nothing rather than
    # say something wrong.
    if not getattr(engine, "has_asr", False):
        print("\n  ASR  skipped: no speech model is loaded.")
        print(f"       Get one with:  python scripts/fetch_models.py --public")
        print("       Without this check the numbers below would be the cost of")
        print("       returning an error string, not of running a model.")
        return

    import wave

    with wave.open(str(clip), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    mono = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    want = int(config.CAPTURE_CHUNK_SECONDS * sr)
    # Only full windows: a short tail is refused by the engine, so including it
    # would measure the refusal rather than the inference.
    windows = [mono[i:i + want] for i in range(0, len(mono) - want + 1, want)]
    if not windows:
        print(f"\n  ASR  skipped: {clip.name} is shorter than one "
              f"{config.CAPTURE_CHUNK_SECONDS:g}s window.")
        return

    window_seconds = want / float(sr)
    samples: list[float] = []
    refuted = 0
    probe = _make_stage_probe(engine)
    stages: dict = {}
    for i in range(args.runs + args.warmup):
        w = windows[i % len(windows)]
        t0 = time.perf_counter()
        if probe is not None:
            with probe as acc:
                result = engine.transcribe(w, sr)
            stages = acc
        else:
            result = engine.transcribe(w, sr)
        dt = (time.perf_counter() - t0) * 1000.0
        if not result.usable or result.text.startswith("["):
            # An engine with no model returns a bracketed marker and is marked
            # usable, so the flag check alone would not catch it.
            refuted += 1
        if i >= args.warmup:
            samples.append(dt)

    if refuted:
        print(f"\n  ASR  WARNING: the engine refused {refuted} of "
              f"{args.runs + args.warmup} windows, so these numbers are NOT")
        print("       inference times. Check that the input is real audio.")

    if refuted == args.runs + args.warmup:
        print("\n  ASR  FAILED: every window was refused, so there is no "
              "inference time to report.")
        return

    stats = _stats(samples)
    results["asr"] = stats
    results["asr"]["window_seconds"] = round(window_seconds, 2)
    results["asr"]["source"] = f"assets/{clip.name}"
    if stages:
        n = args.runs + args.warmup
        enc = stages["encoder_ms"] / n
        dec = stages["decoder_ms"] / n
        results["asr"]["stage_breakdown_ms"] = {
            "encoder": round(enc, 1),
            "decode_loop": round(dec, 1),
            "decode_steps_per_window": round(stages["decode_steps"] / n, 1),
        }

    print(f"\n  ASR  ({window_seconds:g}s window, real speech from {clip.name})")
    if stats:
        print(f"    mean {stats['mean_ms']} ms | median {stats['median_ms']} ms | "
              f"p90 {stats['p90_ms']} ms | stdev {stats['stdev_ms']} ms")
        # A median that rounds to 0.0 is below this harness's resolution, not
        # infinitely fast. Dividing by it yields 0, which the naive comparison
        # then reports as "SLOWER than real time" -- precisely backwards.
        median = stats["median_ms"]
        if median <= 0:
            print("    real-time factor not reported: the median rounded to "
                  "0.0 ms, below this harness's resolution")
        else:
            rtf = (window_seconds * 1000) / median
            results["asr"]["realtime_factor"] = round(rtf, 2)
            print(f"    real-time factor {rtf:.2f}x  "
                  f"({'faster' if rtf >= 1 else 'SLOWER'} than real time)")
            if "stage_breakdown_ms" in results["asr"]:
                b = results["asr"]["stage_breakdown_ms"]
                print(f"    of which: encoder {b['encoder']} ms fixed, "
                      f"{b['decode_steps_per_window']} decode steps "
                      f"{b['decode_loop']} ms")


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
    for k, v in {**host_info(), **math_stack_info()}.items():
        print(f"  {k:18s} {v}")

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

    results: dict = {"host": host_info(), "math_stack": math_stack_info(),
                     "engine": engine.name,
                     "device": engine.device_description,
                     "neural_accelerated": engine.is_neural_accelerated}

    engine.warmup()

    if not args.vision_only:
        _benchmark_asr(engine, args, results)

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
