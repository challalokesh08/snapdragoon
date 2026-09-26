#!/usr/bin/env python3
"""Terminal demo. The fastest way to produce a screen recording for submission.

    # 1. Replay a wav, so the demo is repeatable and needs no microphone
    python scripts/demo.py --file assets/sample.wav

    # 2. Live microphone
    python scripts/demo.py --mic

    # 3. Describe an image
    python scripts/demo.py --image assets/sample.jpg

    # 4. Describe from the camera, every two seconds
    python scripts/demo.py --camera --interval 2

Recording notes: capture the terminal at 2x scale with a dark background, and
start the clip *after* the first caption lands. Judges should see output, not
install logs.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from snapdragoon import config  # noqa: E402
from snapdragoon.engines import get_engine  # noqa: E402

BOLD, DIM, CYAN, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"
)


def _banner(engine) -> None:
    print(f"\n{BOLD}Snapdragoon{RESET}  {DIM}on-device accessibility co-pilot{RESET}")
    print(f"  engine   {CYAN}{engine.name}{RESET}")
    print(f"  device   {engine.device_description}")
    print(f"  NPU      {'yes' if engine.is_neural_accelerated else 'no'}")
    if engine.name == "demo":
        print(f"\n  {YELLOW}Demo engine: no inference is running. Captions below are "
              f"synthetic.{RESET}")
    print()


def _transcribe_wav(engine, path: Path) -> None:
    from snapdragoon.audio.capture import is_silent

    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != config.SAMPLE_RATE:
            print(f"{YELLOW}warning:{RESET} {path.name} is {wf.getframerate()} Hz, "
                  f"expected {config.SAMPLE_RATE} Hz. Results will be poor.")
        want = int(config.CAPTURE_CHUNK_SECONDS * wf.getframerate())
        index = 0
        while True:
            raw = wf.readframes(want)
            if not raw:
                break
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if is_silent(data, threshold=2e-3):
                print(f"{DIM}[{index:02d}] (silence){RESET}")
            else:
                res = engine.transcribe(data, config.SAMPLE_RATE)
                # A clip's final window is short. The engine pads it to the
                # model's 30 s context, and Whisper answers a mostly-empty
                # window with a hallucination rather than with nothing, so the
                # engine marks it unusable and the caption is dropped.
                if not res.usable:
                    print(f"{DIM}[{index:02d}] ({res.skip_reason}, skipped){RESET}")
                else:
                    print(f"{GREEN}[{index:02d}]{RESET} {res.text} "
                          f"{DIM}({res.latency_ms:.0f} ms){RESET}")
            index += 1
            time.sleep(0.15)  # pace it so the recording is watchable


def _transcribe_mic(engine) -> None:
    from snapdragoon.audio.capture import iter_windows, open_stream

    print("  listening — press Ctrl-C to stop\n")
    try:
        stream = open_stream()
    except RuntimeError as exc:
        print(f"{YELLOW}{exc}{RESET}")
        return
    index = 0
    try:
        for window in iter_windows(stream, config.CAPTURE_CHUNK_SECONDS):
            res = engine.transcribe(window, config.SAMPLE_RATE)
            print(f"{GREEN}[{index:02d}]{RESET} {res.text} "
                  f"{DIM}({res.latency_ms:.0f} ms){RESET}")
            index += 1
    except KeyboardInterrupt:
        print(f"\n  stopped after {index} windows")


def _describe_image(engine, path: Path) -> None:
    from snapdragoon.vision.scene import load_image

    frame = load_image(path)
    dets = engine.classify(frame)
    print(f"  {BOLD}{path.name}{RESET}  {DIM}{frame.shape[1]}x{frame.shape[0]}{RESET}\n")
    for d in dets:
        bar = "█" * int(round(d.score * 24))
        print(f"  {d.label:<28s} {d.score * 100:5.1f}%  {CYAN}{bar}{RESET}")
    print(f"\n  {DIM}inference {dets[0].latency_ms:.1f} ms{RESET}\n")


def _describe_camera(engine, interval: float) -> None:
    from snapdragoon.vision.scene import camera_frames

    print("  camera running — press Ctrl-C to stop\n")
    last = 0.0
    try:
        for frame in camera_frames():
            now = time.monotonic()
            if now - last < interval:
                continue
            last = now
            dets = engine.classify(frame)
            top = ", ".join(f"{d.label} {d.score * 100:.0f}%" for d in dets)
            print(f"  {GREEN}▸{RESET} {top}  {DIM}({dets[0].latency_ms:.0f} ms){RESET}")
    except KeyboardInterrupt:
        print("\n  stopped")
    except RuntimeError as exc:
        print(f"{YELLOW}{exc}{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="replay a 16 kHz mono wav")
    src.add_argument("--mic", action="store_true", help="live microphone")
    src.add_argument("--image", type=Path, help="describe a still image")
    src.add_argument("--camera", action="store_true", help="describe camera frames")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "onnx", "qualcomm", "demo"])
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between camera descriptions")
    args = ap.parse_args()

    try:
        engine = get_engine(args.backend)
    except Exception as exc:  # noqa: BLE001
        print(f"could not initialise backend {args.backend!r}: {exc}")
        return 2

    _banner(engine)
    engine.warmup()

    if args.file:
        if not args.file.is_file():
            print(f"no such file: {args.file}")
            return 1
        _transcribe_wav(engine, args.file)
    elif args.mic:
        _transcribe_mic(engine)
    elif args.image:
        if not args.image.is_file():
            print(f"no such file: {args.image}")
            return 1
        _describe_image(engine, args.image)
    else:
        _describe_camera(engine, args.interval)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
