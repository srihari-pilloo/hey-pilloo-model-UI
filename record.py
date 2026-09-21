"""
Record wake-word clips at 16kHz mono 16-bit, the format sweep_wav.py needs.

Usage:
    python record.py                     # 20 clips, 3s each, into recordings/
    python record.py -n 30 -d 2.5
    python record.py --list-devices
    python record.py --device 2
"""
import argparse
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


def save_wav(path, audio):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())


def level_report(audio):
    """Peak level + a verdict, so you catch a dead or clipping mic early."""
    peak = int(np.abs(audio).max())
    if peak < 1000:
        verdict = "TOO QUIET - raise mic gain or move closer"
    elif peak > 32000:
        verdict = "CLIPPING - lower mic gain or back off"
    else:
        verdict = "ok"
    bar = "#" * int(peak / 32768 * 30)
    return f"peak {peak:5d} [{bar:<30}] {verdict}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=20,
                    help="how many clips to record")
    ap.add_argument("-d", "--duration", type=float, default=3.0,
                    help="seconds per clip")
    ap.add_argument("-o", "--outdir", default="recordings")
    ap.add_argument("--device", type=int, default=None,
                    help="input device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    os.makedirs(args.outdir, exist_ok=True)
    frames = int(SAMPLE_RATE * args.duration)

    print(f"Recording {args.count} clips of {args.duration}s into {args.outdir}/")
    print("Say the wake word once per clip, starting right after the beep line.")
    print("Vary it: normal, faster, quieter, further away, some in a noisy room.")
    print("Ctrl+C to stop early.\n")

    try:
        for i in range(1, args.count + 1):
            input(f"  [{i}/{args.count}] press Enter, then speak...")
            for countdown in ("3", "2", "1", "GO"):
                print(f"\r      {countdown}   ", end="", flush=True)
                time.sleep(0.4)

            audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1,
                           dtype="int16", device=args.device)
            sd.wait()
            audio = audio[:, 0]

            path = os.path.join(args.outdir, f"clip{i:02d}.wav")
            save_wav(path, audio)
            print(f"\r      saved {path}  {level_report(audio)}")
    except KeyboardInterrupt:
        print("\nStopped early.")

    print(f"\nDone. Now run:\n  python sweep_wav.py hey_pilloo.onnx {args.outdir}/*.wav")


if __name__ == "__main__":
    main()