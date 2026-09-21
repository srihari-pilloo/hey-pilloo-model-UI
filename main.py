import argparse
import asyncio
import sys
import time
import os
from datetime import datetime
import queue

from livekit.wakeword import WakeWordModel

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.console import Console
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Tip: install 'rich' for a nice live dashboard →  pip install rich\n")


def make_dashboard(
    score: float,
    threshold: float,
    count: int,
    last_name: str,
    last_conf: float,
    last_time: float,
    mode: str,
) -> Panel:
    bar_len = 40
    filled = int(min(max(score, 0.0), 1.0) * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    # Color the bar
    if score >= threshold:
        bar_style = "bold green"
    elif score > threshold * 0.7:
        bar_style = "yellow"
    else:
        bar_style = "dim"

    time_since = f"{time.time() - last_time:.1f}s ago" if last_time > 0 else "—"

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column()

    table.add_row("Mode", f"[bold]{mode}[/]")
    table.add_row("Confidence", f"[{bar_style}]{bar}[/]  {score:.3f}")
    table.add_row("Threshold", f"{threshold:.3f}")
    table.add_row("Triggers", f"[bold green]{count}[/]")
    table.add_row("Last Word", f"[bold]{last_name or '—'}[/]  ({last_conf:.3f})" if last_name else "—")
    table.add_row("Last Trigger", time_since)
    table.add_row("Time", datetime.now().strftime("%H:%M:%S"))

    title = "[bold white]Wake Word Live Dashboard[/]"
    return Panel(
        table,
        title=title,
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )


async def run_simple(model_path: str, threshold: float, debounce: float, model_name: str):
    from livekit.wakeword import WakeWordListener

    model = WakeWordModel(models=[model_path])

    detection_count = 0
    last_name = ""
    last_conf = 0.0
    last_time = 0.0
    current_score = 0.0

    print(f"Loaded model: {model_path}")
    print(f"Threshold: {threshold}  |  Debounce: {debounce}s")
    print("Listening... say your wake word. Ctrl+C to stop.\n")

    if not RICH_AVAILABLE:
        print(f"{'='*50}")
        print(f"  DASHBOARD  |  Triggers: {detection_count}")
        print(f"{'='*50}\n")

    async with WakeWordListener(model, threshold=threshold, debounce=debounce) as listener:
        if RICH_AVAILABLE:
            with Live(make_dashboard(0.0, threshold, 0, "", 0.0, 0.0, "SIMPLE"),
                      refresh_per_second=8, console=Console()) as live:
                while True:
                    detection = await listener.wait_for_detection()
                    detection_count += 1
                    last_name = detection.name
                    last_conf = detection.confidence
                    last_time = time.time()
                    current_score = detection.confidence

                    live.update(
                        make_dashboard(
                            current_score, threshold, detection_count,
                            last_name, last_conf, last_time, "SIMPLE"
                        )
                    )
        else:
            # simple fallback
            while True:
                detection = await listener.wait_for_detection()
                detection_count += 1
                print(f"\033[F\033[K" * 4, end="")
                print(f"{'='*50}")
                print(f"  DASHBOARD  |  Triggers: {detection_count}")
                print(f"{'='*50}")
                print(f"  DETECTED '{detection.name}'  (confidence={detection.confidence:.3f})")


def run_raw(model_path: str, threshold: float, model_name: str):
    import queue
    import threading
    import numpy as np
    try:
        import sounddevice as sd
    except ImportError:
        print("This mode needs sounddevice: pip install sounddevice")
        sys.exit(1)

    model = WakeWordModel(models=[model_path])

    detection_count = 0
    last_conf = 0.0
    last_time = 0.0
    last_trigger_time = 0.0
    COOLDOWN = 1.5

    print(f"Loaded model: {model_path}")
    print(f"Threshold: {threshold}")
    print("Listening (raw mode)... Ctrl+C to stop.\n")

    SAMPLE_RATE = 16000
    CHUNK_SEC = 0.25
    WINDOW_SEC = 2.0
    window_samples = int(SAMPLE_RATE * WINDOW_SEC)
    chunk_samples = int(SAMPLE_RATE * CHUNK_SEC)

    audio_buffer = np.zeros(window_samples, dtype=np.int16)
    audio_q = queue.Queue()
    stop_event = threading.Event()
    state_lock = threading.Lock()

    def callback(indata, frames, time_info, status):
        # Keep this as fast as possible — no inference here.
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    def process_loop():
        nonlocal audio_buffer, detection_count, last_conf, last_time, last_trigger_time
        while not stop_event.is_set():
            try:
                new_chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            audio_buffer = np.concatenate([audio_buffer, new_chunk])[-window_samples:]
            scores = model.predict(audio_buffer)
            score = scores.get(model_name, list(scores.values())[0] if scores else 0.0)

            now = time.time()
            with state_lock:
                last_conf = score
                if score > threshold and (now - last_trigger_time) > COOLDOWN:
                    detection_count += 1
                    last_time = now
                    last_trigger_time = now

    worker = threading.Thread(target=process_loop, daemon=True)
    worker.start()

    if RICH_AVAILABLE:
        console = Console()
        with Live(make_dashboard(0.0, threshold, 0, model_name, 0.0, 0.0, "RAW"),
                  refresh_per_second=10, console=console) as live:

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=chunk_samples,
                callback=callback,
                latency="high",
            ):
                try:
                    while True:
                        with state_lock:
                            conf = last_conf
                            count = detection_count
                            ltime = last_time
                        live.update(
                            make_dashboard(
                                conf, threshold, count,
                                model_name, conf, ltime, "RAW"
                            )
                        )
                        sd.sleep(80)
                except KeyboardInterrupt:
                    stop_event.set()
                    print(f"\nStopped. Total triggers: {detection_count}")
    else:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
            callback=callback,
            latency="high",
        ):
            try:
                while True:
                    with state_lock:
                        conf = last_conf
                        count = detection_count

                    bar_len = int(min(max(conf, 0.0), 1.0) * 40)
                    bar = "#" * bar_len + "-" * (40 - bar_len)
                    marker = " <-- TRIGGER" if conf > threshold else ""
                    print(
                        f"\r  DASHBOARD  |  Triggers: {count:<4}  "
                        f"[{bar}] {conf:.3f}{marker}   ",
                        end="", flush=True,
                    )
                    sd.sleep(80)
            except KeyboardInterrupt:
                stop_event.set()
                print(f"\n\nStopped. Total triggers: {detection_count}")

def main():
    parser = argparse.ArgumentParser(description="Live mic test for a wake word model")
    parser.add_argument("model_path", help="Path to the exported .onnx (or .tflite) model file")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Wake word key as embedded in the model (defaults to the model filename without extension)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold. Use the 'optimal_threshold' from your eval output.",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Seconds to wait before allowing another detection (simple mode only)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Stream continuous confidence scores instead of discrete detections",
    )
    args = parser.parse_args()

    model_name = args.model_name or os.path.splitext(os.path.basename(args.model_path))[0]

    if args.raw:
        run_raw(args.model_path, args.threshold, model_name)
    else:
        try:
            asyncio.run(run_simple(args.model_path, args.threshold, args.debounce, model_name))
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()