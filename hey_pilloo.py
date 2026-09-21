import argparse
import asyncio
import atexit
import json
import sys
import time
import os
from datetime import datetime
import queue
import threading
from collections import deque

from livekit.wakeword import WakeWordModel

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Tip: install 'rich' for a nice live dashboard →  pip install rich\n")


# ---------------------------------------------------------------- Word metrics
#
# `model.predict(audio)` returns a confidence score *per wake word* the model
# knows about (e.g. {"hey_pilloo": 0.83, "stop": 0.02}). The old code kept
# only one of those keys and threw the rest away, so you could never see how
# close a non-target word was to firing, or tune its threshold independently.
#
# WordTracker keeps running stats (current score, threshold, trigger count,
# peak score, last-trigger time, rolling history) for every word it sees, and
# is shared by both the CLI --raw mode and the Streamlit UI so the two
# surfaces report identical numbers.

class WordTracker:
    """Thread-safe per-wake-word metrics: score, threshold, triggers, peak."""

    def __init__(self, default_threshold: float, overrides: dict | None = None,
                 cooldown: float = 1.5, history_len: int = 200):
        self.default_threshold = default_threshold
        self.overrides = dict(overrides or {})
        self.cooldown = cooldown
        self.history_len = history_len
        self.lock = threading.Lock()
        self._words: dict = {}  # name -> {threshold, score, peak, triggers, last_trigger_ts, history}

    def _ensure(self, name: str) -> dict:
        if name not in self._words:
            self._words[name] = {
                "threshold": self.overrides.get(name, self.default_threshold),
                "score": 0.0,
                "peak": 0.0,
                "triggers": 0,
                "last_trigger_ts": 0.0,
                "history": deque(maxlen=self.history_len),
            }
        return self._words[name]

    def update(self, scores: dict, now: float | None = None):
        """Feed one prediction round (word -> raw score). Applies the
        per-word threshold + cooldown and returns the list of words that
        newly triggered on this call."""
        now = now if now is not None else time.time()
        newly_triggered = []
        with self.lock:
            for name, score in scores.items():
                w = self._ensure(name)
                w["score"] = float(score)
                w["peak"] = max(w["peak"], float(score))
                w["history"].append(float(score))
                if score >= w["threshold"] and (now - w["last_trigger_ts"]) > self.cooldown:
                    w["triggers"] += 1
                    w["last_trigger_ts"] = now
                    newly_triggered.append(name)
        return newly_triggered

    def record_trigger(self, name: str, confidence: float, now: float | None = None) -> None:
        """For event-based APIs (e.g. WakeWordListener) that only hand you
        already-triggered detections, not a continuous raw score stream."""
        now = now if now is not None else time.time()
        with self.lock:
            w = self._ensure(name)
            w["score"] = float(confidence)
            w["peak"] = max(w["peak"], float(confidence))
            w["history"].append(float(confidence))
            w["triggers"] += 1
            w["last_trigger_ts"] = now

    def set_threshold(self, name: str, threshold: float) -> None:
        with self.lock:
            self._ensure(name)["threshold"] = threshold
            self.overrides[name] = threshold

    def set_default_threshold(self, threshold: float) -> None:
        with self.lock:
            self.default_threshold = threshold
            for name, w in self._words.items():
                if name not in self.overrides:
                    w["threshold"] = threshold

    def snapshot(self) -> dict:
        """Point-in-time copy safe to read/render without holding the lock."""
        with self.lock:
            return {name: {**w, "history": list(w["history"])} for name, w in self._words.items()}

    def live_snapshot(self) -> dict:
        """Strip everything the dashboard doesn't need (history) so the
        per-tick UI snapshot stays tiny and fast at high refresh rates."""
        with self.lock:
            return {
                name: {
                    "threshold": w["threshold"],
                    "score": w["score"],
                    "peak": w["peak"],
                    "triggers": w["triggers"],
                    "last_trigger_ts": w["last_trigger_ts"],
                }
                for name, w in self._words.items()
            }

    def total_triggers(self) -> int:
        with self.lock:
            return sum(w["triggers"] for w in self._words.values())


def _sparkline_svg(values, width: int = 320, height: int = 60, color: str = "#4CAF50") -> str:
    """Tiny dependency-free sparkline (no pandas/numpy/plotly needed) so the
    Streamlit UI never has to import pandas just to draw a history chart.
    Some locked-down Windows machines block pandas' compiled Arrow-array DLL
    via Application Control policy, which crashes st.line_chart."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(values)
    step = width / max(n - 1, 1)
    points = " ".join(
        f"{i * step:.1f},{height - ((v - lo) / span) * (height - 6) - 3:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round" /></svg>'
    )


def parse_threshold_overrides(spec: str | None) -> dict:
    """Parse '--thresholds hey_pilloo=0.55,stop=0.6' into {"hey_pilloo": 0.55, "stop": 0.6}."""
    overrides: dict = {}
    if not spec:
        return overrides
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --thresholds entry (expected name=value): {part!r}")
        name, val = part.split("=", 1)
        overrides[name.strip()] = float(val.strip())
    return overrides


def make_dashboard(snapshot: dict, mode: str, note: str = "") -> Panel:
    """Render one row per known wake word: live score bar, numeric score,
    threshold, triggered/idle status, trigger count, and peak score — so you
    can see exactly why a word did or didn't fire."""
    total_triggers = sum(w["triggers"] for w in snapshot.values())

    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Word", style="bold cyan", no_wrap=True)
    table.add_column("Status", justify="center", no_wrap=True)
    table.add_column("Confidence", ratio=2)
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Triggers", justify="right")
    table.add_column("Peak", justify="right")
    table.add_column("Last Trigger", justify="right")

    if not snapshot:
        table.add_row("—", "—", "", "—", "—", "—", "—", "—")
    else:
        for name in sorted(snapshot):
            w = snapshot[name]
            score = w["score"]
            threshold = w["threshold"]
            triggered = score >= threshold

            bar_len = 24
            filled = int(min(max(score, 0.0), 1.0) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            bar_style = "bold green" if triggered else ("yellow" if score > threshold * 0.7 else "dim")
            status = "[bold green]TRIGGERED[/]" if triggered else "[dim]idle[/]"

            last_trig = w["last_trigger_ts"]
            since = f"{time.time() - last_trig:.1f}s ago" if last_trig > 0 else "—"

            table.add_row(
                name,
                status,
                f"[{bar_style}]{bar}[/]",
                f"{score:.3f}",
                f"{threshold:.3f}",
                str(w["triggers"]),
                f"{w['peak']:.3f}",
                since,
            )

    header = Table.grid(padding=(0, 2))
    header.add_column(style="cyan", justify="right")
    header.add_column()
    header.add_row("Mode", f"[bold]{mode}[/]")
    header.add_row("Total Triggers", f"[bold green]{total_triggers}[/]")
    header.add_row("Time", datetime.now().strftime("%H:%M:%S"))
    if note:
        header.add_row("Note", f"[italic dim]{note}[/]")

    body = Table.grid()
    body.add_row(header)
    body.add_row(table)

    return Panel(
        body,
        title="[bold white]Wake Word Live Dashboard[/]",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )


async def run_simple(model_path: str, threshold: float, debounce: float, model_name: str,
                      overrides: dict | None = None):
    from livekit.wakeword import WakeWordListener

    model = WakeWordModel(models=[model_path])
    tracker = WordTracker(default_threshold=threshold, overrides=overrides, cooldown=debounce)
    note = "SIMPLE mode only shows scores for words that already crossed the listener's own threshold."

    print(f"Loaded model: {model_path}")
    print(f"Threshold: {threshold}  |  Debounce: {debounce}s")
    print("Listening... say your wake word. Ctrl+C to stop.\n")

    if not RICH_AVAILABLE:
        print(f"{'='*50}")
        print(f"  DASHBOARD  |  Triggers: 0")
        print(f"{'='*50}\n")

    async with WakeWordListener(model, threshold=threshold, debounce=debounce) as listener:
        if RICH_AVAILABLE:
            with Live(make_dashboard({}, "SIMPLE", note=note),
                      refresh_per_second=8, console=Console()) as live:
                while True:
                    detection = await listener.wait_for_detection()
                    tracker.record_trigger(detection.name, detection.confidence)
                    live.update(make_dashboard(tracker.snapshot(), "SIMPLE", note=note))
        else:
            while True:
                detection = await listener.wait_for_detection()
                tracker.record_trigger(detection.name, detection.confidence)
                snap = tracker.snapshot()
                print(f"\033[F\033[K" * (4 + len(snap)), end="")
                print(f"{'='*50}")
                print(f"  DASHBOARD  |  Triggers: {tracker.total_triggers()}")
                print(f"{'='*50}")
                for name, w in sorted(snap.items()):
                    print(f"  {name:<20} score={w['score']:.3f}  threshold={w['threshold']:.3f}  "
                          f"triggers={w['triggers']}  peak={w['peak']:.3f}")


def run_raw(model_path: str, threshold: float, model_name: str, cooldown: float = 1.5,
            overrides: dict | None = None):
    import numpy as np
    try:
        import sounddevice as sd
    except ImportError:
        print("This mode needs sounddevice: pip install sounddevice")
        sys.exit(1)

    model = WakeWordModel(models=[model_path])
    tracker = WordTracker(default_threshold=threshold, overrides=overrides, cooldown=cooldown)

    print(f"Loaded model: {model_path}")
    print(f"Default threshold: {threshold}" + (f"  |  Overrides: {overrides}" if overrides else ""))
    print("Listening (raw mode)... Ctrl+C to stop.\n")

    SAMPLE_RATE = 16000
    CHUNK_SEC = 0.1
    WINDOW_SEC = 2.0
    window_samples = int(SAMPLE_RATE * WINDOW_SEC)
    chunk_samples = int(SAMPLE_RATE * CHUNK_SEC)

    audio_buffer = np.zeros(window_samples, dtype=np.int16)
    audio_q = queue.Queue()
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        # Keep this as fast as possible — no inference here.
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    def process_loop():
        nonlocal audio_buffer
        while not stop_event.is_set():
            try:
                new_chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            audio_buffer = np.concatenate([audio_buffer, new_chunk])[-window_samples:]
            scores = model.predict(audio_buffer)  # {word_name: score, ...} for EVERY word in the model
            tracker.update(scores)

    worker = threading.Thread(target=process_loop, daemon=True)
    worker.start()

    if RICH_AVAILABLE:
        console = Console()
        with Live(make_dashboard({}, "RAW"), refresh_per_second=20, console=console) as live:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=chunk_samples,
                callback=callback,
                latency="low",
            ):
                try:
                    while True:
                        live.update(make_dashboard(tracker.snapshot(), "RAW"))
                        sd.sleep(30)
                except KeyboardInterrupt:
                    stop_event.set()
                    print(f"\nStopped. Total triggers: {tracker.total_triggers()}")
    else:
        prev_lines = 0
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
            callback=callback,
            latency="low",
        ):
            try:
                while True:
                    snap = tracker.snapshot()
                    if prev_lines:
                        print(f"\033[F\033[K" * prev_lines, end="")
                    lines = [f"DASHBOARD  |  Total triggers: {tracker.total_triggers()}"]
                    for name, w in sorted(snap.items()):
                        bar_len = int(min(max(w["score"], 0.0), 1.0) * 30)
                        bar = "#" * bar_len + "-" * (30 - bar_len)
                        marker = " <-- TRIGGERED" if w["score"] >= w["threshold"] else ""
                        lines.append(
                            f"  {name:<20} [{bar}] score={w['score']:.3f} "
                            f"threshold={w['threshold']:.3f} triggers={w['triggers']}{marker}"
                        )
                    print("\n".join(lines))
                    prev_lines = len(lines)
                    sd.sleep(30)
            except KeyboardInterrupt:
                stop_event.set()
                print(f"\n\nStopped. Total triggers: {tracker.total_triggers()}")


# ---------------------------------------------------------------- Streamlit UI


class _LiveState:
    """Shared, thread-safe state bridging the capture worker and the Streamlit UI."""

    def __init__(self) -> None:
        self.tracker: WordTracker | None = None
        self.model_name = ""
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None


# Streamlit reruns the whole script top-to-bottom on most interactions (e.g.
# moving the threshold slider). An unconditional `_live = _LiveState()` here
# would recreate this object on every such rerun and silently disconnect the
# UI from the still-running capture thread. Guard it so it's created once per
# process and simply reused across reruns.
try:
    _live  # type: ignore[used-before-def]
except NameError:
    _live = _LiveState()

STREAM_COOLDOWN = 1.5
SESSION_LOG_DIR = "wakeword_logs"


def _save_session_data(reason: str = "stopped") -> str | None:
    """Dump every tracked word's final score, threshold, trigger count, peak
    score and confidence history to a local JSON file, so a listening session
    can be reviewed later for debugging / accuracy tuning."""
    tracker = _live.tracker
    if tracker is None:
        return None
    snapshot = tracker.snapshot()
    if not snapshot:
        return None  # nothing was ever heard this session — nothing worth saving

    os.makedirs(SESSION_LOG_DIR, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        "model_name": _live.model_name,
        "total_triggers": sum(w["triggers"] for w in snapshot.values()),
        "words": {
            name: {
                "threshold": w["threshold"],
                "final_score": w["score"],
                "peak_score": w["peak"],
                "triggers": w["triggers"],
                "last_trigger_ts": w["last_trigger_ts"],
                "history": w["history"],
            }
            for name, w in snapshot.items()
        },
    }
    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(SESSION_LOG_DIR, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# Best-effort save when the streamlit process itself exits (e.g. Ctrl+C in
# the terminal). Streamlit has no public "browser tab closed" hook, so this
# plus the explicit Stop-listening save below are the two practical points
# at which a "session" can be considered closed for a local dev tool like this.
atexit.register(lambda: _save_session_data(reason="process_exit"))


def _streamlit_capture_worker(model: WakeWordModel, tracker: WordTracker, chunk_sec: float = 0.1) -> None:
    """Same raw-mode pipeline as run_raw: mic -> rolling 2s window -> predict,
    feeding every word's score into the shared WordTracker.

    chunk_sec controls how often we score the audio window — smaller means a
    snappier/more responsive dashboard at the cost of more CPU spent on
    inference (the classifier itself is lightweight, so 100ms is comfortable
    on most machines; drop to 0.25s+ if your CPU is struggling)."""
    import numpy as np

    try:
        import sounddevice as sd
    except ImportError:
        _live.error = "This mode needs sounddevice: pip install sounddevice"
        return

    sample_rate = 16000
    window_sec = 2.0  # fixed: this is the audio window the model's embedding stage expects
    window_samples = int(sample_rate * window_sec)
    chunk_samples = int(sample_rate * chunk_sec)

    audio_buffer = np.zeros(window_samples, dtype=np.int16)
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            _live.error = str(status)
        audio_q.put(indata[:, 0].copy())

    def process_loop() -> None:
        nonlocal audio_buffer
        while not _live.stop_event.is_set():
            try:
                new_chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            audio_buffer = np.concatenate([audio_buffer, new_chunk])[-window_samples:]
            try:
                scores = model.predict(audio_buffer)  # every word's raw score, not just one
            except Exception as exc:
                _live.error = f"Predict error: {exc}"
                continue
            tracker.update(scores)

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
            callback=callback,
            latency="low",
        ):
            _live.error = None
            process_loop()
    except Exception as exc:
        _live.error = f"Microphone error: {exc}"


def _start_listening(model_path: str, model_name: str, threshold: float,
                      overrides: dict | None = None, chunk_sec: float = 0.1,
                      cooldown: float = STREAM_COOLDOWN) -> bool:
    if _live.thread is not None and _live.thread.is_alive():
        return False
    try:
        model = WakeWordModel(models=[model_path])
    except Exception as exc:
        _live.error = f"Could not load model: {exc}"
        return False
    _live.model_name = model_name
    _live.error = None
    _live.tracker = WordTracker(default_threshold=threshold, overrides=overrides, cooldown=cooldown)
    _live.stop_event = threading.Event()
    _live.thread = threading.Thread(
        target=_streamlit_capture_worker, args=(model, _live.tracker, chunk_sec), daemon=True
    )
    _live.thread.start()
    return True


def _stop_listening(reason: str = "stopped") -> str | None:
    _live.stop_event.set()
    if _live.thread is not None:
        _live.thread.join(timeout=1.5)
    _live.thread = None
    return _save_session_data(reason=reason)


def _is_listening() -> bool:
    return _live.thread is not None and _live.thread.is_alive()


def run_streamlit() -> None:
    """Streamlit live tester.

    Run with:  streamlit run hey_pilloo.py
    Audio is captured from THIS machine's microphone (server-side via
    sounddevice) and every wake word's confidence, threshold and trigger
    count is streamed to the browser UI.
    """
    import streamlit as st

    st.set_page_config(page_title="Hey Pilloo - Live Wake Word Test", layout="wide")

    st.title("Hey Pilloo - Live Wake Word Test")
    st.caption(
        "Microphone input is captured on the machine running this app "
        "(server-side via sounddevice) and streamed to this dashboard. "
        "Every word the model knows is tracked, not just the one you're testing."
    )
    st.markdown(_DASHBOARD_CSS, unsafe_allow_html=True)  # injected once, not on every refresh tick

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input("Model path", value="hey_pilloo.tflite")
        model_name = st.text_input(
            "Model name",
            value=os.path.splitext(os.path.basename("hey_pilloo.tflite"))[0],
        )
        default_threshold = st.slider("Default threshold", 0.05, 0.95, 0.5, 0.01)
        overrides_text = st.text_input(
            "Per-word threshold overrides (optional)",
            value="",
            placeholder="hey_pilloo=0.55, stop=0.6",
            help="Words not listed here use the default threshold above.",
        )
        try:
            overrides = parse_threshold_overrides(overrides_text)
        except ValueError as exc:
            st.error(str(exc))
            overrides = {}

        st.header("Performance")
        infer_ms = st.slider(
            "Inference interval (ms)", 50, 500, 66, 10,
            help="How often audio is scored. Lower = more responsive, more CPU. Applied on next Start.",
        )
        refresh_ms = st.slider(
            "Dashboard refresh (ms)", 50, 500, 66, 10,
            help="How often the UI redraws. Lower = smoother, more browser/network overhead.",
        )
        cooldown = st.slider(
            "Trigger cooldown (s)", 0.2, 3.0, 1.0, 0.1,
            help="Minimum seconds between two triggers of the same word. Lower = counts repeat more aggressively.",
        )

        start_clicked = st.button("Start listening", type="primary", use_container_width=True)
        stop_clicked = st.button("Stop listening", use_container_width=True)
        save_clicked = st.button("Save session data now", use_container_width=True)

        st.divider()
        st.caption("Tip: say your wake word (and a few non-wake-word phrases) and watch each row react.")
        st.caption(f"Session data auto-saves on Stop to ./{SESSION_LOG_DIR}/")

    if start_clicked:
        if not os.path.exists(model_path):
            st.error(f"Model not found: {model_path}")
        elif _is_listening() and _live.model_name == model_name:
            st.info("Already listening with this model.")
        else:
            if _is_listening():
                _stop_listening()
            if _start_listening(model_path, model_name, default_threshold, overrides, chunk_sec=infer_ms / 1000.0, cooldown=cooldown):
                st.success("Listening - say your wake word!")
            else:
                st.error(_live.error or "Could not start listening.")
    if stop_clicked:
        saved_path = _stop_listening(reason="user_stopped")
        if saved_path:
            st.info(f"Stopped. Session data saved to `{saved_path}`.")
        else:
            st.info("Stopped. (Nothing was heard, so no file was written.)")
    if save_clicked:
        saved_path = _save_session_data(reason="manual_save")
        if saved_path:
            st.success(f"Saved to `{saved_path}`.")
        else:
            st.warning("Nothing to save yet — start listening first.")

    # Let live threshold-slider tweaks take effect immediately without restarting capture.
    if _is_listening() and _live.tracker is not None:
        _live.tracker.set_default_threshold(default_threshold)
        for name, val in overrides.items():
            _live.tracker.set_threshold(name, val)

    @st.fragment(run_every=refresh_ms / 1000.0)
    def dashboard() -> None:
        tracker = _live.tracker
        snapshot = tracker.live_snapshot() if tracker else {}
        st.markdown(
            _render_dashboard_html(snapshot, listening=_is_listening(), err=_live.error),
            unsafe_allow_html=True,
        )

    dashboard()


_DASHBOARD_CSS = """
<style>
.wd-wrap { font-family: -apple-system, "Segoe UI", system-ui, sans-serif; color: #e6ecf7; }
.wd-header { display: flex; align-items: flex-end; gap: 32px; padding: 2px 2px 20px 2px; flex-wrap: wrap; }
.wd-stat { display: flex; flex-direction: column; gap: 4px; }
.wd-stat.grow { margin-left: auto; }
.wd-k { font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: #8aa0bf; }
.wd-v { font-size: 42px; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums; }
.wd-v.accent { color: #34d399; }
.wd-v.md { font-size: 22px; color: #dbe6f5; }
.wd-chip { display: inline-flex; align-items: center; gap: 8px; font-weight: 800; font-size: 13px;
           letter-spacing: .12em; padding: 4px 0; }
.wd-chip .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.wd-chip.live   { color: #34d399; }
.wd-chip.live   .dot { background: #34d399; box-shadow: 0 0 10px #34d399; }
.wd-chip.idle   { color: #94a3b8; }
.wd-chip.idle   .dot { background: #94a3b8; }
.wd-chip.error  { color: #f87171; }
.wd-chip.error  .dot { background: #f87171; box-shadow: 0 0 10px #f87171; }
.wd-sub { margin: 0 0 8px 2px; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: #8aa0bf; }
.wd-lanes { display: flex; flex-direction: column; gap: 10px; }
.wd-lane { padding: 12px 16px 10px; border-radius: 12px;
           background: rgba(148,163,184,.07); border: 1px solid rgba(148,163,184,.14); }
.wd-lane.fire { border-color: rgba(52,211,153,.55); background: rgba(52,211,153,.07); box-shadow: 0 0 16px rgba(52,211,153,.16); }
.wd-lane-top { display: flex; align-items: center; gap: 12px; }
.wd-name { font-weight: 800; }
.wd-triggers { font-weight: 800; font-variant-numeric: tabular-nums; color: #8aa0bf; }
.wd-triggers.on { color: #34d399; }
.wd-frac { margin-left: auto; font-size: .82rem; color: #8aa0bf; font-variant-numeric: tabular-nums; }
.wd-bar { position: relative; height: 14px; margin-top: 10px;
          background: rgba(148,163,184,.15); border-radius: 999px; overflow: hidden; }
.wd-fill { height: 100%; border-radius: 999px;
           background: linear-gradient(90deg, #3b82f6, #22d3ee); transition: width .07s linear; }
.wd-fill.above { background: linear-gradient(90deg, #10b981, #34d399); box-shadow: 0 0 10px rgba(52,211,153,.55); }
.wd-tick { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #f87171; border-radius: 2px; }
.wd-meta { display: flex; gap: 18px; margin-top: 7px; font-size: .78rem; color: #8aa0bf; font-variant-numeric: tabular-nums; }
.wd-meta b { color: #c7d2e8; }
.wd-empty { padding: 18px 2px; font-size: .92rem; color: #94a3b8; }
</style>
"""


def _render_dashboard_html(snapshot: dict, listening: bool, err: str | None) -> str:
    """Live testing view: a big trigger counter plus one 'lane' per wake word
    with an animated confidence bar (green flash while firing), a red threshold
    tick, peak score and time since last trigger. Pure HTML so each 66ms tick
    stays cheap and lag-free."""
    now = time.time()
    total_triggers = sum(w["triggers"] for w in snapshot.values())

    if err:
        status_class, status_text = "error", "ERROR"
    elif listening:
        status_class, status_text = "live", "LISTENING"
    else:
        status_class, status_text = "idle", "STANDBY"

    lanes = []
    for name in sorted(snapshot, key=lambda n: snapshot[n]["score"], reverse=True):
        w = snapshot[name]
        pct = min(max(w["score"], 0.0), 1.0)
        above = w["score"] >= w["threshold"]
        recent = w["last_trigger_ts"] > 0 and (now - w["last_trigger_ts"]) < 1.5
        th_pct = min(max(w["threshold"], 0.0), 1.0) * 100.0
        ago = f"{now - w['last_trigger_ts']:.1f}s" if w["last_trigger_ts"] > 0 else "never"
        lanes.append(
            '<div class="wd-lane' + (" fire" if recent else "") + '">'
            '<div class="wd-lane-top">'
            f'<span class="wd-name">{name}</span>'
            f'<span class="wd-triggers{" on" if w["triggers"] else ""}">{w["triggers"]}</span>'
            f'<span class="wd-frac">{w["score"]:.3f} / {w["threshold"]:.3f}</span>'
            '</div>'
            '<div class="wd-bar">'
            f'<div class="wd-fill{" above" if above else ""}" style="width:{pct * 100:.1f}%"></div>'
            f'<div class="wd-tick" style="left:{th_pct:.1f}%"></div>'
            '</div>'
            '<div class="wd-meta">'
            f'<span>peak <b>{w["peak"]:.3f}</b></span>'
            f'<span>triggered <b>{ago}</b> ago</span>'
            '</div>'
            '</div>'
        )

    body = "".join(lanes) if lanes else (
        '<div class="wd-empty">No words yet — start listening and say your wake word.</div>'
    )

    return (
        '<div class="wd-wrap">'
        '<div class="wd-header">'
        '<div class="wd-stat"><span class="wd-k">Triggers</span>'
        f'<span class="wd-v accent">{total_triggers}</span></div>'
        '<div class="wd-stat"><span class="wd-k">Status</span>'
        f'<span class="wd-chip {status_class}"><span class="dot"></span>{status_text}</span></div>'
        '<div class="wd-stat grow"><span class="wd-k">Session</span>'
        f'<span class="wd-v md">{datetime.now().strftime("%H:%M:%S")}</span></div>'
        '</div>'
        '<div class="wd-sub">Live confidence</div>'
        '<div class="wd-lanes">' + body + '</div>'
        '</div>'
    )


def _is_streamlit_runtime() -> bool:
    try:
        import streamlit.runtime
    except Exception:
        return False
    try:
        return streamlit.runtime.exists()
    except Exception:
        return False


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
        help="Default detection threshold applied to every word. Use the 'optimal_threshold' from your eval output.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional per-word threshold overrides, e.g. 'hey_pilloo=0.55,stop=0.6'. "
             "Words not listed use --threshold.",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Seconds to wait before allowing another detection/trigger for the same word",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Stream continuous confidence scores for every word instead of discrete detections",
    )
    args = parser.parse_args()

    model_name = args.model_name or os.path.splitext(os.path.basename(args.model_path))[0]
    overrides = parse_threshold_overrides(args.thresholds)

    if args.raw:
        run_raw(args.model_path, args.threshold, model_name, cooldown=args.debounce, overrides=overrides)
    else:
        try:
            asyncio.run(run_simple(args.model_path, args.threshold, args.debounce, model_name, overrides=overrides))
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    if _is_streamlit_runtime():
        run_streamlit()
    else:
        main()