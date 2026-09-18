"""Dash tester for hey_pilloo_wakeword.tflite.

Run locally on the laptop that has the microphone:
    python wakeword_dash_test.py --model hey_pilloo_wakeword.tflite

The page displays the detected wake-word count and a visual listening indicator.
"""

from __future__ import annotations

import argparse
import threading
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from dash import Dash, dcc, html, Input, Output

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError as exc:
            raise ImportError(
                "Install one TFLite runtime: tensorflow, ai-edge-litert, or tflite-runtime."
            ) from exc


SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE  # The notebook exports a [16000] waveform input.
BLOCK_SAMPLES = 1600         # 100 ms capture blocks; 10 inferences per second.
THRESHOLD = 0.50              # Notebook OPERATING_THRESHOLD.
RELEASE_THRESHOLD = 0.50      # Must fall below this before another trigger is allowed.
NORMALIZE_PEAK = 0.50         # Same preprocessing as the notebook.


class WakeWordListener:
    def __init__(self, model_path: str, device=None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")

        self.interpreter = Interpreter(model_path=str(self.model_path), num_threads=2)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.window = deque(maxlen=WINDOW_SAMPLES)
        self.lock = threading.Lock()
        self.trigger_count = 0
        self.armed = True
        self.stream = None
        self.device = device

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            # Do not put status text on the Dash page; the UI is intentionally count-only.
            pass
        samples = np.asarray(indata[:, 0], dtype=np.float32)
        with self.lock:
            self.window.extend(samples.tolist())
            if len(self.window) < WINDOW_SAMPLES:
                return
            audio = np.asarray(self.window, dtype=np.float32)

        probability = self.predict(audio)
        with self.lock:
            if self.armed and probability >= THRESHOLD:
                self.trigger_count += 1
                self.armed = False
            elif not self.armed and probability < RELEASE_THRESHOLD:
                self.armed = True

    def predict(self, audio: np.ndarray) -> float:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < WINDOW_SAMPLES:
            audio = np.pad(audio, (0, WINDOW_SAMPLES - audio.size))
        elif audio.size > WINDOW_SAMPLES:
            audio = audio[-WINDOW_SAMPLES:]

        # Match load_and_pad_wav(): peak-normalize before the 1-second window is inferred.
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-6:
            audio = audio * (NORMALIZE_PEAK / peak)

        expected_shape = tuple(self.input_details["shape"])
        if int(np.prod(expected_shape)) != WINDOW_SAMPLES:
            raise ValueError(
                f"Expected a TFLite input containing {WINDOW_SAMPLES} samples, "
                f"but the model input shape is {expected_shape}."
            )
        model_input = audio.reshape(expected_shape).astype(self.input_details["dtype"])
        self.interpreter.set_tensor(self.input_details["index"], model_input)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details["index"])
        return float(np.asarray(output).reshape(-1)[0])

    def start(self) -> None:
        if self.stream is not None:
            return
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SAMPLES,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def count(self) -> int:
        with self.lock:
            return self.trigger_count

    def is_listening(self) -> bool:
        with self.lock:
            return self.stream is not None


def build_app(listener: WakeWordListener) -> Dash:
    app = Dash(__name__)
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Hey Pilloo Wake Word</title>
            {%favicon%}
            {%css%}
            <style>
                :root {
                    color-scheme: dark;
                    font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
                    background: #080b14;
                }
                body {
                    margin: 0;
                    min-height: 100vh;
                    background:
                        radial-gradient(circle at 50% 35%, #182344 0%, #080b14 48%, #05060b 100%);
                    color: #f4f7ff;
                }
                .wake-card {
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: 24px;
                }
                .mic-stage {
                    position: relative;
                    width: 150px;
                    height: 150px;
                    display: grid;
                    place-items: center;
                }
                .mic-ring, .mic-ring::before, .mic-ring::after {
                    position: absolute;
                    inset: 0;
                    border-radius: 50%;
                    border: 2px solid rgba(94, 234, 212, 0.42);
                    content: "";
                    animation: breathe 2.1s ease-out infinite;
                }
                .mic-ring::before { animation-delay: 0.7s; }
                .mic-ring::after { animation-delay: 1.4s; }
                .mic-core {
                    width: 78px;
                    height: 78px;
                    display: grid;
                    place-items: center;
                    border-radius: 50%;
                    background: linear-gradient(145deg, #5eead4, #38bdf8);
                    color: #07111f;
                    box-shadow: 0 0 34px rgba(94, 234, 212, 0.42);
                    animation: glow 1.3s ease-in-out infinite alternate;
                    z-index: 1;
                }
                .mic-symbol {
                    font-size: 38px;
                    line-height: 1;
                }
                .listening-label {
                    margin: 0;
                    color: #8cf5e4;
                    font-size: 16px;
                    letter-spacing: 0.18em;
                    text-transform: uppercase;
                }
                .count-label {
                    margin: 0;
                    color: #8e9bb9;
                    font-size: 13px;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                }
                .trigger-count {
                    margin: 0;
                    font-size: clamp(72px, 14vw, 148px);
                    line-height: 0.9;
                    font-weight: 750;
                    letter-spacing: -0.06em;
                    color: #ffffff;
                    text-shadow: 0 0 28px rgba(255,255,255,0.16);
                }
                @keyframes breathe {
                    0% { transform: scale(0.52); opacity: 0.9; }
                    75%, 100% { transform: scale(1.12); opacity: 0; }
                }
                @keyframes glow {
                    from { transform: scale(0.96); box-shadow: 0 0 26px rgba(94, 234, 212, 0.34); }
                    to { transform: scale(1.04); box-shadow: 0 0 48px rgba(56, 189, 248, 0.62); }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """
    app.layout = html.Main(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(className="mic-ring"),
                            html.Div("\N{MICROPHONE}", className="mic-symbol"),
                        ],
                        className="mic-core",
                    ),
                ],
                id="listening-indicator",
                className="mic-stage",
            ),
            html.P("Listening", className="listening-label"),
            html.P("Triggers detected", className="count-label"),
            html.H1(id="trigger-count", children="0", className="trigger-count"),
            dcc.Interval(id="refresh", interval=200, n_intervals=0),
        ],
        className="wake-card",
    )

    @app.callback(
        Output("trigger-count", "children"),
        Input("refresh", "n_intervals"),
    )
    def update_count(_n_intervals):
        return str(listener.count())

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="hey_pilloo_wakeword.tflite",
        help="Path to the TFLite model exported by the notebook.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--device", default=None, help="Optional sounddevice input device name or index.")
    args = parser.parse_args()

    listener = WakeWordListener(args.model, device=args.device)
    try:
        listener.start()
    except Exception as exc:
        raise RuntimeError(
            "Could not open the microphone at exactly 16 kHz. Check the input device and "
            "sounddevice installation; run `python -m sounddevice` to list devices."
        ) from exc

    app = build_app(listener)
    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
