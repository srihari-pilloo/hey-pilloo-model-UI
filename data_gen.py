import argparse
import csv
import os
import random
import sys
import threading
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

SAMPLE_RATE = 16000          # what sweep_wav.py / LiveKit wake-word training wants
GEMINI_NATIVE_RATE = 24000   # what Gemini TTS actually returns

VOICES = [
    ("Kore", "female"),
    ("Puck", "male"),
    ("Aoede", "female"),
    ("Charon", "male"),
    ("Leda", "female"),
    ("Orus", "male"),
    ("Zephyr", "female"),
    ("Fenrir", "male"),
    ("Autonoe", "female"),
    ("Umbriel", "male"),
]

STYLES = {
    "normal": "Say naturally, in an Indian English accent:",
    "faster": "Say a bit faster than normal, in an Indian English accent:",
    "quieter": "Say quietly and softly, in an Indian English accent:",
    "further": "Say as if speaking from further away in the room, in an Indian English accent:",
    "noisy": "Say naturally, in an Indian English accent:",  # noise is added afterward
}

VARIATION_WEIGHTS = {"normal": 6, "faster": 1, "quieter": 1, "further": 1, "noisy": 1}


class RateLimiter:
    def __init__(self, rpm):
        self.rpm = rpm
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()
                if len(self.calls) < self.rpm:
                    self.calls.append(now)
                    return
                wait = 60 - (now - self.calls[0]) + 0.05
            time.sleep(max(wait, 0.05))


def build_client(api_key):
    try:
        from google import genai
    except ImportError:
        sys.exit("Missing dependency. Run:\n    pip install google-genai numpy scipy")
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            "No API key found. Pass --api-key or set GEMINI_API_KEY "
            "(get one at https://aistudio.google.com/apikey)."
        )
    return genai.Client(api_key=key)


def synthesize(client, model, text, voice_name, limiter):
    from google.genai import types

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )
    last_err = None
    for attempt in range(5):
        limiter.acquire()
        try:
            response = client.models.generate_content(model=model, contents=text, config=config)
            return response.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Gemini TTS failed after retries: {last_err}")


def resample_to_16k(pcm_24k_bytes):
    from scipy.signal import resample_poly

    audio = np.frombuffer(pcm_24k_bytes, dtype=np.int16)
    resampled = resample_poly(audio.astype(np.float32), up=2, down=3)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def fit_to_duration(audio, seconds):
    target_len = int(SAMPLE_RATE * seconds)
    if len(audio) >= target_len:
        return audio[:target_len]
    pad = np.zeros(target_len - len(audio), dtype=np.int16)
    return np.concatenate([audio, pad])


def random_time_shift(audio, seconds):
    target_len = int(SAMPLE_RATE * seconds)
    if len(audio) >= target_len:
        start_max = len(audio) - target_len
        start = random.randint(0, start_max) if start_max > 0 else 0
        return audio[start:start + target_len]
    pad_total = target_len - len(audio)
    pad_left = random.randint(0, pad_total)
    pad_right = pad_total - pad_left
    return np.concatenate([
        np.zeros(pad_left, dtype=audio.dtype), audio, np.zeros(pad_right, dtype=audio.dtype)
    ])


def _lowpass(audio_f, cutoff=2500, fs=SAMPLE_RATE, order=4):
    from scipy.signal import butter, lfilter

    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return lfilter(b, a, audio_f)


def _noise(n, color="white"):
    white = np.random.normal(0, 1, n)
    if color == "white":
        return white
    # cheap pink-ish noise: shape the spectrum by 1/sqrt(f)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n)
    peak = np.max(np.abs(pink))
    return pink / peak if peak > 0 else pink


def _add_room_noise(audio, snr_db, color="white"):
    audio_f = audio.astype(np.float64)
    signal_power = np.mean(audio_f ** 2)
    if signal_power == 0:
        return audio
    noise = _noise(len(audio_f), color=color)
    noise_power = np.mean(noise ** 2)
    noise *= np.sqrt((signal_power / (10 ** (snr_db / 10))) / noise_power)
    return np.clip(audio_f + noise, -32768, 32767).astype(np.int16)


def _add_echo(audio_f, delay_ms, decay, fs=SAMPLE_RATE):
    delay = int(fs * delay_ms / 1000)
    echo = np.zeros_like(audio_f)
    if 0 < delay < len(audio_f):
        echo[delay:] = audio_f[:-delay] * decay
    return audio_f + echo


def apply_variation(audio, variation):
    if variation == "quieter":
        return np.clip(audio.astype(np.float64) * 0.35, -32768, 32767).astype(np.int16)
    if variation == "further":
        audio_f = _lowpass(audio.astype(np.float64), cutoff=2500) * 0.4
        audio_f = _add_echo(audio_f, delay_ms=random.uniform(40, 90), decay=random.uniform(0.15, 0.3))
        return np.clip(audio_f, -32768, 32767).astype(np.int16)
    if variation == "noisy":
        return _add_room_noise(audio, snr_db=random.uniform(5, 15), color=random.choice(["white", "pink"]))
    return audio


def augment_clip(audio, duration_seconds):
    from scipy.signal import resample

    gain = random.uniform(0.7, 1.3)
    speed = random.uniform(0.92, 1.08)  # shifts pitch slightly too, which is fine/realistic
    audio_f = audio.astype(np.float64) * gain
    new_len = max(1, int(len(audio_f) / speed))
    audio_f = resample(audio_f, new_len)
    audio_i = np.clip(audio_f, -32768, 32767).astype(np.int16)
    audio_i = random_time_shift(audio_i, duration_seconds)

    roll = random.random()
    if roll < 0.12:
        audio_i = _add_room_noise(audio_i, snr_db=random.uniform(8, 20), color=random.choice(["white", "pink"]))
    elif roll < 0.20:
        audio_f2 = _add_echo(audio_i.astype(np.float64), delay_ms=random.uniform(30, 70), decay=random.uniform(0.1, 0.25))
        audio_i = np.clip(audio_f2, -32768, 32767).astype(np.int16)
    return audio_i


def save_wav(path, audio):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())


def load_wav_i16(path):
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def level_report(audio):
    peak = int(np.abs(audio).max()) if len(audio) else 0
    verdict = "TOO QUIET" if peak < 1000 else ("CLIPPING" if peak > 32000 else "ok")
    bar = "#" * int(peak / 32768 * 30)
    return f"peak {peak:5d} [{bar:<30}] {verdict}"


def load_existing_manifest(manifest_path, outdir):
    rows, base_audio, max_i = {}, {}, 0
    if not os.path.exists(manifest_path):
        return max_i, rows, base_audio
    with open(manifest_path, newline="") as mf:
        for row in csv.DictReader(mf):
            fname = row["file"]
            try:
                idx = int(fname[4:8])
            except (ValueError, IndexError):
                continue
            max_i = max(max_i, idx)
            rows[idx] = [row["file"], row["voice"], row["gender"], row["variation"], row["source"], row["text"]]
            if row["source"] == "tts":
                wav_path = os.path.join(outdir, row["file"])
                if os.path.exists(wav_path):
                    base_audio[idx] = load_wav_i16(wav_path)
    return max_i, rows, base_audio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wakeword", nargs="?", default="hey pilloo",
                    help="the wake word/phrase to synthesize, e.g. 'hey pilloo'")
    ap.add_argument("-n", "--count", type=int, default=40,
                    help="total clips this run should add (default 40)")
    ap.add_argument("-u", "--unique", type=int, default=None,
                    help="how many of those are real API-generated TTS calls this run; "
                         "the rest are made locally by augmenting the full real pool "
                         "(default: min(count, 100))")
    ap.add_argument("-o", "--outdir", default="synthetic_recordings")
    ap.add_argument("-d", "--duration", type=float, default=2.0, help="seconds per clip (default 2.0)")
    ap.add_argument("--model", default="gemini-3.1-flash-tts-preview",
                    choices=["gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"])
    ap.add_argument("--api-key", default=None, help="overrides GEMINI_API_KEY env var")
    ap.add_argument("--rpm", type=int, default=8,
                    help="max API requests per minute -- check Google AI Studio for your real "
                         "limit and set this a couple below it (default 8)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent API calls, capped by --rpm (default 4)")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing outdir's manifest instead of starting over -- "
                         "run this daily to grow your real TTS pool within quota limits")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    if args.list_voices:
        for name, gender in VOICES:
            print(f"  {name:12s} {gender}")
        return

    unique = args.unique if args.unique is not None else min(args.count, 100)
    unique = min(unique, args.count)

    os.makedirs(args.outdir, exist_ok=True)
    manifest_path = os.path.join(args.outdir, "manifest.csv")

    start_i = 0
    existing_rows, base_audio = {}, {}
    if args.append:
        start_i, existing_rows, base_audio = load_existing_manifest(manifest_path, args.outdir)
        if start_i:
            print(f"--append: found {len(existing_rows)} existing clips ({len(base_audio)} real), "
                  f"continuing from clip{start_i + 1:04d}\n")

    client = build_client(args.api_key)
    limiter = RateLimiter(args.rpm)
    variations = list(VARIATION_WEIGHTS.keys())
    weights = list(VARIATION_WEIGHTS.values())

    jobs = []
    for offset in range(1, args.count + 1):
        i = start_i + offset
        voice_name, gender = VOICES[i % len(VOICES)]
        variation = random.choices(variations, weights=weights)[0]
        jobs.append({"i": i, "voice": voice_name, "gender": gender, "variation": variation,
                     "is_base": offset <= unique})

    print(f"Adding {args.count} clips ({unique} real TTS calls + {args.count - unique} local augmentations)")
    print(f"Model: {args.model}, rate limit {args.rpm} req/min, {args.workers} workers\n")

    manifest_rows = dict(existing_rows)
    print_lock = threading.Lock()

    def run_base_job(job):
        prompt_text = f"{STYLES[job['variation']]} {args.wakeword}"
        pcm_24k = synthesize(client, args.model, prompt_text, job["voice"], limiter)
        audio = resample_to_16k(pcm_24k)
        audio = fit_to_duration(audio, args.duration)
        audio = apply_variation(audio, job["variation"])
        return job["i"], audio

    base_jobs = [j for j in jobs if j["is_base"]]
    aug_jobs = [j for j in jobs if not j["is_base"]]

    done_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_base_job, job): job for job in base_jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                i, audio = fut.result()
            except Exception as e:
                with print_lock:
                    print(f"  [FAILED] clip{job['i']:04d}: {e}")
                continue
            base_audio[i] = audio
            path = os.path.join(args.outdir, f"clip{i:04d}_{job['variation']}.wav")
            save_wav(path, audio)
            manifest_rows[i] = [os.path.basename(path), job["voice"], job["gender"], job["variation"], "tts", args.wakeword]
            done_count += 1
            with print_lock:
                print(f"  [{done_count}/{unique}] {path}  voice={job['voice']:10s} "
                      f"variation={job['variation']:8s} {level_report(audio)}")

    if not base_audio:
        sys.exit("No real TTS clips available (none generated, none from a prior --append run) "
                  "-- check your API key and quota.")

    print(f"\nReal clip pool: {len(base_audio)} total. Generating {len(aug_jobs)} local augmentations "
          f"(no API calls, fast)...")
    base_ids = list(base_audio.keys())
    for job in aug_jobs:
        source_i = random.choice(base_ids)
        audio = augment_clip(base_audio[source_i], args.duration)
        path = os.path.join(args.outdir, f"clip{job['i']:04d}_{job['variation']}_aug.wav")
        save_wav(path, audio)
        manifest_rows[job["i"]] = [os.path.basename(path), job["voice"], job["gender"], job["variation"], "augmented", args.wakeword]

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["file", "voice", "gender", "variation", "source", "text"])
        for idx in sorted(manifest_rows):
            writer.writerow(manifest_rows[idx])

    print(f"\nDone. {len(manifest_rows)} clips total in {args.outdir}/ ({len(base_audio)} real). "
          f"Manifest at {manifest_path}")
    print(f"Now run:\n  python sweep_wav.py hey_pilloo.onnx {args.outdir}/*.wav")


if __name__ == "__main__":
    main()