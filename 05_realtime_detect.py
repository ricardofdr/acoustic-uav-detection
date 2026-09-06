"""
Real-time acoustic UAV detection from a live microphone feed.

Mirrors the exact feature-extraction pipeline used in notebooks 01-04:
    - 44 100 Hz sample rate
    - 200 ms frames (8 820 samples), peak-normalised
    - 40 MFCCs, 0-8 kHz Mel range, n_fft=2048
    - L2-normalised MFCC vectors
    - windows of 10 consecutive frames -> (10, 40) tensor fed to the LSTM
    - rolling 3-window average of the softmax output (~6 s), as in notebook 04

Install once:
    pip install sounddevice librosa tensorflow numpy

List available microphones:
    python realtime_detect.py --list-devices

Run:
    python realtime_detect.py --model models/classifier_X_openmax_closed_set.keras
    python realtime_detect.py --model models/classifier_X_background.keras --device 1
"""

import argparse
import collections
import queue
import sys
import time

import numpy as np
import librosa
import sounddevice as sd
from tensorflow import keras

# ---- Constants: must match notebooks 01/02 exactly ----
SAMPLE_RATE = 44_100
FRAME_MS = 200
FRAME_SAMPLES = int(FRAME_MS * SAMPLE_RATE / 1000)  # 8 820
N_MFCC = 40
FMAX = 8_000
N_FFT = 2048
N_FRAMES = 10           # frames per window (~2 s), matches notebook 02
N_WINDOWS_AVG = 3        # rolling average, per DronePrint Sec. 4.5.1 (~6 s)
DEFAULT_THRESHOLD = 0.5  # tune on your validation set (Sec. 3.6 / OpenMax thresholds)


def peak_normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Time-domain peak normalisation per 200 ms frame (matches notebook 02)."""
    peak = np.max(np.abs(frame))
    return frame / peak if peak > 0 else frame


def l2_normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Feature-vector L2 rescaling (matches notebook 02)."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def frame_to_mfcc(frame: np.ndarray) -> np.ndarray:
    """One 200 ms frame -> (40,) L2-normalised MFCC vector, identical to notebook 02."""
    frame = peak_normalize_frame(frame)
    mfcc = librosa.feature.mfcc(
        y=frame, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=len(frame) + 1, fmax=FMAX,
    )  # shape (N_MFCC, 1)
    return l2_normalize_vector(mfcc[:, 0])


class RealtimeDetector:
    def __init__(self, model_path: str, threshold: float = DEFAULT_THRESHOLD,
                 class_names=("non_drone", "drone")):
        print(f"Loading model: {model_path}")
        self.model = keras.models.load_model(model_path)
        self.threshold = threshold
        self.class_names = class_names

        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        # keep only as many frames as one window needs
        self.mfcc_frames = collections.deque(maxlen=N_FRAMES)
        self.prob_history = collections.deque(maxlen=N_WINDOWS_AVG)
        self.n_windows_seen = 0

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # indata shape: (FRAME_SAMPLES, channels); take mono channel 0
        self.audio_q.put(indata[:, 0].copy())

    def run(self, device=None):
        print(f"Sample rate: {SAMPLE_RATE} Hz | frame: {FRAME_MS} ms "
              f"({FRAME_SAMPLES} samples) | window: {N_FRAMES} frames "
              f"(~{N_FRAMES * FRAME_MS / 1000:.1f}s) | rolling avg: "
              f"{N_WINDOWS_AVG} windows (~{N_WINDOWS_AVG * N_FRAMES * FRAME_MS / 1000:.1f}s)")
        print("Listening... (Ctrl+C to stop)\n")

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SAMPLES,
                             device=device, callback=self._audio_callback):
            try:
                while True:
                    block = self.audio_q.get()  # exactly FRAME_SAMPLES samples
                    t0 = time.perf_counter()

                    mfcc_vec = frame_to_mfcc(block.astype(np.float32))
                    self.mfcc_frames.append(mfcc_vec)

                    if len(self.mfcc_frames) < N_FRAMES:
                        remaining = N_FRAMES - len(self.mfcc_frames)
                        print(f"\rBuffering... {remaining} frame(s) until first prediction",
                              end="", flush=True)
                        continue

                    window = np.array(self.mfcc_frames)[np.newaxis, ...]  # (1, 10, 40)
                    probs = self.model.predict(window, verbose=0)[0]
                    self.prob_history.append(probs)
                    self.n_windows_seen += 1

                    avg_probs = np.mean(self.prob_history, axis=0)
                    pred_class = int(np.argmax(avg_probs))
                    confidence = float(avg_probs[pred_class])
                    latency_ms = (time.perf_counter() - t0) * 1000

                    label = self.class_names[pred_class] if pred_class < len(self.class_names) \
                        else f"class_{pred_class}"
                    flag = "*** DRONE DETECTED ***" if label == "drone" and confidence >= self.threshold \
                        else label
                    bar = "#" * int(confidence * 30)
                    n_avg = len(self.prob_history)
                    print(f"\r[{time.strftime('%H:%M:%S')}] {flag:24s} "
                          f"p={confidence:.2f} avg_over={n_avg}/{N_WINDOWS_AVG} "
                          f"infer={latency_ms:5.1f}ms {bar:<30s}", end="", flush=True)

            except KeyboardInterrupt:
                print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Path to a .keras model file (e.g. Classifier X)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", type=int, default=None, help="Input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    if not args.model:
        parser.error("--model is required unless --list-devices is given")

    detector = RealtimeDetector(args.model, threshold=args.threshold)
    detector.run(device=args.device)
