from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from threading import Condition
from time import monotonic
from typing import Protocol

import numpy as np

from app.config import AudioConfig

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore
try:
    import soundcard as sc  # type: ignore
except Exception:  # pragma: no cover
    sc = None  # type: ignore


class AudioSource(Protocol):
    def read_frame(self) -> np.ndarray:
        """Return mono float32 audio frame in [-1, 1]."""


class MockAudioSource:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = deque(frames)

    def read_frame(self) -> np.ndarray:
        if self._frames:
            return self._frames.popleft()
        return np.zeros((480,), dtype=np.float32)


class WasapiLoopbackSource:
    def __init__(self, config: AudioConfig) -> None:
        if sd is None and sc is None:
            raise RuntimeError("Neither sounddevice nor soundcard is installed")
        self._sample_rate = config.sample_rate
        self._frame_len = max(128, int(config.sample_rate * config.frame_ms / 1000))
        self._queue: deque[np.ndarray] = deque(maxlen=8)
        self._cv = Condition()
        self._closed = False
        self._config = config
        self.selected_device: int | None = None
        self.selected_sample_rate: int = self._sample_rate
        self.selected_backend: str = "unknown"
        self.selected_endpoint: str | None = None
        self._sc_recorder = None

        backend = (config.backend or "auto").lower()
        try_soundcard = backend == "soundcard" or (backend == "auto" and config.input_device is None)
        if try_soundcard:
            try:
                self._open_soundcard_loopback()
                return
            except Exception:
                if backend == "soundcard":
                    raise
        self._stream = self._open_stream()
        self._stream.start()
        self.selected_backend = "sounddevice"

    def _open_soundcard_loopback(self) -> None:
        if sc is None:
            raise RuntimeError("soundcard is not installed")
        # Compatibility fix for soundcard on NumPy >=2.
        np.fromstring = np.frombuffer  # type: ignore[assignment]
        speaker = None
        if self._config.loopback_speaker_contains:
            key = self._config.loopback_speaker_contains.lower()
            for cand in sc.all_speakers():
                if key in cand.name.lower():
                    speaker = cand
                    break
        if speaker is None:
            speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("No default speaker for loopback capture")
        self.selected_device = None
        self.selected_sample_rate = self._sample_rate
        self.selected_endpoint = speaker.name
        mic = sc.get_microphone(id=str(speaker.id), include_loopback=True)
        self._sc_recorder = mic.recorder(samplerate=self._sample_rate, channels=2)
        self._sc_recorder.__enter__()
        self.selected_backend = "soundcard"

    def _candidate_devices(self) -> list[int]:
        assert sd is not None
        devices = list(sd.query_devices())
        candidates: list[int] = []
        if self._config.input_device is not None:
            chosen = int(self._config.input_device)
            if 0 <= chosen < len(devices):
                candidates.append(chosen)

        preferred_terms = (
            "stereo mix",
            "stereo input",
            "what u hear",
            "loopback",
            "wave out",
            "立体声混音",
        )

        for idx, dev in enumerate(devices):
            name = str(dev["name"]).lower()
            if int(dev["max_input_channels"]) <= 0:
                continue
            if any(term in name for term in preferred_terms):
                candidates.append(idx)

        default_device = sd.default.device
        if hasattr(default_device, "__getitem__"):
            candidate = int(default_device[0])
            if candidate >= 0 and candidate not in candidates:
                candidates.append(candidate)

        for idx, dev in enumerate(devices):
            if int(dev["max_input_channels"]) > 0 and idx not in candidates:
                candidates.append(idx)

        return candidates

    def _open_stream(self) -> object:
        assert sd is not None
        last_error: Exception | None = None
        for device_id in self._candidate_devices():
            dev_info = sd.query_devices(device_id)
            default_sr = int(float(dev_info["default_samplerate"]))
            channels = max(1, min(2, int(dev_info["max_input_channels"])))
            sample_rates = [self._sample_rate]
            if default_sr not in sample_rates:
                sample_rates.append(default_sr)
            for sr in sample_rates:
                try:
                    stream = sd.InputStream(
                        samplerate=sr,
                        device=device_id,
                        channels=channels,
                        dtype="float32",
                        blocksize=0,
                        callback=self._on_audio,
                    )
                    self.selected_device = device_id
                    self._sample_rate = sr
                    self.selected_sample_rate = sr
                    self.selected_endpoint = str(sd.query_devices(device_id)["name"])
                    return stream
                except Exception as exc:  # pragma: no cover
                    last_error = exc
                    continue
        raise RuntimeError(f"Failed to open any audio input stream: {last_error}")

    def _on_audio(self, indata: np.ndarray, frames: int, t: object, status: object) -> None:
        del frames, t, status
        if indata.ndim == 2:
            frame = np.asarray(indata.mean(axis=1), dtype=np.float32).copy()
        else:
            frame = np.asarray(indata, dtype=np.float32).copy()
        with self._cv:
            self._queue.append(frame)
            self._cv.notify()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.selected_backend == "soundcard" and self._sc_recorder is not None:
            self._sc_recorder.__exit__(None, None, None)
        else:
            self._stream.stop()
            self._stream.close()
            with self._cv:
                self._cv.notify_all()

    def read_frame(self) -> np.ndarray:
        if self.selected_backend == "soundcard" and self._sc_recorder is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chunk = self._sc_recorder.record(numframes=self._frame_len)
            if chunk.ndim == 2:
                return np.asarray(chunk[:, 0], dtype=np.float32)
            return np.asarray(chunk, dtype=np.float32)
        with self._cv:
            if not self._queue and not self._closed:
                self._cv.wait(timeout=1.0)
            if self._queue:
                return self._queue.popleft()
        return np.zeros((self._frame_len,), dtype=np.float32)


def list_input_devices() -> list[dict[str, object]]:
    if sd is None:
        return []
    result: list[dict[str, object]] = []
    for i, dev in enumerate(sd.query_devices()):
        if int(dev["max_input_channels"]) <= 0:
            continue
        result.append(
            {
                "index": i,
                "name": str(dev["name"]),
                "max_input_channels": int(dev["max_input_channels"]),
                "default_samplerate": int(float(dev["default_samplerate"])),
            }
        )
    return result


def list_loopback_speakers() -> list[str]:
    if sc is None:
        return []
    return [sp.name for sp in sc.all_speakers()]


@dataclass(slots=True)
class AudioEvent:
    ts_ms: int
    energy: float
    threshold: float


class SplashDetector:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.noise_floor = 1e-5
        self._boot_count = 0
        self._last_event_ms = -10_000

    def update(self, frame: np.ndarray, now_ms: int | None = None) -> AudioEvent | None:
        if now_ms is None:
            now_ms = int(monotonic() * 1000)

        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        if self._boot_count < self.config.bootstrap_frames:
            self._boot_count += 1
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
            return None

        self.noise_floor = 0.995 * self.noise_floor + 0.005 * min(rms, self.noise_floor * 1.5)
        threshold = max(self.noise_floor * self.config.threshold_k, 0.01)

        refractory_ok = (now_ms - self._last_event_ms) >= self.config.refractory_ms
        if refractory_ok and rms >= threshold:
            self._last_event_ms = now_ms
            return AudioEvent(ts_ms=now_ms, energy=rms, threshold=threshold)
        return None

