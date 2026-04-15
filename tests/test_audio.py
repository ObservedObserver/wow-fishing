import numpy as np

from app import audio
from app.audio import SplashDetector
from app.config import AudioConfig


def test_splash_detector_detects_burst() -> None:
    cfg = AudioConfig(bootstrap_frames=5, refractory_ms=100)
    detector = SplashDetector(cfg)

    # Bootstrap with low-noise frames.
    for i in range(10):
        frame = np.random.normal(0, 0.005, 480).astype(np.float32)
        assert detector.update(frame, now_ms=i * 30) is None

    burst = np.random.normal(0, 0.3, 480).astype(np.float32)
    event = detector.update(burst, now_ms=500)
    assert event is not None
    assert event.energy > event.threshold


def test_soundcard_backend_disabled_on_numpy2(monkeypatch) -> None:
    monkeypatch.setattr(audio, "sc", object())
    monkeypatch.setattr(audio.np, "__version__", "2.1.0")

    error = audio._soundcard_backend_error()

    assert error is not None
    assert "NumPy >=2" in error


def test_soundcard_backend_allowed_before_numpy2(monkeypatch) -> None:
    monkeypatch.setattr(audio, "sc", object())
    monkeypatch.setattr(audio.np, "__version__", "1.26.4")

    assert audio._soundcard_backend_error() is None
