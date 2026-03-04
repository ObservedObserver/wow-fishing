import numpy as np

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

