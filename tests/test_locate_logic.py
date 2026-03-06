import numpy as np

from app.vision import Detection
from main import _detect_near_anchor, _select_stable_detection


class _FakeVision:
    def __init__(self, det: Detection | None) -> None:
        self._det = det

    def detect(
        self,
        frame: np.ndarray,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> Detection | None:
        del frame
        del preferred_x
        del preferred_y
        return self._det


def test_detect_near_anchor_rejects_outside_circle() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # This maps to local (80, 80) in ROI and is outside radius=80 from (100,100).
    det = Detection(x=160, y=160, conf=0.9, source="template")
    out = _detect_near_anchor(
        vision=_FakeVision(det),  # type: ignore[arg-type]
        frame=frame,
        anchor_x=100,
        anchor_y=100,
        radius=80,
    )
    assert out is None


def test_select_stable_detection_requires_cluster() -> None:
    detections = [
        Detection(x=100, y=100, conf=0.75, source="template"),
        Detection(x=105, y=98, conf=0.77, source="template"),
        Detection(x=300, y=310, conf=0.92, source="template"),
    ]
    out = _select_stable_detection(detections, anchor_x=100, anchor_y=100, radius=120)
    assert out is not None
    assert abs(out.x - 100) < 12
    assert abs(out.y - 100) < 12
