import numpy as np

from app.vision import Detection
from app.capture import CaptureFrame
from main import _detect_near_anchor, _locate_stable_near_anchor, _select_stable_detection


class _FakeVision:
    def __init__(
        self,
        onnx_det: Detection | None = None,
        template_det: Detection | None = None,
        has_onnx: bool = False,
    ) -> None:
        self._onnx_det = onnx_det
        self._template_det = template_det
        self._has_onnx = has_onnx
        self.onnx_frames: list[tuple[int, int]] = []
        self.template_frames: list[tuple[int, int]] = []

    def has_onnx(self) -> bool:
        return self._has_onnx

    def detect_onnx_only(
        self,
        frame: np.ndarray,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> Detection | None:
        del preferred_x
        del preferred_y
        self.onnx_frames.append((int(frame.shape[1]), int(frame.shape[0])))
        return self._onnx_det

    def detect_template_fallback_only(
        self,
        frame: np.ndarray,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> Detection | None:
        del preferred_x
        del preferred_y
        self.template_frames.append((int(frame.shape[1]), int(frame.shape[0])))
        return self._template_det

    def detect(
        self,
        frame: np.ndarray,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> Detection | None:
        del frame
        del preferred_x
        del preferred_y
        return self._template_det


class _FakeCapture:
    def __init__(self, window_shape: tuple[int, int], monitor_shape: tuple[int, int]) -> None:
        self.window_shape = window_shape
        self.monitor_shape = monitor_shape
        self.window_calls = 0
        self.monitor_calls = 0

    def grab_window_with_offset(
        self,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> CaptureFrame:
        del preferred_x
        del preferred_y
        self.window_calls += 1
        h, w = self.window_shape
        return CaptureFrame(frame_bgr=np.zeros((h, w, 3), dtype=np.uint8), left=10, top=20)

    def grab_with_offset(
        self,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> CaptureFrame:
        del preferred_x
        del preferred_y
        self.monitor_calls += 1
        h, w = self.monitor_shape
        return CaptureFrame(frame_bgr=np.zeros((h, w, 3), dtype=np.uint8), left=0, top=0)


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


def test_locate_uses_full_window_for_onnx_before_template_roi() -> None:
    vision = _FakeVision(
        onnx_det=Detection(x=100, y=120, conf=0.8, source="onnx"),
        template_det=Detection(x=50, y=50, conf=0.9, source="template"),
        has_onnx=True,
    )
    capture = _FakeCapture(window_shape=(600, 800), monitor_shape=(1080, 1920))

    det, hit_count = _locate_stable_near_anchor(
        vision=vision,  # type: ignore[arg-type]
        capture=capture,  # type: ignore[arg-type]
        anchor_x=110,
        anchor_y=140,
        radius=200,
    )

    assert hit_count == 1
    assert det is not None
    assert det.source == "onnx"
    assert vision.onnx_frames == [(800, 600)]
    assert vision.template_frames == []
    assert capture.window_calls == 1
    assert capture.monitor_calls == 0


def test_locate_falls_back_to_template_roi_when_onnx_misses() -> None:
    vision = _FakeVision(
        onnx_det=None,
        template_det=Detection(x=160, y=160, conf=0.8, source="template"),
        has_onnx=True,
    )
    capture = _FakeCapture(window_shape=(600, 800), monitor_shape=(1080, 1920))

    det, hit_count = _locate_stable_near_anchor(
        vision=vision,  # type: ignore[arg-type]
        capture=capture,  # type: ignore[arg-type]
        anchor_x=200,
        anchor_y=200,
        radius=120,
    )

    assert hit_count == 1
    assert det is not None
    assert det.source == "template"
    assert vision.onnx_frames == [(800, 600)]
    assert vision.template_frames == [(240, 240)]
    assert capture.window_calls == 1
    assert capture.monitor_calls == 1
