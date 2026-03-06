from __future__ import annotations

import numpy as np

from app.vision import _crop_onnx_frame


def test_crop_onnx_frame_applies_configured_edges() -> None:
    frame = np.zeros((1000, 2000, 3), dtype=np.uint8)

    cropped, x0, y0 = _crop_onnx_frame(
        frame,
        left_ratio=0.04,
        right_ratio=0.04,
        top_ratio=0.0,
        bottom_ratio=0.12,
    )

    assert x0 == 80
    assert y0 == 0
    assert cropped.shape[:2] == (880, 1840)


def test_crop_onnx_frame_preserves_minimum_size() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    cropped, x0, y0 = _crop_onnx_frame(
        frame,
        left_ratio=0.45,
        right_ratio=0.45,
        top_ratio=0.45,
        bottom_ratio=0.45,
    )

    assert (x0, y0) == (4, 4)
    assert cropped.shape[:2] == (16, 16)
