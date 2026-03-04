from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.vision import BobberDetector

RAW_POS = ROOT / "data" / "raw" / "positive"


def _bbox_from_shape(shape: dict[str, Any]) -> tuple[float, float, float, float] | None:
    points = shape.get("points") or []
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    thresholds = [0.55, 0.40, 0.30, 0.25]
    json_files = sorted(RAW_POS.glob("*.json"))
    if not json_files:
        raise RuntimeError("No positive json files found")

    for th in thresholds:
        cfg.vision.conf_threshold = th
        cfg.vision.template_dir = None
        cfg.vision.template_paths = ()
        detector = BobberDetector(cfg.vision)
        detector.load()

        detected = 0
        hit_iou = 0
        total = 0

        for jf in json_files:
            payload = json.loads(jf.read_text(encoding="utf-8"))
            image_data = payload.get("imageData")
            if not image_data:
                continue
            arr = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            gt_box: tuple[float, float, float, float] | None = None
            for shape in payload.get("shapes", []):
                if str(shape.get("label", "")).lower() != "bobber":
                    continue
                gt_box = _bbox_from_shape(shape)
                if gt_box is not None:
                    break
            if gt_box is None:
                continue

            total += 1
            work = img
            crop_h = int(img.shape[0] * (1.0 - cfg.vision.ignore_bottom_ratio))
            if 0 < crop_h < img.shape[0]:
                work = img[:crop_h, :]

            pred = detector._detect_onnx(work)  # ONNX-only evaluation
            if pred is None:
                continue
            detected += 1

            # approximate prediction bbox around point using GT box size
            gw = gt_box[2] - gt_box[0]
            gh = gt_box[3] - gt_box[1]
            pred_box = (
                float(pred.x - gw / 2.0),
                float(pred.y - gh / 2.0),
                float(pred.x + gw / 2.0),
                float(pred.y + gh / 2.0),
            )
            if _iou(pred_box, gt_box) >= 0.30:
                hit_iou += 1

        det_rate = (detected / total) if total else 0.0
        hit_rate = (hit_iou / total) if total else 0.0
        print(
            f"threshold={th:.2f} total={total} detected={detected} "
            f"det_rate={det_rate:.3f} iou_hit@0.30={hit_iou} hit_rate={hit_rate:.3f}"
        )


if __name__ == "__main__":
    main()

