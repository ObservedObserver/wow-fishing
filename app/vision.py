from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

from app.config import VisionConfig

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore


@dataclass(slots=True)
class Detection:
    x: int
    y: int
    conf: float
    source: str = "unknown"


class ModelManager:
    def __init__(self, cfg: VisionConfig) -> None:
        self.cfg = cfg

    def ensure_model(self) -> Path:
        model_path = Path(self.cfg.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            return model_path

        with urlopen(self.cfg.model_url, timeout=30) as response:
            payload = response.read()
        model_path.write_bytes(payload)
        return model_path


class BobberDetector:
    """
    ONNX detector first; HSV fallback if runtime packages unavailable.
    """

    def __init__(self, cfg: VisionConfig) -> None:
        self.cfg = cfg
        self._session: Any | None = None
        self._input_name: str | None = None
        self._enabled_classes = set(int(c) for c in cfg.onnx_class_ids)
        self._fallback_only = False
        self._templates_gray: list[np.ndarray] = []
        self._templates_lab_ab: list[tuple[np.ndarray, np.ndarray]] = []
        self.last_template_score: float = 0.0

    def load(self) -> None:
        self._templates_gray = []
        self._templates_lab_ab = []
        if cv2 is not None:
            template_files: list[str] = list(self.cfg.template_paths)
            if self.cfg.template_dir:
                tdir = Path(self.cfg.template_dir)
                if tdir.exists() and tdir.is_dir():
                    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                        for p in sorted(tdir.glob(ext)):
                            template_files.append(str(p))

            seen: set[str] = set()
            for path in template_files:
                if path in seen:
                    continue
                seen.add(path)
                tpl_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if tpl_bgr is None or tpl_bgr.size == 0:
                    continue
                crop = int(self.cfg.template_crop_size)
                if crop > 0:
                    h, w = tpl_bgr.shape[:2]
                    if h > crop or w > crop:
                        cy = h // 2
                        cx = w // 2
                        y0 = max(0, cy - crop // 2)
                        x0 = max(0, cx - crop // 2)
                        y1 = min(h, y0 + crop)
                        x1 = min(w, x0 + crop)
                        tpl_bgr = tpl_bgr[y0:y1, x0:x1]
                tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
                tpl_lab = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2LAB)
                self._templates_gray.append(tpl_gray)
                self._templates_lab_ab.append((tpl_lab[:, :, 1], tpl_lab[:, :, 2]))
        if self._templates_gray:
            print(f"[vision] loaded templates: {len(self._templates_gray)}")
        elif self.cfg.template_dir or self.cfg.template_paths:
            print("[vision] warning: template configured but no valid template image was loaded")

        if ort is None or cv2 is None:
            self._fallback_only = True
            return

        manager = ModelManager(self.cfg)
        try:
            model_path = manager.ensure_model()
            self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:
            # Keep template/hsv path available even if model is unavailable.
            self._session = None
            self._input_name = None
            print(f"[vision] warning: model unavailable, continue without ONNX ({exc})")

    def has_onnx(self) -> bool:
        return (not self._fallback_only) and self._session is not None and cv2 is not None

    def detect(
        self,
        frame_bgr: np.ndarray,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> Detection | None:
        return self._detect_with_mode(
            frame_bgr,
            preferred_x=preferred_x,
            preferred_y=preferred_y,
            mode="combined",
        )

    def detect_onnx_only(
        self,
        frame_bgr: np.ndarray,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> Detection | None:
        return self._detect_with_mode(
            frame_bgr,
            preferred_x=preferred_x,
            preferred_y=preferred_y,
            mode="onnx",
        )

    def detect_template_fallback_only(
        self,
        frame_bgr: np.ndarray,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> Detection | None:
        return self._detect_with_mode(
            frame_bgr,
            preferred_x=preferred_x,
            preferred_y=preferred_y,
            mode="template",
        )

    def _detect_with_mode(
        self,
        frame_bgr: np.ndarray,
        preferred_x: int | None,
        preferred_y: int | None,
        mode: str,
    ) -> Detection | None:
        if self.cfg.roi:
            x, y, w, h = self.cfg.roi
            roi = frame_bgr[y : y + h, x : x + w]
            roi_pref_x = None if preferred_x is None else preferred_x - x
            roi_pref_y = None if preferred_y is None else preferred_y - y
            det = self._detect_core(
                roi,
                preferred_x=roi_pref_x,
                preferred_y=roi_pref_y,
                mode=mode,
            )
            if det is None:
                return None
            return Detection(x=det.x + x, y=det.y + y, conf=det.conf, source=det.source)
        return self._detect_core(
            frame_bgr,
            preferred_x=preferred_x,
            preferred_y=preferred_y,
            mode=mode,
        )

    def _detect_core(
        self,
        frame_bgr: np.ndarray,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
        mode: str = "combined",
    ) -> Detection | None:
        work = frame_bgr
        crop_h = int(frame_bgr.shape[0] * (1.0 - self.cfg.ignore_bottom_ratio))
        if 0 < crop_h < frame_bgr.shape[0]:
            work = frame_bgr[:crop_h, :]
        if mode in {"combined", "onnx"} and self.has_onnx():
            det = self._detect_onnx(work, preferred_x=preferred_x, preferred_y=preferred_y)
            if det is not None:
                return det
        if mode in {"combined", "template"} and cv2 is not None and self._templates_gray:
            det = self._detect_template(work)
            if det is not None:
                return det
        if mode in {"combined", "template"}:
            return self._detect_hsv_fallback(work)
        return None

    def _detect_template(self, frame_bgr: np.ndarray) -> Detection | None:
        assert cv2 is not None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        frame_a = lab[:, :, 1]
        frame_b = lab[:, :, 2]
        best: Detection | None = None
        best_score = 0.0
        gray_w = max(0.0, float(self.cfg.template_gray_weight))
        color_w = max(0.0, float(self.cfg.template_color_weight)) if self.cfg.template_use_color else 0.0
        if gray_w == 0.0 and color_w == 0.0:
            gray_w = 1.0
        weight_sum = gray_w + color_w

        for idx, tpl in enumerate(self._templates_gray):
            tpl_a, tpl_b = self._templates_lab_ab[idx]
            for scale in self.cfg.template_scales:
                if scale <= 0:
                    continue
                if abs(scale - 1.0) < 1e-6:
                    use_tpl = tpl
                    use_tpl_a = tpl_a
                    use_tpl_b = tpl_b
                else:
                    h = max(8, int(round(tpl.shape[0] * scale)))
                    w = max(8, int(round(tpl.shape[1] * scale)))
                    use_tpl = cv2.resize(tpl, (w, h), interpolation=cv2.INTER_LINEAR)
                    use_tpl_a = cv2.resize(tpl_a, (w, h), interpolation=cv2.INTER_LINEAR)
                    use_tpl_b = cv2.resize(tpl_b, (w, h), interpolation=cv2.INTER_LINEAR)
                th, tw = use_tpl.shape[:2]
                if th > gray.shape[0] or tw > gray.shape[1]:
                    continue
                res_gray = cv2.matchTemplate(gray, use_tpl, cv2.TM_CCOEFF_NORMED)
                if color_w > 0.0:
                    res_a = cv2.matchTemplate(frame_a, use_tpl_a, cv2.TM_CCOEFF_NORMED)
                    res_b = cv2.matchTemplate(frame_b, use_tpl_b, cv2.TM_CCOEFF_NORMED)
                    res_color = (res_a + res_b) * 0.5
                    res = (gray_w * res_gray + color_w * res_color) / max(1e-6, weight_sum)
                else:
                    res = res_gray

                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                score = float(max_val)
                if score > best_score:
                    best_score = score
                if score < self.cfg.template_threshold:
                    continue
                x = int(max_loc[0] + tw / 2)
                y = int(max_loc[1] + th / 2)
                cand = Detection(x=x, y=y, conf=score, source="template")
                if best is None or cand.conf > best.conf:
                    best = cand
        self.last_template_score = best_score
        return best

    def _detect_onnx(
        self,
        frame_bgr: np.ndarray,
        preferred_x: int | None = None,
        preferred_y: int | None = None,
    ) -> Detection | None:
        assert self._session is not None and self._input_name is not None and cv2 is not None
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src_h, src_w = image.shape[:2]
        size = self.cfg.input_size
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]

        output = self._session.run(None, {self._input_name: tensor})[0]
        if output.ndim != 3:
            return None

        preds = output[0].T
        candidates: list[dict[str, float | int]] = []

        for row in preds:
            cx, cy, w, h = row[0:4]
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if self._enabled_classes and cls_id not in self._enabled_classes:
                continue
            if conf < self.cfg.conf_threshold:
                continue

            x_scale = src_w / size
            y_scale = src_h / size
            x = float(cx * x_scale)
            y = float(cy * y_scale)
            bw = float(w * x_scale)
            bh = float(h * y_scale)
            x1 = max(0.0, x - (bw / 2.0))
            y1 = max(0.0, y - (bh / 2.0))
            x2 = min(float(src_w - 1), x + (bw / 2.0))
            y2 = min(float(src_h - 1), y + (bh / 2.0))
            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf,
                }
            )

        if not candidates:
            return None

        kept = self._nms(candidates, iou_threshold=0.45)
        best = max(
            kept,
            key=lambda cand: self._candidate_score(
                int(cand["x"]),
                int(cand["y"]),
                float(cand["conf"]),
                preferred_x=preferred_x,
                preferred_y=preferred_y,
                frame_w=src_w,
                frame_h=src_h,
            ),
        )
        return Detection(
            x=int(best["x"]),
            y=int(best["y"]),
            conf=float(best["conf"]),
            source="onnx",
        )

    def _candidate_score(
        self,
        x: int,
        y: int,
        conf: float,
        preferred_x: int | None,
        preferred_y: int | None,
        frame_w: int,
        frame_h: int,
    ) -> float:
        if preferred_x is None or preferred_y is None:
            return conf

        diag = max(1.0, float((frame_w * frame_w + frame_h * frame_h) ** 0.5))
        dx = float(x - preferred_x)
        dy = float(y - preferred_y)
        dist_ratio = ((dx * dx + dy * dy) ** 0.5) / diag
        return conf - (0.35 * dist_ratio)

    def _nms(
        self,
        candidates: list[dict[str, float | int]],
        iou_threshold: float,
    ) -> list[dict[str, float | int]]:
        ordered = sorted(candidates, key=lambda cand: float(cand["conf"]), reverse=True)
        kept: list[dict[str, float | int]] = []

        while ordered:
            current = ordered.pop(0)
            kept.append(current)
            ordered = [
                cand
                for cand in ordered
                if self._iou(current, cand) < iou_threshold
            ]
        return kept

    def _iou(
        self,
        a: dict[str, float | int],
        b: dict[str, float | int],
    ) -> float:
        ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
        bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def _detect_hsv_fallback(self, frame_bgr: np.ndarray) -> Detection | None:
        if cv2 is None:
            return None

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        low = np.array(self.cfg.fallback_hsv_low, dtype=np.uint8)
        high = np.array(self.cfg.fallback_hsv_high, dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid: list[tuple[float, Any]] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if self.cfg.fallback_min_area <= area <= self.cfg.fallback_max_area:
                valid.append((area, cnt))
        if not valid:
            return None
        area, largest = max(valid, key=lambda pair: pair[0])
        m = cv2.moments(largest)
        if m["m00"] == 0:
            return None
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        conf = min(0.85, 0.25 + area / max(200.0, self.cfg.fallback_max_area))
        return Detection(x=x, y=y, conf=conf, source="fallback")
