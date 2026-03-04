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
        manager = ModelManager(self.cfg)
        model_path = manager.ensure_model()
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
        if ort is None or cv2 is None:
            self._fallback_only = True
            return
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, frame_bgr: np.ndarray) -> Detection | None:
        if self.cfg.roi:
            x, y, w, h = self.cfg.roi
            roi = frame_bgr[y : y + h, x : x + w]
            det = self._detect_core(roi)
            if det is None:
                return None
            return Detection(x=det.x + x, y=det.y + y, conf=det.conf)
        return self._detect_core(frame_bgr)

    def _detect_core(self, frame_bgr: np.ndarray) -> Detection | None:
        work = frame_bgr
        crop_h = int(frame_bgr.shape[0] * (1.0 - self.cfg.ignore_bottom_ratio))
        if 0 < crop_h < frame_bgr.shape[0]:
            work = frame_bgr[:crop_h, :]
        if not self._fallback_only and self._session is not None and cv2 is not None:
            det = self._detect_onnx(work)
            if det is not None:
                return det
        if cv2 is not None and self._templates_gray:
            det = self._detect_template(work)
            if det is not None:
                return det
        return self._detect_hsv_fallback(work)

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

    def _detect_onnx(self, frame_bgr: np.ndarray) -> Detection | None:
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
        best: Detection | None = None

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
            x = int(cx * x_scale)
            y = int(cy * y_scale)
            cand = Detection(x=x, y=y, conf=conf, source="onnx")
            if best is None or cand.conf > best.conf:
                best = cand
        return best

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

