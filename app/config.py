from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AudioConfig:
    backend: str = "auto"
    loopback_speaker_contains: str | None = None
    sample_rate: int = 16_000
    frame_ms: int = 30
    threshold_k: float = 2.2
    refractory_ms: int = 450
    bite_lock_ms: int = 3_500
    bootstrap_frames: int = 30
    input_device: int | None = None


@dataclass(slots=True)
class TimingConfig:
    key_detect_delay_ms: int = 1_000
    key_retry_interval_ms: int = 500
    key_retry_max_attempts: int = 3
    recast_on_miss: bool = True
    recast_miss_delay_ms: int = 300
    auto_cast_base_ms: int = 4_000
    auto_cast_jitter_max_ms: int = 1_500
    auto_cast_initial_delay_ms: int = 500
    ignore_after_cast_ms: int = 900
    bite_window_start_ms: int = 4_000
    bite_window_end_ms: int = 26_000
    click_cooldown_ms: int = 1_200


@dataclass(slots=True)
class VisionConfig:
    model_url: str = (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
    )
    model_path: str = "models/bobber.onnx"
    input_size: int = 640
    conf_threshold: float = 0.55
    onnx_class_ids: tuple[int, ...] = (0,)
    template_dir: str | None = None
    template_paths: tuple[str, ...] = ()
    template_threshold: float = 0.72
    template_use_color: bool = True
    template_gray_weight: float = 0.35
    template_color_weight: float = 0.65
    template_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    template_crop_size: int = 96
    fallback_hsv_low: tuple[int, int, int] = (0, 0, 180)
    fallback_hsv_high: tuple[int, int, int] = (179, 80, 255)
    fallback_min_area: float = 20.0
    fallback_max_area: float = 1200.0
    ignore_bottom_ratio: float = 0.18
    allow_fallback_for_action: bool = False
    key_search_radius: int = 520
    roi: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class ControlConfig:
    move_duration_ms: int = 35
    jitter_px: int = 6
    click_delay_min_ms: int = 450
    click_delay_max_ms: int = 650


@dataclass(slots=True)
class AppConfig:
    audio: AudioConfig
    timing: TimingConfig
    vision: VisionConfig
    control: ControlConfig

    @staticmethod
    def default() -> "AppConfig":
        return AppConfig(
            audio=AudioConfig(),
            timing=TimingConfig(),
            vision=VisionConfig(),
            control=ControlConfig(),
        )


def _merge_dict(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig.default()
    if path is None:
        return cfg

    cfg_path = Path(path)
    if not cfg_path.exists():
        return cfg

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base = asdict(cfg)
    merged = _merge_dict(base, raw)

    return AppConfig(
        audio=AudioConfig(**merged["audio"]),
        timing=TimingConfig(**merged["timing"]),
        vision=VisionConfig(**merged["vision"]),
        control=ControlConfig(**merged["control"]),
    )
