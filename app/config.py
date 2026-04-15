from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    key_detect_delay_ms: int = 1_400
    key_retry_interval_ms: int = 500
    key_retry_max_attempts: int = 3
    recast_on_miss: bool = True
    recast_miss_delay_ms: int = 300
    auto_cast_base_ms: int = 2_000
    auto_cast_jitter_max_ms: int = 1_500
    auto_cast_initial_delay_ms: int = 500
    ignore_after_cast_ms: int = 900
    precast_cleanup_delay_ms: int = 900
    bite_window_start_ms: int = 4_000
    bite_window_end_ms: int = 26_000
    max_cast_lifetime_ms: int = 30_000
    click_cooldown_ms: int = 1_200
    anti_afk_jump_every_casts: int = 1_000
    anti_afk_jump_wait_ms: int = 5_000
    slot2_cycle_base_ms: int = 600_000
    slot2_cycle_jitter_min_ms: int = 15_000
    slot2_cycle_jitter_max_ms: int = 30_000
    slot2_after_reel_delay_ms: int = 1_000
    slot2_post_use_wait_ms: int = 8_000
    bag_cycle_base_ms: int = 180_000
    bag_cycle_jitter_min_ms: int = 0
    bag_cycle_jitter_max_ms: int = 60_000
    bag_open_min_ms: int = 3_000
    bag_open_max_ms: int = 5_000
    max_runtime_hours: float = 2.5


@dataclass(slots=True)
class VisionConfig:
    model_url: str | None = None
    model_path: str = "models/bobber.onnx"
    model_sha256: str | None = "bab89e87f85f4672e53c0d04c570b111557179394c5313768c3182e79fd8f588"
    input_size: int = 1280
    conf_threshold: float = 0.55
    onnx_class_ids: tuple[int, ...] = (0,)
    onnx_providers: tuple[str, ...] = ()
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
    enable_precast_cleanup: bool = False
    precast_cleanup_radius: int = 220
    precast_cleanup_min_conf: float = 0.72
    onnx_force_top1: bool = False
    onnx_use_preferred_anchor: bool = False
    onnx_crop_left_ratio: float = 0.04
    onnx_crop_right_ratio: float = 0.04
    onnx_crop_top_ratio: float = 0.0
    onnx_crop_bottom_ratio: float = 0.12
    roi: tuple[int, int, int, int] | None = None
    debug_save_model_input: bool = False


@dataclass(slots=True)
class ControlConfig:
    bite_action_mode: str = "mouse"
    interaction_key: str = "F12"
    bag_key: str = "B"
    return_key: str = "4"
    move_duration_ms: int = 35
    jitter_px: int = 6
    key_press_hold_ms: int = 30
    slot2_key_press_hold_ms: int = 80
    click_delay_min_ms: int = 450
    click_delay_max_ms: int = 650
    key_hold_min_ms: int = 40
    key_hold_max_ms: int = 140
    key_hold_sigma: float = 0.3
    slot2_key_hold_min_ms: int = 60
    slot2_key_hold_max_ms: int = 200
    slot2_key_hold_sigma: float = 0.28
    randomize_key_hold: bool = False
    randomize_mouse_click_timing: bool = False


@dataclass(slots=True)
class HumanizeConfig:
    enabled: bool = False

    bite_reaction_median_ms: int = 520
    bite_reaction_sigma: float = 0.35
    bite_reaction_min_ms: int = 300
    bite_reaction_max_ms: int = 1_200

    cast_interval_median_ms: int = 4_750
    cast_interval_sigma: float = 0.25
    cast_interval_min_ms: int = 3_000
    cast_interval_max_ms: int = 8_000

    slot2_jitter_median_ms: int = 22_500
    slot2_jitter_sigma: float = 0.2
    slot2_jitter_min_ms: int = 15_000
    slot2_jitter_max_ms: int = 30_000

    fatigue_drift_per_hour: float = 0.08
    fatigue_max_drift: float = 0.35
    fatigue_spike_prob: float = 0.02
    fatigue_spike_multiplier_min: float = 1.8
    fatigue_spike_multiplier_max: float = 4.0


@dataclass(slots=True)
class SessionConfig:
    enabled: bool = False

    fishing_segment_median_min: int = 28
    fishing_segment_min_min: int = 15
    fishing_segment_max_min: int = 50

    micro_break_min_s: int = 30
    micro_break_max_s: int = 120

    short_break_min_s: int = 120
    short_break_max_s: int = 360
    short_break_probability: float = 0.3

    long_break_min_s: int = 480
    long_break_max_s: int = 1_200
    long_break_every: int = 3

    max_session_min: int = 180


@dataclass(slots=True)
class MousePathConfig:
    enabled: bool = False
    target_width_px: int = 20
    fitts_a_ms: float = 50.0
    fitts_b_ms: float = 150.0
    arc_sigma: float = 0.08
    speed_gamma: float = 1.5
    tremor_amplitude_px: float = 1.2
    overshoot_prob: float = 0.18
    overshoot_min_dist_px: float = 60.0
    min_move_duration_ms: int = 50
    max_move_duration_ms: int = 2_500


@dataclass(slots=True)
class NoiseConfig:
    enabled: bool = False
    base_probability: float = 0.08
    cooldown_min_ms: int = 30_000
    max_probability: float = 0.35
    casts_scale_per_cast: float = 0.012
    weights: dict[str, int] = field(
        default_factory=lambda: {
            "camera_pan": 3,
            "toggle_bag": 2,
            "short_walk": 1,
            "jump": 2,
            "idle_pause": 4,
            "look_around": 2,
        }
    )


@dataclass(slots=True)
class InputConfig:
    backend: str = "send_input"


@dataclass(slots=True)
class AppConfig:
    audio: AudioConfig
    timing: TimingConfig
    vision: VisionConfig
    control: ControlConfig
    humanize: HumanizeConfig
    session: SessionConfig
    mouse_path: MousePathConfig
    noise: NoiseConfig
    input: InputConfig

    @staticmethod
    def default() -> "AppConfig":
        return AppConfig(
            audio=AudioConfig(),
            timing=TimingConfig(),
            vision=VisionConfig(),
            control=ControlConfig(),
            humanize=HumanizeConfig(),
            session=SessionConfig(),
            mouse_path=MousePathConfig(),
            noise=NoiseConfig(),
            input=InputConfig(),
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
        humanize=HumanizeConfig(**merged["humanize"]),
        session=SessionConfig(**merged["session"]),
        mouse_path=MousePathConfig(**merged["mouse_path"]),
        noise=NoiseConfig(**merged["noise"]),
        input=InputConfig(**merged["input"]),
    )
