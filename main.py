from __future__ import annotations

import argparse
import ctypes
import random
import time

import numpy as np
try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore

from app.audio import (
    MockAudioSource,
    SplashDetector,
    WasapiLoopbackSource,
    list_input_devices,
    list_loopback_speakers,
)
from app.behavior_noise import BehaviorNoiseInjector, NoiseContext
from app.capture import ScreenCapture
from app.config import AppConfig, ControlConfig, load_config
from app.event_log import get_event_logger, log_event
from app.humanize import FatigueModel, HumanDelay
from app.input_control import MouseController
from app.session import SessionPhase, SessionScheduler
from app.state_machine import FishingStateMachine
from app.vision import BobberDetector, Detection, ModelManager


_VK_1 = 0x31
_VK_NUMPAD1 = 0x61
_VK_ESC = 0x1B
_LOCATE_CONFIRM_FRAMES = 1
_LOCATE_CONFIRM_INTERVAL_MS = 0
LOGGER = get_event_logger("fishing.runtime")


class KeyOneTrigger:
    def __init__(self) -> None:
        self.user32 = ctypes.windll.user32
        self._prev_down = False

    def poll_pressed_edge(self) -> bool:
        down_1 = bool(self.user32.GetAsyncKeyState(_VK_1) & 0x8000)
        down_num1 = bool(self.user32.GetAsyncKeyState(_VK_NUMPAD1) & 0x8000)
        down = down_1 or down_num1
        edge = down and not self._prev_down
        self._prev_down = down
        return edge


class EscTrigger:
    def __init__(self) -> None:
        self.user32 = ctypes.windll.user32
        self._prev_down = False

    def poll_pressed_edge(self) -> bool:
        down = bool(self.user32.GetAsyncKeyState(_VK_ESC) & 0x8000)
        edge = down and not self._prev_down
        self._prev_down = down
        return edge


def _cast_has_timed_out(
    now_ms: int,
    cast_started_at_ms: int | None,
    max_cast_lifetime_ms: int,
) -> bool:
    if cast_started_at_ms is None:
        return False
    return (now_ms - cast_started_at_ms) >= max(0, max_cast_lifetime_ms)


def _roll_slot2_interval_ms(
    cfg: AppConfig,
    fatigue: FatigueModel | None,
) -> int:
    base = max(0, cfg.timing.slot2_cycle_base_ms)
    if cfg.humanize.enabled and fatigue is not None:
        jitter = fatigue.adjusted_sample(HumanDelay.slot2_jitter(cfg.humanize))
        return base + jitter
    low = min(cfg.timing.slot2_cycle_jitter_min_ms, cfg.timing.slot2_cycle_jitter_max_ms)
    high = max(cfg.timing.slot2_cycle_jitter_min_ms, cfg.timing.slot2_cycle_jitter_max_ms)
    return base + random.randint(max(0, low), max(0, high))


def _prime_slot2_timer(
    now_ms: int,
    cfg: AppConfig,
    fatigue: FatigueModel | None,
) -> tuple[int, int]:
    next_slot2_at_ms = now_ms + _roll_slot2_interval_ms(cfg, fatigue)
    next_cast_at_ms = now_ms + max(0, cfg.timing.slot2_post_use_wait_ms)
    return next_slot2_at_ms, next_cast_at_ms


def _prime_slot2_after_reel(now_ms: int, cfg: AppConfig) -> int:
    return now_ms + max(0, cfg.timing.slot2_after_reel_delay_ms)


def _roll_bag_interval_ms(cfg: AppConfig) -> int:
    base = max(0, cfg.timing.bag_cycle_base_ms)
    low = min(cfg.timing.bag_cycle_jitter_min_ms, cfg.timing.bag_cycle_jitter_max_ms)
    high = max(cfg.timing.bag_cycle_jitter_min_ms, cfg.timing.bag_cycle_jitter_max_ms)
    return base + random.randint(max(0, low), max(0, high))


def _prime_bag_timer(now_ms: int, cfg: AppConfig) -> int:
    return now_ms + _roll_bag_interval_ms(cfg)


def _roll_bag_open_duration_ms(cfg: AppConfig) -> int:
    low = min(cfg.timing.bag_open_min_ms, cfg.timing.bag_open_max_ms)
    high = max(cfg.timing.bag_open_min_ms, cfg.timing.bag_open_max_ms)
    return random.randint(max(0, low), max(0, high))


def _bag_timer_conflicts_with_slot2(
    now_ms: int,
    cfg: AppConfig,
    slot2_next_at_ms: int | None,
    slot2_pending_use_at_ms: int | None,
) -> bool:
    guard_ms = max(0, cfg.timing.bag_open_max_ms) + 1_000
    if slot2_pending_use_at_ms is not None and slot2_pending_use_at_ms <= (now_ms + guard_ms):
        return True
    if slot2_next_at_ms is not None and slot2_next_at_ms <= (now_ms + guard_ms):
        return True
    return False


def _schedule_resume_cast_at_ms(now_ms: int, cfg: AppConfig) -> int:
    return now_ms + max(0, cfg.timing.auto_cast_initial_delay_ms)


def _max_runtime_ms(cfg: AppConfig) -> int:
    return int(max(0.0, cfg.timing.max_runtime_hours) * 3_600_000)


def _runtime_limit_reached(
    now_ms: int,
    run_started_at_ms: int | None,
    cfg: AppConfig,
) -> bool:
    if run_started_at_ms is None:
        return False
    limit_ms = _max_runtime_ms(cfg)
    if limit_ms <= 0:
        return False
    return (now_ms - run_started_at_ms) >= limit_ms


def _resume_after_session_break(
    now_ms: int,
    cfg: AppConfig,
    session: SessionScheduler,
) -> tuple[bool, int | None]:
    should_resume = session.should_resume_auto()
    session.snapshot_before_break_auto(False)
    if not should_resume:
        return False, None
    return True, _schedule_resume_cast_at_ms(now_ms, cfg)


def _normalize_bite_action_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"mouse", "mouse_right_click", "right_click"}:
        return "mouse"
    if normalized in {"interact_hotkey", "interaction_hotkey", "interaction_key", "hotkey"}:
        return "interact_hotkey"
    raise ValueError(
        "unsupported control.bite_action_mode: "
        f"{mode!r}; expected 'mouse' or 'interact_hotkey'"
    )


def _perform_bite_action(mouse: MouseController, cfg: ControlConfig) -> str:
    mode = _normalize_bite_action_mode(cfg.bite_action_mode)
    if mode == "mouse":
        mouse.right_click()
        return "mouse_right_click"

    mouse.press_interaction_key()
    return f"interaction_hotkey:{cfg.interaction_key.strip().upper()}"


def _is_move_close_enough(
    actual_x: int,
    actual_y: int,
    target_x: int,
    target_y: int,
    jitter_px: int,
) -> bool:
    tolerance_px = max(12, max(0, jitter_px) + 8)
    dx = actual_x - target_x
    dy = actual_y - target_y
    return (dx * dx + dy * dy) <= (tolerance_px * tolerance_px)


def _synthetic_audio_frames(count: int, splash_idx: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(count):
        if i == splash_idx:
            frame = np.random.normal(0, 0.30, 480).astype(np.float32)
        else:
            frame = np.random.normal(0, 0.01, 480).astype(np.float32)
        frames.append(frame)
    return frames


def _detect_near_anchor(
    vision: BobberDetector,
    frame: np.ndarray,
    anchor_x: int | None,
    anchor_y: int | None,
    radius: int,
) -> Detection | None:
    if anchor_x is None or anchor_y is None or radius <= 0:
        return vision.detect_template_fallback_only(frame)

    h, w = frame.shape[:2]
    x0 = max(0, anchor_x - radius)
    y0 = max(0, anchor_y - radius)
    x1 = min(w, anchor_x + radius)
    y1 = min(h, anchor_y + radius)
    if x1 - x0 < 16 or y1 - y0 < 16:
        # If anchor falls outside the captured monitor, keep old full-frame fallback behavior.
        return vision.detect_template_fallback_only(frame)

    roi = frame[y0:y1, x0:x1]
    det = vision.detect_template_fallback_only(
        roi,
        preferred_x=anchor_x - x0,
        preferred_y=anchor_y - y0,
    )
    if det is None:
        return None
    det_x = det.x + x0
    det_y = det.y + y0
    dx = det_x - anchor_x
    dy = det_y - anchor_y
    if dx * dx + dy * dy > radius * radius:
        return None
    return Detection(x=det_x, y=det_y, conf=det.conf, source=det.source)


def _detect_onnx_in_window(
    vision: BobberDetector,
    frame: np.ndarray,
    anchor_x: int | None,
    anchor_y: int | None,
    radius: int,
) -> Detection | None:
    del anchor_x
    del anchor_y
    del radius
    return vision.detect_onnx_only(frame)

def _select_stable_detection(
    detections: list[Detection],
    anchor_x: int | None,
    anchor_y: int | None,
    radius: int,
) -> Detection | None:
    if not detections:
        return None
    cluster_px = max(18, int(radius * 0.12)) if radius > 0 else 32
    cluster_px2 = cluster_px * cluster_px
    best_cluster: list[Detection] = []
    for seed in detections:
        cluster: list[Detection] = []
        for cand in detections:
            dx = cand.x - seed.x
            dy = cand.y - seed.y
            if dx * dx + dy * dy <= cluster_px2:
                cluster.append(cand)
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
        elif len(cluster) == len(best_cluster):
            if cluster and best_cluster:
                if max(c.conf for c in cluster) > max(c.conf for c in best_cluster):
                    best_cluster = cluster

    pool = best_cluster if len(best_cluster) >= 2 else detections
    if anchor_x is None or anchor_y is None or radius <= 0:
        return max(pool, key=lambda d: d.conf)

    def score(cand: Detection) -> float:
        dx = cand.x - anchor_x
        dy = cand.y - anchor_y
        dist_ratio = ((dx * dx + dy * dy) ** 0.5) / max(1.0, float(radius))
        return cand.conf - (0.20 * dist_ratio)

    return max(pool, key=score)


def _locate_stable_near_anchor(
    vision: BobberDetector,
    capture: ScreenCapture,
    anchor_x: int | None,
    anchor_y: int | None,
    radius: int,
    confirm_frames: int = _LOCATE_CONFIRM_FRAMES,
) -> tuple[Detection | None, int]:
    detections: list[Detection] = []
    samples = max(1, confirm_frames)
    for i in range(samples):
        det = None

        if vision.has_onnx():
            model_shot = capture.grab_window_with_offset(preferred_x=anchor_x, preferred_y=anchor_y)
            model_anchor_x = None if anchor_x is None else (anchor_x - model_shot.left)
            model_anchor_y = None if anchor_y is None else (anchor_y - model_shot.top)
            model_det = _detect_onnx_in_window(
                vision=vision,
                frame=model_shot.frame_bgr,
                anchor_x=model_anchor_x,
                anchor_y=model_anchor_y,
                radius=radius,
            )
            if model_det is not None:
                det = Detection(
                    x=model_det.x + model_shot.left,
                    y=model_det.y + model_shot.top,
                    conf=model_det.conf,
                    source=model_det.source,
                )
            else:
                log_event(
                    LOGGER,
                    "locate.onnx_miss",
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                    radius=radius,
                )

        if det is None:
            shot = capture.grab_with_offset(preferred_x=anchor_x, preferred_y=anchor_y)
            local_anchor_x = None if anchor_x is None else (anchor_x - shot.left)
            local_anchor_y = None if anchor_y is None else (anchor_y - shot.top)
            local_det = _detect_near_anchor(
                vision=vision,
                frame=shot.frame_bgr,
                anchor_x=local_anchor_x,
                anchor_y=local_anchor_y,
                radius=radius,
            )
            if local_det is not None:
                det = Detection(
                    x=local_det.x + shot.left,
                    y=local_det.y + shot.top,
                    conf=local_det.conf,
                    source=local_det.source,
                )

        if det is not None:
            detections.append(det)
        if i + 1 < samples:
            time.sleep(_LOCATE_CONFIRM_INTERVAL_MS / 1000.0)
    return _select_stable_detection(detections, anchor_x, anchor_y, radius), len(detections)


def _clear_lingering_bobber_before_cast(
    vision: BobberDetector,
    capture: ScreenCapture,
    mouse: MouseController,
    anchor_x: int | None,
    anchor_y: int | None,
    radius: int,
    min_conf: float,
) -> tuple[bool, int | None, int | None]:
    if anchor_x is None or anchor_y is None or radius <= 0:
        return False, anchor_x, anchor_y

    shot = capture.grab_with_offset(preferred_x=anchor_x, preferred_y=anchor_y)
    local_anchor_x = anchor_x - shot.left
    local_anchor_y = anchor_y - shot.top
    det = _detect_near_anchor(
        vision=vision,
        frame=shot.frame_bgr,
        anchor_x=local_anchor_x,
        anchor_y=local_anchor_y,
        radius=radius,
    )
    if det is None:
        return False, anchor_x, anchor_y
    if det.conf < min_conf:
        log_event(
            LOGGER,
            "precast_cleanup.skipped_weak_match",
            conf=round(det.conf, 4),
            min_conf=round(min_conf, 4),
        )
        return False, anchor_x, anchor_y

    abs_x = det.x + shot.left
    abs_y = det.y + shot.top
    moved_x, moved_y = mouse.move_and_right_click(abs_x, abs_y)
    log_event(
        LOGGER,
        "precast_cleanup.clicked",
        x=moved_x,
        y=moved_y,
        conf=round(det.conf, 4),
        source=det.source,
    )
    return True, moved_x, moved_y


def command_download_model(cfg: AppConfig) -> None:
    manager = ModelManager(cfg.vision)
    path = manager.ensure_model()
    print(f"model ready: {path}")


def command_test_audio(cfg: AppConfig) -> None:
    detector = SplashDetector(cfg.audio)
    source = MockAudioSource(_synthetic_audio_frames(count=80, splash_idx=56))
    hit_count = 0
    for i in range(80):
        ev = detector.update(source.read_frame(), now_ms=i * cfg.audio.frame_ms)
        if ev is not None:
            hit_count += 1
            print(f"audio event at {ev.ts_ms}ms rms={ev.energy:.4f} threshold={ev.threshold:.4f}")
    print(f"audio test complete, events={hit_count}")


def command_run(cfg: AppConfig) -> None:
    print("starting fishing bot loop")
    print("Press Ctrl+C to stop.")
    print("Mechanism: press 1 once to start loop; ESC to pause loop.")

    detector = SplashDetector(cfg.audio)
    vision = BobberDetector(cfg.vision)
    vision.load()
    capture = ScreenCapture()
    fatigue: FatigueModel | None = (
        FatigueModel(cfg.humanize) if cfg.humanize.enabled else None
    )
    session = SessionScheduler(cfg.session)
    mouse = MouseController(
        cfg.control,
        input_cfg=cfg.input,
        mouse_path_cfg=cfg.mouse_path,
        humanize_cfg=cfg.humanize,
    )
    noise_injector = BehaviorNoiseInjector(cfg.noise, mouse.backend)
    machine = FishingStateMachine(cfg.timing, action_lock_ms=cfg.audio.bite_lock_ms)
    bite_action_mode = _normalize_bite_action_mode(cfg.control.bite_action_mode)
    audio_source = WasapiLoopbackSource(cfg.audio)
    key_trigger = KeyOneTrigger()
    esc_trigger = EscTrigger()
    frame_interval_s = cfg.audio.frame_ms / 1000.0
    next_cast_at_ms: int | None = None
    cast_anchor_x: int | None = None
    cast_anchor_y: int | None = None
    cast_count = 0
    last_anti_afk_cast_count = 0
    auto_enabled = False
    needs_precast_cleanup = False
    slot2_next_at_ms: int | None = None
    slot2_pending_after_first_reel = False
    slot2_pending_use_at_ms: int | None = None
    bag_next_at_ms: int | None = None
    run_started_at_ms: int | None = None
    run_started_ms = int(time.monotonic() * 1000)
    last_noise_ms = -1
    casts_since_last_noise = 0
    last_session_phase = SessionPhase.FISHING

    log_event(
        LOGGER,
        "runtime.control_mode",
        bite_action_mode=bite_action_mode,
        interaction_key=cfg.control.interaction_key,
        humanize_enabled=cfg.humanize.enabled,
        session_enabled=cfg.session.enabled,
        noise_enabled=cfg.noise.enabled,
        input_backend=cfg.input.backend,
        mouse_path_enabled=cfg.mouse_path.enabled,
    )

    def schedule_next_cast(now_ms: int, reason: str, extra_ms: int, cast_id: int | None = None) -> None:
        nonlocal next_cast_at_ms
        next_cast_at_ms = now_ms + max(0, extra_ms)
        log_event(
            LOGGER,
            "cast.scheduled",
            cast_id=cast_id,
            delay_ms=max(0, extra_ms),
            reason=reason,
        )

    try:
        while True:
            now_ms = int(time.monotonic() * 1000)

            if _runtime_limit_reached(now_ms, run_started_at_ms, cfg):
                mouse.press_return_key()
                auto_enabled = False
                next_cast_at_ms = None
                needs_precast_cleanup = False
                machine.reset()
                slot2_next_at_ms = None
                slot2_pending_after_first_reel = False
                slot2_pending_use_at_ms = None
                bag_next_at_ms = None
                run_started_at_ms = None
                if cfg.session.enabled:
                    session.reset()
                    last_session_phase = SessionPhase.FISHING
                log_event(
                    LOGGER,
                    "loop.max_runtime_reached",
                    elapsed_ms=_max_runtime_ms(cfg),
                    return_key=cfg.control.return_key,
                )
                time.sleep(frame_interval_s)
                continue

            if cfg.session.enabled:
                phase = session.check(now_ms)
                if phase == SessionPhase.SESSION_END:
                    if last_session_phase != phase:
                        log_event(
                            LOGGER,
                            "session.macro_ended",
                            total_fishing_ms=session.total_fishing_ms,
                        )
                        auto_enabled = False
                        next_cast_at_ms = None
                        needs_precast_cleanup = False
                        machine.reset()
                        slot2_next_at_ms = None
                        slot2_pending_after_first_reel = False
                        slot2_pending_use_at_ms = None
                        bag_next_at_ms = None
                        run_started_at_ms = None
                    last_session_phase = phase
                elif phase != SessionPhase.FISHING:
                    if last_session_phase != phase:
                        session.snapshot_before_break_auto(auto_enabled)
                        if auto_enabled:
                            auto_enabled = False
                            next_cast_at_ms = None
                            needs_precast_cleanup = False
                            machine.reset()
                            slot2_next_at_ms = None
                            slot2_pending_after_first_reel = False
                            slot2_pending_use_at_ms = None
                            bag_next_at_ms = None
                        log_event(
                            LOGGER,
                            "session.break",
                            phase=phase.name,
                        )
                    last_session_phase = phase
                    time.sleep(1.0)
                    continue
                if (
                    last_session_phase != SessionPhase.FISHING
                    and phase == SessionPhase.FISHING
                ):
                    auto_enabled, next_cast_at_ms = _resume_after_session_break(
                        now_ms=now_ms,
                        cfg=cfg,
                        session=session,
                    )
                    if auto_enabled:
                        bag_next_at_ms = _prime_bag_timer(now_ms=now_ms, cfg=cfg)
                    if auto_enabled and next_cast_at_ms is not None:
                        log_event(
                            LOGGER,
                            "session.resume_fishing",
                            resume_in_ms=max(0, next_cast_at_ms - now_ms),
                        )
                last_session_phase = phase

            if esc_trigger.poll_pressed_edge():
                auto_enabled = False
                next_cast_at_ms = None
                needs_precast_cleanup = False
                machine.reset()
                slot2_next_at_ms = None
                slot2_pending_after_first_reel = False
                slot2_pending_use_at_ms = None
                bag_next_at_ms = None
                run_started_at_ms = None
                log_event(LOGGER, "loop.paused")

            if key_trigger.poll_pressed_edge():
                if cfg.session.enabled and not session.active:
                    session.start(now_ms)
                    if cfg.humanize.enabled:
                        fatigue = FatigueModel(cfg.humanize)
                if not auto_enabled:
                    if run_started_at_ms is None:
                        run_started_at_ms = now_ms
                    auto_enabled = True
                    log_event(LOGGER, "loop.activated")
                cast_count += 1
                cast_id = machine.on_cast(now_ms)
                cast_anchor_x, cast_anchor_y = mouse.get_position()
                needs_precast_cleanup = False
                next_cast_at_ms = None
                slot2_next_at_ms = None
                slot2_pending_after_first_reel = True
                slot2_pending_use_at_ms = None
                bag_next_at_ms = _prime_bag_timer(now_ms=now_ms, cfg=cfg)
                log_event(
                    LOGGER,
                    "cast.manual_started",
                    cast_id=cast_id,
                    anchor_x=cast_anchor_x,
                    anchor_y=cast_anchor_y,
                    locate_delay_ms=cfg.timing.key_detect_delay_ms,
                    cast_count=cast_count,
                )

            if cfg.session.enabled and session.session_started and not session.active:
                time.sleep(frame_interval_s)
                continue

            if auto_enabled and slot2_pending_use_at_ms is not None and now_ms >= slot2_pending_use_at_ms:
                mouse.press_key_2()
                slot2_triggered_at_ms = int(time.monotonic() * 1000)
                slot2_next_at_ms, next_cast_at_ms = _prime_slot2_timer(
                    now_ms=slot2_triggered_at_ms,
                    cfg=cfg,
                    fatigue=fatigue,
                )
                slot2_pending_use_at_ms = None
                log_event(
                    LOGGER,
                    "slot2.used_after_first_reel",
                    next_auto_use_in_ms=max(0, slot2_next_at_ms - slot2_triggered_at_ms),
                    resume_cast_in_ms=cfg.timing.slot2_post_use_wait_ms,
                )
                continue

            if auto_enabled and next_cast_at_ms is not None and now_ms >= next_cast_at_ms:
                if slot2_next_at_ms is not None and now_ms >= slot2_next_at_ms:
                    mouse.press_key_2()
                    slot2_triggered_at_ms = int(time.monotonic() * 1000)
                    slot2_next_at_ms, next_cast_at_ms = _prime_slot2_timer(
                        now_ms=slot2_triggered_at_ms,
                        cfg=cfg,
                        fatigue=fatigue,
                    )
                    log_event(
                        LOGGER,
                        "slot2.auto_used",
                        next_auto_use_in_ms=max(0, slot2_next_at_ms - slot2_triggered_at_ms),
                        resume_cast_in_ms=cfg.timing.slot2_post_use_wait_ms,
                    )
                    continue
                if bag_next_at_ms is not None and now_ms >= bag_next_at_ms:
                    if _bag_timer_conflicts_with_slot2(
                        now_ms=now_ms,
                        cfg=cfg,
                        slot2_next_at_ms=slot2_next_at_ms,
                        slot2_pending_use_at_ms=slot2_pending_use_at_ms,
                    ):
                        bag_next_at_ms = _prime_bag_timer(now_ms=now_ms, cfg=cfg)
                        log_event(
                            LOGGER,
                            "bag.auto_deferred_for_slot2",
                            next_bag_in_ms=max(0, bag_next_at_ms - now_ms),
                        )
                        continue
                    bag_open_ms = _roll_bag_open_duration_ms(cfg)
                    mouse.press_bag_key()
                    time.sleep(bag_open_ms / 1000.0)
                    mouse.press_bag_key()
                    bag_closed_at_ms = int(time.monotonic() * 1000)
                    bag_next_at_ms = _prime_bag_timer(now_ms=bag_closed_at_ms, cfg=cfg)
                    next_cast_at_ms = bag_closed_at_ms
                    log_event(
                        LOGGER,
                        "bag.auto_toggled",
                        open_duration_ms=bag_open_ms,
                        next_bag_in_ms=max(0, bag_next_at_ms - bag_closed_at_ms),
                    )
                    continue
                if cfg.vision.enable_precast_cleanup and needs_precast_cleanup:
                    cleaned, cleaned_x, cleaned_y = _clear_lingering_bobber_before_cast(
                        vision=vision,
                        capture=capture,
                        mouse=mouse,
                        anchor_x=cast_anchor_x,
                        anchor_y=cast_anchor_y,
                        radius=cfg.vision.precast_cleanup_radius,
                        min_conf=cfg.vision.precast_cleanup_min_conf,
                    )
                    needs_precast_cleanup = False
                    if cleaned:
                        cast_anchor_x, cast_anchor_y = cleaned_x, cleaned_y
                        schedule_next_cast(
                            now_ms=now_ms,
                            reason="precast_cleanup",
                            extra_ms=cfg.timing.precast_cleanup_delay_ms,
                            cast_id=machine.cast_id,
                        )
                        continue
                if (
                    not cfg.noise.enabled
                    and cfg.timing.anti_afk_jump_every_casts > 0
                    and cast_count > 0
                    and (cast_count % cfg.timing.anti_afk_jump_every_casts) == 0
                    and last_anti_afk_cast_count != cast_count
                ):
                    mouse.press_space()
                    last_anti_afk_cast_count = cast_count
                    schedule_next_cast(
                        now_ms=now_ms,
                        reason="anti_afk_jump",
                        extra_ms=cfg.timing.anti_afk_jump_wait_ms,
                        cast_id=machine.cast_id,
                    )
                    log_event(
                        LOGGER,
                        "anti_afk.jump",
                        cast_count=cast_count,
                        resume_in_ms=cfg.timing.anti_afk_jump_wait_ms,
                    )
                    continue
                mouse.press_key_1()
                cast_count += 1
                cast_id = machine.on_cast(now_ms)
                cast_anchor_x, cast_anchor_y = mouse.get_position()
                next_cast_at_ms = None
                needs_precast_cleanup = False
                log_event(
                    LOGGER,
                    "cast.auto_started",
                    cast_id=cast_id,
                    anchor_x=cast_anchor_x,
                    anchor_y=cast_anchor_y,
                    locate_delay_ms=cfg.timing.key_detect_delay_ms,
                    cast_count=cast_count,
                )

            if machine.should_attempt_locate(now_ms):
                locate_attempt = machine.locate_attempt
                det, hit_count = _locate_stable_near_anchor(
                    vision=vision,
                    capture=capture,
                    anchor_x=cast_anchor_x,
                    anchor_y=cast_anchor_y,
                    radius=cfg.vision.key_search_radius,
                )

                accepted = (
                    det is not None
                    and (det.source != "fallback" or cfg.vision.allow_fallback_for_action)
                )
                if accepted and det is not None:
                    moved_x, moved_y = mouse.move_to(det.x, det.y)
                    if _is_move_close_enough(
                        actual_x=moved_x,
                        actual_y=moved_y,
                        target_x=det.x,
                        target_y=det.y,
                        jitter_px=cfg.control.jitter_px,
                    ):
                        cast_anchor_x, cast_anchor_y = moved_x, moved_y
                        machine.on_locate_success()
                        log_event(
                            LOGGER,
                            "locate.success",
                            cast_id=machine.cast_id,
                            target_x=det.x,
                            target_y=det.y,
                            actual_x=moved_x,
                            actual_y=moved_y,
                            conf=round(det.conf, 4),
                            source=det.source,
                            attempt=locate_attempt,
                            hit_count=hit_count,
                        )
                        time.sleep(frame_interval_s)
                        continue
                    log_event(
                        LOGGER,
                        "locate.move_verify_failed",
                        cast_id=machine.cast_id,
                        target_x=det.x,
                        target_y=det.y,
                        actual_x=moved_x,
                        actual_y=moved_y,
                        attempt=locate_attempt,
                    )
                decision = machine.on_locate_failure(now_ms)
                if decision.reason == "locate_retry":
                    log_event(
                        LOGGER,
                        "locate.retry",
                        cast_id=decision.cast_id,
                        next_attempt=machine.locate_attempt,
                        max_attempts=cfg.timing.key_retry_max_attempts,
                        delay_ms=cfg.timing.key_retry_interval_ms,
                        hit_count=hit_count,
                    )
                else:
                    log_event(
                        LOGGER,
                        "locate.failed",
                        cast_id=decision.cast_id,
                        max_attempts=cfg.timing.key_retry_max_attempts,
                        hit_count=hit_count,
                    )
                    needs_precast_cleanup = False
                    if decision.should_recast and auto_enabled:
                        schedule_next_cast(
                            now_ms=now_ms,
                            reason="miss_recast",
                            extra_ms=cfg.timing.recast_miss_delay_ms,
                            cast_id=decision.cast_id,
                        )

            audio_frame = audio_source.read_frame()
            audio_event = detector.update(audio_frame, now_ms=now_ms)
            decision = machine.update(now_ms=now_ms, audio_event=audio_event)
            if decision.should_recast:
                log_event(
                    LOGGER,
                    "cast.timeout",
                    cast_id=decision.cast_id,
                    max_cast_lifetime_ms=cfg.timing.max_cast_lifetime_ms,
                )
                needs_precast_cleanup = False
                if auto_enabled:
                    schedule_next_cast(
                        now_ms=now_ms,
                        reason=decision.reason,
                        extra_ms=cfg.timing.recast_miss_delay_ms,
                        cast_id=decision.cast_id,
                    )
                time.sleep(frame_interval_s)
                continue
            if decision.should_reel and audio_event is not None:
                low = min(cfg.control.click_delay_min_ms, cfg.control.click_delay_max_ms)
                high = max(cfg.control.click_delay_min_ms, cfg.control.click_delay_max_ms)
                if cfg.humanize.enabled and fatigue is not None:
                    delay_ms = fatigue.adjusted_sample(HumanDelay.bite_reaction(cfg.humanize))
                else:
                    delay_ms = random.randint(max(0, low), max(0, high))
                time.sleep(delay_ms / 1000.0)
                log_event(
                    LOGGER,
                    "bite.audio_detected",
                    cast_id=decision.cast_id,
                    audio_ts_ms=audio_event.ts_ms,
                    rms=round(audio_event.energy, 4),
                    threshold=round(audio_event.threshold, 4),
                    delay_ms=delay_ms,
                )
                action_name = _perform_bite_action(mouse, cfg.control)
                action_at_ms = int(time.monotonic() * 1000)
                machine.on_reel(action_at_ms)
                log_event(
                    LOGGER,
                    "bite.action",
                    cast_id=decision.cast_id,
                    action=action_name,
                )
                needs_precast_cleanup = cfg.vision.enable_precast_cleanup
                if auto_enabled:
                    if slot2_pending_after_first_reel:
                        slot2_pending_after_first_reel = False
                        slot2_pending_use_at_ms = _prime_slot2_after_reel(
                            now_ms=action_at_ms,
                            cfg=cfg,
                        )
                        log_event(
                            LOGGER,
                            "slot2.queued_after_first_reel",
                            cast_id=decision.cast_id,
                            execute_in_ms=max(0, slot2_pending_use_at_ms - action_at_ms),
                        )
                    else:
                        if bag_next_at_ms is None:
                            bag_next_at_ms = _prime_bag_timer(now_ms=action_at_ms, cfg=cfg)
                        if cfg.humanize.enabled and fatigue is not None:
                            extra_after = fatigue.adjusted_sample(
                                HumanDelay.cast_interval(cfg.humanize)
                            )
                        else:
                            extra_after = cfg.timing.auto_cast_base_ms + random.randint(
                                0, max(0, cfg.timing.auto_cast_jitter_max_ms)
                            )
                        noise_ms = 0
                        if cfg.noise.enabled:
                            noise_ctx = NoiseContext(
                                casts_since_last_noise=casts_since_last_noise,
                                session_elapsed_ms=max(0, now_ms - run_started_ms),
                                last_noise_ms=last_noise_ms,
                            )
                            noise_ms = noise_injector.maybe_inject(now_ms, noise_ctx)
                            if noise_ms > 0:
                                last_noise_ms = int(time.monotonic() * 1000)
                                casts_since_last_noise = 0
                                log_event(
                                    LOGGER,
                                    "noise.injected",
                                    duration_ms=noise_ms,
                                )
                            else:
                                casts_since_last_noise += 1
                        schedule_next_cast(
                            now_ms=action_at_ms,
                            reason="after_reel",
                            extra_ms=extra_after + noise_ms,
                            cast_id=decision.cast_id,
                        )
            time.sleep(frame_interval_s)
    finally:
        audio_source.close()


def command_mouse_test(cfg: AppConfig, seconds: int) -> None:
    print(f"starting mouse test for {seconds}s")
    print("Mouse will move to detected target; no click.")

    detector = SplashDetector(cfg.audio)
    vision = BobberDetector(cfg.vision)
    vision.load()
    capture = ScreenCapture()
    audio_source = WasapiLoopbackSource(cfg.audio)
    mouse = MouseController(
        cfg.control,
        input_cfg=cfg.input,
        mouse_path_cfg=cfg.mouse_path,
        humanize_cfg=cfg.humanize,
    )

    end_ts = time.monotonic() + max(1, seconds)
    frame_interval_s = cfg.audio.frame_ms / 1000.0
    last_bite_ms = -10_000_000

    try:
        while time.monotonic() < end_ts:
            now_ms = int(time.monotonic() * 1000)
            audio_frame = audio_source.read_frame()
            audio_event = detector.update(audio_frame, now_ms=now_ms)
            if audio_event is None:
                time.sleep(frame_interval_s)
                continue
            if (audio_event.ts_ms - last_bite_ms) < cfg.audio.bite_lock_ms:
                time.sleep(frame_interval_s)
                continue

            shot = capture.grab_with_offset()
            det = vision.detect(shot.frame_bgr)
            if det is None:
                print("[mouse-test] bite detected but no visual target")
                last_bite_ms = audio_event.ts_ms
                time.sleep(frame_interval_s)
                continue
            if det.source == "fallback" and not cfg.vision.allow_fallback_for_action:
                print(
                    "[mouse-test] ignored fallback detection "
                    f"(best_template_score={vision.last_template_score:.3f})"
                )
                last_bite_ms = audio_event.ts_ms
                time.sleep(frame_interval_s)
                continue

            low = min(cfg.control.click_delay_min_ms, cfg.control.click_delay_max_ms)
            high = max(cfg.control.click_delay_min_ms, cfg.control.click_delay_max_ms)
            delay_ms = random.randint(max(0, low), max(0, high))
            time.sleep(delay_ms / 1000.0)
            abs_x = det.x + shot.left
            abs_y = det.y + shot.top
            moved_x, moved_y = mouse.move_to(abs_x, abs_y)
            if _is_move_close_enough(
                actual_x=moved_x,
                actual_y=moved_y,
                target_x=abs_x,
                target_y=abs_y,
                jitter_px=cfg.control.jitter_px,
            ):
                print(
                    f"[mouse-test] moved target=({abs_x}, {abs_y}) actual=({moved_x}, {moved_y}) "
                    f"after {delay_ms}ms conf={det.conf:.3f} source={det.source}"
                )
            else:
                print(
                    f"[mouse-test] move verify failed target=({abs_x}, {abs_y}) "
                    f"actual=({moved_x}, {moved_y}) after {delay_ms}ms"
                )
            last_bite_ms = audio_event.ts_ms
            time.sleep(frame_interval_s)
    finally:
        audio_source.close()
    print("mouse test complete")


def command_listen_test(cfg: AppConfig, seconds: int) -> None:
    print(f"starting listen test for {seconds}s")
    print("No mouse input will be performed in this mode.")

    detector = SplashDetector(cfg.audio)
    vision = BobberDetector(cfg.vision)
    vision.load()
    capture = ScreenCapture()
    audio_source = WasapiLoopbackSource(cfg.audio)

    end_ts = time.monotonic() + max(1, seconds)
    frame_interval_s = cfg.audio.frame_ms / 1000.0
    last_bite_ms = -10_000_000

    try:
        while time.monotonic() < end_ts:
            now_ms = int(time.monotonic() * 1000)
            audio_frame = audio_source.read_frame()
            audio_event = detector.update(audio_frame, now_ms=now_ms)
            if audio_event is None:
                time.sleep(frame_interval_s)
                continue
            if (audio_event.ts_ms - last_bite_ms) < cfg.audio.bite_lock_ms:
                time.sleep(frame_interval_s)
                continue

            print(
                f"[audio] ts={audio_event.ts_ms}ms rms={audio_event.energy:.4f} "
                f"th={audio_event.threshold:.4f}"
            )
            shot = capture.grab_with_offset()
            det = vision.detect(shot.frame_bgr)
            if det is None:
                print("[vision] no target")
            else:
                if det.source == "fallback" and not cfg.vision.allow_fallback_for_action:
                    print(
                        "[vision] fallback ignored "
                        f"(best_template_score={vision.last_template_score:.3f})"
                    )
                    last_bite_ms = audio_event.ts_ms
                    time.sleep(frame_interval_s)
                    continue
                abs_x = det.x + shot.left
                abs_y = det.y + shot.top
                print(
                    f"[vision] x={abs_x} y={abs_y} conf={det.conf:.3f} source={det.source}"
                )
            last_bite_ms = audio_event.ts_ms
            time.sleep(frame_interval_s)
    finally:
        audio_source.close()
    print("listen test complete")


def command_audio_diagnose(cfg: AppConfig, seconds: int) -> None:
    speakers = list_loopback_speakers()
    if speakers:
        print("loopback speakers:")
        for name in speakers:
            print(f"  - {name}")
    print("input devices:")
    devices = list_input_devices()
    if not devices:
        print("  no input devices found")
    for dev in devices:
        print(
            f"  [{dev['index']}] {dev['name']} "
            f"(ch={dev['max_input_channels']} sr={dev['default_samplerate']})"
        )
    print(
        "tip: set audio.input_device in config.yaml to one device index, "
        "then rerun listen-test"
    )

    source = WasapiLoopbackSource(cfg.audio)
    print(
        f"opened backend={source.selected_backend} device={source.selected_device} "
        f"samplerate={source.selected_sample_rate} endpoint={source.selected_endpoint}"
    )
    end_ts = time.monotonic() + max(1, seconds)
    frame_interval_s = cfg.audio.frame_ms / 1000.0
    last_print = 0.0
    try:
        while time.monotonic() < end_ts:
            frame = source.read_frame()
            rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
            peak = float(np.max(np.abs(frame))) if frame.size else 0.0
            now = time.monotonic()
            if now - last_print >= 0.8:
                print(f"[diag] rms={rms:.4f} peak={peak:.4f}")
                last_print = now
            time.sleep(frame_interval_s)
    finally:
        source.close()
    print("audio diagnose complete")


def command_audio_selftest(cfg: AppConfig) -> None:
    if sd is None:
        raise RuntimeError("sounddevice is required for audio-selftest tone playback")
    print("starting audio selftest (play tone + loopback capture)")
    source = WasapiLoopbackSource(cfg.audio)
    fs = 48_000
    dur = 1.0
    t = np.arange(int(fs * dur)) / fs
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tone_st = np.column_stack([tone, tone])
    sd.play(tone_st, fs, blocking=False)

    end_ts = time.monotonic() + 1.8
    rms_values: list[float] = []
    peak_values: list[float] = []
    try:
        while time.monotonic() < end_ts:
            frame = source.read_frame()
            rms_values.append(float(np.sqrt(np.mean(np.square(frame)) + 1e-12)))
            peak_values.append(float(np.max(np.abs(frame))) if frame.size else 0.0)
            time.sleep(cfg.audio.frame_ms / 1000.0)
    finally:
        source.close()
        sd.stop()

    rms_max = max(rms_values) if rms_values else 0.0
    peak_max = max(peak_values) if peak_values else 0.0
    print(
        f"selftest backend={source.selected_backend} endpoint={source.selected_endpoint} "
        f"rms_max={rms_max:.4f} peak_max={peak_max:.4f}"
    )
    if rms_max <= 0.001:
        print("selftest result: capture path failed or wrong output endpoint")
    else:
        print("selftest result: capture path OK")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WoW classic fishing helper bot")
    parser.add_argument("--config", default="config.yaml", help="path to YAML config")
    parser.add_argument(
        "--seconds",
        type=int,
        default=15,
        help="listen test duration in seconds",
    )
    parser.add_argument(
        "command",
        choices=[
            "download-model",
            "test-audio",
            "audio-diagnose",
            "audio-selftest",
            "listen-test",
            "mouse-test",
            "run",
        ],
        help="command to execute",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "download-model":
        command_download_model(cfg)
    elif args.command == "test-audio":
        command_test_audio(cfg)
    elif args.command == "audio-diagnose":
        command_audio_diagnose(cfg, args.seconds)
    elif args.command == "audio-selftest":
        command_audio_selftest(cfg)
    elif args.command == "listen-test":
        command_listen_test(cfg, args.seconds)
    elif args.command == "mouse-test":
        command_mouse_test(cfg, args.seconds)
    elif args.command == "run":
        command_run(cfg)


if __name__ == "__main__":
    main()
