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
from app.capture import ScreenCapture
from app.config import AppConfig, load_config
from app.input_control import MouseController
from app.vision import BobberDetector, Detection, ModelManager


_VK_1 = 0x31
_VK_NUMPAD1 = 0x61
_VK_ESC = 0x1B
_LOCATE_CONFIRM_FRAMES = 1
_LOCATE_CONFIRM_INTERVAL_MS = 0


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

    abs_x = det.x + shot.left
    abs_y = det.y + shot.top
    moved_x, moved_y = mouse.move_and_right_click(abs_x, abs_y)
    print(
        f"[precast-cleanup] clicked lingering bobber at ({moved_x}, {moved_y}) "
        f"conf={det.conf:.3f} source={det.source}"
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
    mouse = MouseController(cfg.control)
    audio_source = WasapiLoopbackSource(cfg.audio)
    key_trigger = KeyOneTrigger()
    esc_trigger = EscTrigger()
    frame_interval_s = cfg.audio.frame_ms / 1000.0
    pending_locate_at_ms: int | None = None
    pending_locate_attempt = 0
    last_sound_click_ms = -10_000_000
    next_cast_at_ms: int | None = None
    cast_anchor_x: int | None = None
    cast_anchor_y: int | None = None
    bobber_tracked = False
    auto_enabled = False
    needs_precast_cleanup = False

    def schedule_next_cast(now_ms: int, reason: str, extra_ms: int) -> None:
        nonlocal next_cast_at_ms
        next_cast_at_ms = now_ms + max(0, extra_ms)
        print(f"[cast] scheduled in {max(0, extra_ms)}ms reason={reason}")

    try:
        while True:
            now_ms = int(time.monotonic() * 1000)

            if esc_trigger.poll_pressed_edge():
                auto_enabled = False
                next_cast_at_ms = None
                pending_locate_at_ms = None
                pending_locate_attempt = 0
                bobber_tracked = False
                needs_precast_cleanup = False
                print("[loop] paused by ESC")

            if key_trigger.poll_pressed_edge():
                if not auto_enabled:
                    auto_enabled = True
                    print("[loop] activated by key 1")
                pending_locate_at_ms = now_ms + cfg.timing.key_detect_delay_ms
                pending_locate_attempt = 1
                cast_anchor_x, cast_anchor_y = mouse.get_position()
                bobber_tracked = False
                needs_precast_cleanup = False
                next_cast_at_ms = None
                print(
                    f"[key] detected 1 at ({cast_anchor_x}, {cast_anchor_y}), "
                    f"schedule locate at +{cfg.timing.key_detect_delay_ms}ms"
                )

            if auto_enabled and next_cast_at_ms is not None and now_ms >= next_cast_at_ms:
                if needs_precast_cleanup:
                    cleaned, cleaned_x, cleaned_y = _clear_lingering_bobber_before_cast(
                        vision=vision,
                        capture=capture,
                        mouse=mouse,
                        anchor_x=cast_anchor_x,
                        anchor_y=cast_anchor_y,
                        radius=cfg.vision.key_search_radius,
                    )
                    needs_precast_cleanup = False
                    if cleaned:
                        cast_anchor_x, cast_anchor_y = cleaned_x, cleaned_y
                        schedule_next_cast(
                            now_ms=now_ms,
                            reason="precast_cleanup",
                            extra_ms=cfg.timing.precast_cleanup_delay_ms,
                        )
                        continue
                mouse.press_key_1()
                cast_anchor_x, cast_anchor_y = mouse.get_position()
                pending_locate_at_ms = now_ms + cfg.timing.key_detect_delay_ms
                pending_locate_attempt = 1
                next_cast_at_ms = None
                bobber_tracked = False
                needs_precast_cleanup = False
                print(
                    f"[cast] key1 triggered at ({cast_anchor_x}, {cast_anchor_y}), "
                    f"locate in {cfg.timing.key_detect_delay_ms}ms"
                )

            if pending_locate_at_ms is not None and now_ms >= pending_locate_at_ms:
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
                    cast_anchor_x, cast_anchor_y = moved_x, moved_y
                    print(
                        f"[key-move] moved to ({moved_x}, {moved_y}) "
                        f"conf={det.conf:.3f} source={det.source} attempt={pending_locate_attempt} "
                        f"hits={hit_count}/{_LOCATE_CONFIRM_FRAMES}"
                    )
                    pending_locate_at_ms = None
                    pending_locate_attempt = 0
                    bobber_tracked = True
                else:
                    if pending_locate_attempt < max(1, cfg.timing.key_retry_max_attempts):
                        pending_locate_attempt += 1
                        pending_locate_at_ms = now_ms + cfg.timing.key_retry_interval_ms
                        print(
                            f"[key-move] detect failed, retry {pending_locate_attempt}/"
                            f"{cfg.timing.key_retry_max_attempts} in "
                            f"{cfg.timing.key_retry_interval_ms}ms "
                            f"(hits={hit_count}/{_LOCATE_CONFIRM_FRAMES})"
                        )
                    else:
                        print(
                            "[key-move] detect failed after max retries "
                            f"(hits={hit_count}/{_LOCATE_CONFIRM_FRAMES})"
                        )
                        pending_locate_at_ms = None
                        pending_locate_attempt = 0
                        bobber_tracked = False
                        needs_precast_cleanup = False
                        if cfg.timing.recast_on_miss:
                            if auto_enabled:
                                schedule_next_cast(
                                    now_ms=now_ms,
                                    reason="miss_recast",
                                    extra_ms=cfg.timing.recast_miss_delay_ms,
                                )

            audio_frame = audio_source.read_frame()
            audio_event = detector.update(audio_frame, now_ms=now_ms)
            if (
                bobber_tracked
                and audio_event is not None
                and (now_ms - last_sound_click_ms) >= cfg.audio.bite_lock_ms
            ):
                low = min(cfg.control.click_delay_min_ms, cfg.control.click_delay_max_ms)
                high = max(cfg.control.click_delay_min_ms, cfg.control.click_delay_max_ms)
                delay_ms = random.randint(max(0, low), max(0, high))
                time.sleep(delay_ms / 1000.0)
                print(
                    f"[audio-click] ts={audio_event.ts_ms} rms={audio_event.energy:.4f} "
                    f"th={audio_event.threshold:.4f} delay={delay_ms}ms"
                )
                mouse.right_click()
                last_sound_click_ms = int(time.monotonic() * 1000)
                bobber_tracked = False
                needs_precast_cleanup = True
                if auto_enabled:
                    schedule_next_cast(
                        now_ms=last_sound_click_ms,
                        reason="after_reel",
                        extra_ms=cfg.timing.auto_cast_base_ms
                        + random.randint(0, max(0, cfg.timing.auto_cast_jitter_max_ms)),
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
    mouse = MouseController(cfg.control)

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
            print(
                f"[mouse-test] moved to ({moved_x}, {moved_y}) "
                f"after {delay_ms}ms conf={det.conf:.3f} source={det.source}"
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
