from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from app.config import NoiseConfig
from app.input_backend import InputBackend, MouseButton


_VK_B = 0x42
_VK_W = 0x57
_VK_A = 0x41
_VK_S = 0x53
_VK_D = 0x44
_VK_SPACE = 0x20


class NoiseAction(Enum):
    CAMERA_PAN = auto()
    TOGGLE_BAG = auto()
    SHORT_WALK = auto()
    JUMP = auto()
    IDLE_PAUSE = auto()
    LOOK_AROUND = auto()


@dataclass(slots=True)
class NoiseContext:
    casts_since_last_noise: int
    session_elapsed_ms: int
    last_noise_ms: int


class BehaviorNoiseInjector:
    """Optional non-fishing actions between casts; reduces repetitive action chains."""

    def __init__(
        self,
        cfg: NoiseConfig,
        backend: InputBackend,
        rng: random.Random | None = None,
    ) -> None:
        self._cfg = cfg
        self._backend = backend
        self._rng = rng or random
        self._weights = self._normalize_weights(cfg.weights)

    def _normalize_weights(self, raw: dict[str, int]) -> dict[NoiseAction, float]:
        mapping = {
            "camera_pan": NoiseAction.CAMERA_PAN,
            "toggle_bag": NoiseAction.TOGGLE_BAG,
            "short_walk": NoiseAction.SHORT_WALK,
            "jump": NoiseAction.JUMP,
            "idle_pause": NoiseAction.IDLE_PAUSE,
            "look_around": NoiseAction.LOOK_AROUND,
        }
        out: dict[NoiseAction, float] = {}
        for key, act in mapping.items():
            w = float(raw.get(key, 0))
            if w > 0:
                out[act] = w
        if not out:
            out = {NoiseAction.IDLE_PAUSE: 1.0}
        return out

    def _probability(self, ctx: NoiseContext) -> float:
        base = max(0.0, min(1.0, self._cfg.base_probability))
        extra = max(0, ctx.casts_since_last_noise) * max(0.0, self._cfg.casts_scale_per_cast)
        cap = max(base, self._cfg.max_probability)
        return min(cap, base + extra)

    def maybe_inject(self, now_ms: int, ctx: NoiseContext) -> int:
        if not self._cfg.enabled or not self._weights:
            return 0
        if ctx.last_noise_ms >= 0 and (now_ms - ctx.last_noise_ms) < self._cfg.cooldown_min_ms:
            return 0
        p = self._probability(ctx)
        if self._rng.random() > p:
            return 0
        actions = list(self._weights.keys())
        weights = [self._weights[a] for a in actions]
        action = self._rng.choices(actions, weights=weights, k=1)[0]
        return self._execute(action)

    def _execute(self, action: NoiseAction) -> int:
        start = time.monotonic()
        if action == NoiseAction.IDLE_PAUSE:
            time.sleep(self._rng.uniform(2.0, 8.0))
        elif action == NoiseAction.JUMP:
            self._tap_key(_VK_SPACE)
        elif action == NoiseAction.TOGGLE_BAG:
            self._tap_key(_VK_B)
            time.sleep(self._rng.uniform(0.12, 0.28))
            self._tap_key(_VK_B)
        elif action == NoiseAction.SHORT_WALK:
            key = self._rng.choice([_VK_W, _VK_A, _VK_S, _VK_D])
            dur = self._rng.uniform(0.25, 1.2)
            self._backend.key_down(key)
            time.sleep(dur)
            self._backend.key_up(key)
        elif action == NoiseAction.CAMERA_PAN:
            self._backend.mouse_down(MouseButton.MIDDLE)
            for _ in range(self._rng.randint(8, 18)):
                dx = self._rng.randint(-14, 14)
                dy = self._rng.randint(-10, 10)
                self._backend.mouse_move_relative(dx, dy)
                time.sleep(self._rng.uniform(0.006, 0.014))
            self._backend.mouse_up(MouseButton.MIDDLE)
        elif action == NoiseAction.LOOK_AROUND:
            self._backend.mouse_down(MouseButton.RIGHT)
            for _ in range(self._rng.randint(10, 22)):
                dx = self._rng.randint(-18, 18)
                dy = self._rng.randint(-8, 8)
                self._backend.mouse_move_relative(dx, dy)
                time.sleep(self._rng.uniform(0.007, 0.015))
            self._backend.mouse_up(MouseButton.RIGHT)
        return int((time.monotonic() - start) * 1000)

    def _tap_key(self, vk: int) -> None:
        self._backend.key_down(vk)
        time.sleep(self._rng.uniform(0.04, 0.12))
        self._backend.key_up(vk)
