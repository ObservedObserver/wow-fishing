from __future__ import annotations

from app.behavior_noise import BehaviorNoiseInjector, NoiseContext
from app.config import NoiseConfig
from app.input_backend import InputBackend, MouseButton, create_backend


def test_noise_respects_cooldown() -> None:
    cfg = NoiseConfig(enabled=True, cooldown_min_ms=100_000)
    inj = BehaviorNoiseInjector(cfg, create_backend("send_input"))
    ctx = NoiseContext(
        casts_since_last_noise=100,
        session_elapsed_ms=0,
        last_noise_ms=0,
    )
    assert inj.maybe_inject(50_000, ctx) == 0


class _SpyBackend(InputBackend):
    def __init__(self) -> None:
        self.keys: list[int] = []

    def move_cursor(self, x: int, y: int) -> None:
        return None

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        return None

    def mouse_down(self, button: MouseButton) -> None:
        return None

    def mouse_up(self, button: MouseButton) -> None:
        return None

    def key_down(self, vk_code: int) -> None:
        self.keys.append(vk_code)

    def key_up(self, vk_code: int) -> None:
        return None


def test_noise_toggle_bag_hits_key() -> None:
    cfg = NoiseConfig(
        enabled=True,
        cooldown_min_ms=0,
        base_probability=1.0,
        weights={"toggle_bag": 1},
    )
    spy = _SpyBackend()
    inj = BehaviorNoiseInjector(cfg, spy, rng=__import__("random").Random(0))
    ctx = NoiseContext(1, 0, -1)
    inj.maybe_inject(1_000_000, ctx)
    assert 0x42 in spy.keys
