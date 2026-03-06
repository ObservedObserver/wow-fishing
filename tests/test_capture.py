from __future__ import annotations

from app.capture import _contains_point, _select_monitor


def test_select_monitor_prefers_monitor_that_contains_anchor() -> None:
    monitors = [
        {"left": 0, "top": 0, "width": 3200, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": 0, "width": 1280, "height": 1024},
    ]

    selected = _select_monitor(monitors, preferred_x=2200, preferred_y=500)

    assert selected is monitors[2]


def test_select_monitor_falls_back_to_virtual_desktop() -> None:
    monitors = [
        {"left": -1920, "top": 0, "width": 3840, "height": 1080},
        {"left": -1920, "top": 0, "width": 1920, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
    ]

    selected = _select_monitor(monitors, preferred_x=5000, preferred_y=5000)

    assert selected is monitors[0]


def test_contains_point_uses_monitor_bounds() -> None:
    monitor = {"left": 100, "top": 200, "width": 300, "height": 400}

    assert _contains_point(monitor, 100, 200)
    assert _contains_point(monitor, 399, 599)
    assert not _contains_point(monitor, 400, 599)
    assert not _contains_point(monitor, 399, 600)
