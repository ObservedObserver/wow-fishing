from __future__ import annotations

import sys

import pytest

from app.input_backend import InputBackend, MouseButton, create_backend


def test_create_backend_non_windows_is_stub() -> None:
    if sys.platform == "win32":
        pytest.skip("non-Windows stub behavior")
    b = create_backend("send_input")
    assert isinstance(b, InputBackend)
    b.move_cursor(0, 0)
    b.mouse_down(MouseButton.LEFT)
    b.mouse_up(MouseButton.LEFT)
