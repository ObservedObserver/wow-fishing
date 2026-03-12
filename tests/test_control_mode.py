from app.config import ControlConfig
from main import _normalize_bite_action_mode, _perform_bite_action


class _FakeMouse:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def right_click(self) -> None:
        self.actions.append("right_click")

    def press_interaction_key(self) -> None:
        self.actions.append("interaction_key")


def test_normalize_bite_action_mode_accepts_aliases() -> None:
    assert _normalize_bite_action_mode("mouse") == "mouse"
    assert _normalize_bite_action_mode("right-click") == "mouse"
    assert _normalize_bite_action_mode("interaction_key") == "interact_hotkey"


def test_perform_bite_action_uses_mouse_right_click() -> None:
    mouse = _FakeMouse()
    action = _perform_bite_action(
        mouse=mouse,  # type: ignore[arg-type]
        cfg=ControlConfig(bite_action_mode="mouse"),
    )

    assert action == "mouse_right_click"
    assert mouse.actions == ["right_click"]


def test_perform_bite_action_uses_interaction_hotkey() -> None:
    mouse = _FakeMouse()
    action = _perform_bite_action(
        mouse=mouse,  # type: ignore[arg-type]
        cfg=ControlConfig(bite_action_mode="interact_hotkey", interaction_key="F12"),
    )

    assert action == "interaction_hotkey:F12"
    assert mouse.actions == ["interaction_key"]
