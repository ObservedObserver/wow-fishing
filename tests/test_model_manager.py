from pathlib import Path

from app.config import VisionConfig
from app import vision
from app.vision import ModelManager, _resolve_onnx_providers


def test_model_manager_reuses_existing_file(tmp_path: Path) -> None:
    model = tmp_path / "bobber.onnx"
    model.write_bytes(b"onnx")
    cfg = VisionConfig(model_path=str(model), model_url="https://example.com/unused.onnx")
    manager = ModelManager(cfg)
    path = manager.ensure_model()
    assert path == model
    assert model.read_bytes() == b"onnx"


def test_resolve_onnx_providers_prefers_cuda_when_available(monkeypatch) -> None:
    class _FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(vision, "ort", _FakeOrt)
    cfg = VisionConfig()

    assert _resolve_onnx_providers(cfg) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_onnx_providers_respects_configured_order(monkeypatch) -> None:
    class _FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider", "CUDAExecutionProvider"]

    monkeypatch.setattr(vision, "ort", _FakeOrt)
    cfg = VisionConfig(onnx_providers=("CPUExecutionProvider", "CUDAExecutionProvider"))

    assert _resolve_onnx_providers(cfg) == ["CPUExecutionProvider", "CUDAExecutionProvider"]
