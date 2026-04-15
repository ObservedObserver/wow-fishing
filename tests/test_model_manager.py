from pathlib import Path

from app.config import VisionConfig
from app import vision
from app.vision import ModelManager, _resolve_onnx_providers


def test_model_manager_reuses_existing_file(tmp_path: Path) -> None:
    model = tmp_path / "bobber.onnx"
    model.write_bytes(b"onnx")
    cfg = VisionConfig(
        model_path=str(model),
        model_url="https://example.com/unused.onnx",
        model_sha256="87e93f89f2be0db364e8be052f79f389e6c2da239831922e24513288af522a43",
    )
    manager = ModelManager(cfg)
    path = manager.ensure_model()
    assert path == model
    assert model.read_bytes() == b"onnx"


def test_model_manager_rejects_checksum_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "bobber.onnx"
    model.write_bytes(b"wrong")
    cfg = VisionConfig(model_path=str(model), model_sha256="abc123")
    manager = ModelManager(cfg)

    try:
        manager.ensure_model()
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum validation to fail")


def test_model_manager_requires_trusted_source_when_model_missing(tmp_path: Path) -> None:
    model = tmp_path / "missing.onnx"
    cfg = VisionConfig(model_path=str(model), model_url=None, model_sha256=None)
    manager = ModelManager(cfg)

    try:
        manager.ensure_model()
    except FileNotFoundError as exc:
        assert "controlled model artifact missing" in str(exc)
    else:
        raise AssertionError("expected missing controlled artifact to fail")


def test_model_manager_downloads_via_temp_file_and_replaces_target(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "bobber.onnx"
    payload = b"fresh-model"

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            return payload

    monkeypatch.setattr(vision, "urlopen", lambda *args, **kwargs: _FakeResponse())
    cfg = VisionConfig(
        model_path=str(model),
        model_url="https://example.com/model.onnx",
        model_sha256="607e57cbd807572da816841060834e6b58299c694e6937d6193a260096a3bb8e",
    )
    manager = ModelManager(cfg)

    path = manager.ensure_model()

    assert path == model
    assert model.read_bytes() == payload
    assert list(tmp_path.glob("*.tmp")) == []


def test_model_manager_removes_temp_file_after_failed_validation(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "bobber.onnx"
    payload = b"bad-model"

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            return payload

    monkeypatch.setattr(vision, "urlopen", lambda *args, **kwargs: _FakeResponse())
    cfg = VisionConfig(
        model_path=str(model),
        model_url="https://example.com/model.onnx",
        model_sha256="abc123",
    )
    manager = ModelManager(cfg)

    try:
        manager.ensure_model()
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum validation to fail")

    assert not model.exists()
    assert list(tmp_path.iterdir()) == []


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
