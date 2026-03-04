from pathlib import Path

from app.config import VisionConfig
from app.vision import ModelManager


def test_model_manager_reuses_existing_file(tmp_path: Path) -> None:
    model = tmp_path / "bobber.onnx"
    model.write_bytes(b"onnx")
    cfg = VisionConfig(model_path=str(model), model_url="https://example.com/unused.onnx")
    manager = ModelManager(cfg)
    path = manager.ensure_model()
    assert path == model
    assert model.read_bytes() == b"onnx"

