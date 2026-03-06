from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT / "data" / "yolo" / "dataset.yaml"
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"


def _run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _latest_base_checkpoint() -> Path | None:
    candidates = sorted(
        (path for path in RUNS_DIR.glob("bobber_train*/weights/best.pt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    if not DATASET_YAML.exists():
        raise RuntimeError("dataset.yaml not found. Run scripts/prepare_yolo_dataset.py first.")

    run_name = _run_name("bobber_overfit")
    run_dir = RUNS_DIR / run_name
    base_best = _latest_base_checkpoint()
    start_model = str(base_best if base_best is not None else (ROOT / "yolov8n.pt"))
    model = YOLO(start_model)

    model.train(
        data=str(DATASET_YAML),
        epochs=100,
        imgsz=640,
        batch=4,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
        device=_device(),
        workers=0,
        patience=120,
        # Weak augmentations to allow mild overfit to scene-specific data.
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.002,
        hsv_s=0.10,
        hsv_v=0.10,
        degrees=0.0,
        translate=0.0,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
    )

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if not best_pt.exists():
        raise RuntimeError("best.pt not found after overfit training")
    if not last_pt.exists():
        raise RuntimeError("last.pt not found after overfit training")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_versioned = MODELS_DIR / f"{run_name}.best.pt"
    best_latest = MODELS_DIR / "bobber.best.pt"
    last_versioned = MODELS_DIR / f"{run_name}.last.pt"
    last_latest = MODELS_DIR / "bobber.last.pt"
    best_versioned.write_bytes(best_pt.read_bytes())
    best_latest.write_bytes(best_versioned.read_bytes())
    last_versioned.write_bytes(last_pt.read_bytes())
    last_latest.write_bytes(last_versioned.read_bytes())

    trained = YOLO(str(best_pt))
    onnx_path = trained.export(format="onnx", imgsz=640, opset=12, simplify=False)

    versioned = MODELS_DIR / f"{run_name}.onnx"
    latest = MODELS_DIR / "bobber.onnx"
    exported = Path(onnx_path)
    if versioned.exists():
        versioned.unlink()
    exported.replace(versioned)
    latest.write_bytes(versioned.read_bytes())
    print(f"base checkpoint: {start_model}")
    print(f"run dir: {run_dir}")
    print(f"checkpoint: {best_pt}")
    print(f"last checkpoint: {last_pt}")
    print(f"best model copy: {best_versioned}")
    print(f"last model copy: {last_versioned}")
    print(f"exported model: {versioned}")
    print(f"latest model: {latest}")


if __name__ == "__main__":
    main()
