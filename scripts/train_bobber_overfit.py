from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT / "data" / "yolo" / "dataset.yaml"
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"


def main() -> None:
    if not DATASET_YAML.exists():
        raise RuntimeError("dataset.yaml not found. Run scripts/prepare_yolo_dataset.py first.")

    base_best = RUNS_DIR / "bobber_train" / "weights" / "best.pt"
    start_model = str(base_best if base_best.exists() else "yolov8n.pt")
    model = YOLO(start_model)

    model.train(
        data=str(DATASET_YAML),
        epochs=100,
        imgsz=640,
        batch=4,
        project=str(RUNS_DIR),
        name="bobber_overfit",
        exist_ok=True,
        device="cpu",
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

    best_pt = RUNS_DIR / "bobber_overfit" / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError("best.pt not found after overfit training")

    trained = YOLO(str(best_pt))
    onnx_path = trained.export(format="onnx", imgsz=640, opset=12, simplify=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "bobber.onnx"
    if target.exists():
        target.unlink()
    Path(onnx_path).replace(target)
    print(f"exported model: {target}")


if __name__ == "__main__":
    main()

