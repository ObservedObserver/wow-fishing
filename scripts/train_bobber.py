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

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATASET_YAML),
        epochs=120,
        imgsz=640,
        batch=4,
        project=str(RUNS_DIR),
        name="bobber_train",
        exist_ok=True,
        device="cpu",
        workers=0,
        patience=30,
    )

    best_pt = RUNS_DIR / "bobber_train" / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError("best.pt not found after training")

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

