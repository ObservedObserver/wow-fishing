from __future__ import annotations

import base64
import json
import random
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_POS = ROOT / "data" / "raw" / "positive"
RAW_NEG = ROOT / "data" / "raw" / "negative"
EXTERNAL_ROOT = ROOT / "data" / "external"
OUT_IMAGES = ROOT / "data" / "yolo" / "images"
OUT_LABELS = ROOT / "data" / "yolo" / "labels"
DATASET_YAML = ROOT / "data" / "yolo" / "dataset.yaml"

CLASS_NAME = "bobber"
CLASS_ID = 0
TRAIN_RATIO = 0.8
SEED = 42


def ensure_dirs() -> None:
    yolo_root = DATASET_YAML.parent
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    for split in ("train", "val"):
        (OUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
        (OUT_LABELS / split).mkdir(parents=True, exist_ok=True)


def _shape_to_bbox(shape: dict[str, Any]) -> tuple[float, float, float, float] | None:
    points = shape.get("points") or []
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_to_yolo(
    bbox: tuple[float, float, float, float], image_w: int, image_h: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / image_w
    cy = ((y1 + y2) / 2.0) / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h
    return cx, cy, w, h


def _write_label_file(path: Path, label_lines: list[str]) -> None:
    path.write_text("\n".join(label_lines), encoding="utf-8")


def collect_positive_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_file in sorted(RAW_POS.glob("*.json")):
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        image_data = payload.get("imageData")
        image_path = payload.get("imagePath")
        if not image_data or not image_path:
            continue

        image_bytes = base64.b64decode(image_data)
        image_h = int(payload.get("imageHeight", 0))
        image_w = int(payload.get("imageWidth", 0))
        if image_h <= 0 or image_w <= 0:
            continue

        label_lines: list[str] = []
        for shape in payload.get("shapes", []):
            if str(shape.get("label", "")).lower() != CLASS_NAME:
                continue
            bbox = _shape_to_bbox(shape)
            if bbox is None:
                continue
            cx, cy, w, h = _bbox_to_yolo(bbox, image_w=image_w, image_h=image_h)
            label_lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        out_name = Path(image_path).with_suffix(".png").name
        records.append(
            {
                "name": out_name,
                "image_bytes": image_bytes,
                "label_lines": label_lines,
            }
        )
    return records


def collect_negative_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not RAW_NEG.exists():
        return records

    for img in sorted(RAW_NEG.glob("*")):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        records.append(
            {
                "name": img.with_suffix(".png").name,
                "image_bytes": img.read_bytes(),
                "label_lines": [],
            }
        )
    return records


def _normalize_external_label_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    label_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        try:
            _ = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        label_lines.append(f"{CLASS_ID} {' '.join(parts[1:])}")
    return label_lines


def collect_external_yolo_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not EXTERNAL_ROOT.exists():
        return records

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for images_dir in sorted(EXTERNAL_ROOT.glob("**/train/images")) + sorted(
        EXTERNAL_ROOT.glob("**/val/images")
    ):
        labels_dir = images_dir.parent / "labels"
        source_name = images_dir.parents[2].name
        split_name = images_dir.parent.name
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in image_exts or not image_path.is_file():
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            out_name = f"{source_name}__{split_name}__{image_path.name}"
            records.append(
                {
                    "name": out_name,
                    "image_bytes": image_path.read_bytes(),
                    "label_lines": _normalize_external_label_file(label_path),
                }
            )
    return records


def write_dataset_yaml() -> None:
    content = "\n".join(
        [
            f"path: {ROOT.as_posix()}/data/yolo",
            "train: images/train",
            "val: images/val",
            "names:",
            f"  0: {CLASS_NAME}",
            "",
        ]
    )
    DATASET_YAML.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    positives = collect_positive_records()
    negatives = collect_negative_records()
    external = collect_external_yolo_records()
    all_records = positives + negatives + external
    if not all_records:
        raise RuntimeError("No dataset records found in data/raw or data/external")

    random.seed(SEED)
    random.shuffle(all_records)
    train_count = max(1, int(len(all_records) * TRAIN_RATIO))

    for idx, record in enumerate(all_records):
        split = "train" if idx < train_count else "val"
        img_path = OUT_IMAGES / split / record["name"]
        label_path = OUT_LABELS / split / f"{Path(record['name']).stem}.txt"
        img_path.write_bytes(record["image_bytes"])
        _write_label_file(label_path, record["label_lines"])

    write_dataset_yaml()
    print(
        f"prepared dataset: total={len(all_records)} "
        f"positive={len(positives)} negative={len(negatives)} external={len(external)} "
        f"train={train_count} "
        f"val={len(all_records) - train_count}"
    )
    print(f"dataset yaml: {DATASET_YAML}")


if __name__ == "__main__":
    main()
