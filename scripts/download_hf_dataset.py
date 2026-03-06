from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "talex02/wow-fishing-bobber-dataset"
DEFAULT_OUT_DIR = ROOT / "data" / "external" / "talex02-wow-fishing-bobber-dataset"
TREE_API = "https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _fetch_json(url: str) -> list[dict[str, object]]:
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())


def main() -> None:
    repo = DEFAULT_REPO
    out_dir = DEFAULT_OUT_DIR
    tree_url = TREE_API.format(repo=quote(repo, safe="/"))
    items = _fetch_json(tree_url)

    dataset_files = [
        item
        for item in items
        if item.get("type") == "file" and str(item.get("path", "")).startswith("dataset/")
    ]
    if not dataset_files:
        raise RuntimeError(f"No visual dataset files found in {repo}")

    downloaded = 0
    skipped = 0
    for item in dataset_files:
        rel_path = Path(str(item["path"]))
        size = int(item.get("size", 0) or 0)
        target = out_dir / rel_path
        if target.exists() and target.stat().st_size == size:
            skipped += 1
            continue
        url = RESOLVE_URL.format(repo=repo, path=quote(rel_path.as_posix(), safe="/"))
        _download(url, target)
        downloaded += 1
        print(f"downloaded {rel_path}")

    print(
        f"dataset sync complete: repo={repo} downloaded={downloaded} skipped={skipped} "
        f"target={out_dir}"
    )


if __name__ == "__main__":
    main()
