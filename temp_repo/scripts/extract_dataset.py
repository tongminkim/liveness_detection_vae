# liveness_detection_vae-feature-band-vae-visualization/scripts/extract_dataset.py
"""
Extract the `final_test` split from `data_split.json` and copy the listed files
into a new `test/extracted` directory while preserving the original
sub‑directory structure (`real` / `fake`).

The JSON contains absolute paths that point to the original dataset location,
e.g.:

    /home/elicer/liveness_detection/model1/test_balanced_npz/real/...

Within this repository the same files are stored under `test/real` and
`test/fake`. This script maps the absolute paths to the repository paths,
creates the target directories and copies the files.

Usage:
    python scripts/extract_dataset.py
"""

import json
import os
import shutil
from pathlib import Path

# Prefix used in the JSON paths that should be stripped to obtain the repository‑relative path.
_JSON_PREFIX = "/home/elicer/liveness_detection/model1/test_balanced_npz/"


def _load_data_split(json_path: Path) -> dict:
    """Load the JSON file containing the dataset splits."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_file(src: Path, dst_dir: Path) -> None:
    """Copy a single file to the destination directory, creating the directory if needed."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.is_file():
        shutil.copy2(src, dst)
        print(f"Copied: {src} → {dst}")
    else:
        print(f"Warning: source file does not exist – skipping: {src}")


def main() -> None:
    # Resolve the repository root (the directory containing this script's parent).
    repo_root = Path(__file__).resolve().parents[1]

    # Load data_split.json.
    split_path = repo_root / "data_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"data_split.json not found at expected location: {split_path}"
        )

    data = _load_data_split(split_path)

    # Extract the final_test split.
    final_test = data.get("final_test")
    if not final_test:
        raise KeyError("`final_test` key not found in data_split.json")

    # Destination base directory: <repo_root>/test/extracted
    extracted_base = repo_root / "test" / "extracted"

    for category in ("real", "fake"):
        src_list = final_test.get(category, [])
        if not src_list:
            print(f"No entries for category '{category}' – skipping.")
            continue

        # Destination directory for this category.
        dst_category_dir = extracted_base / category

        for abs_path in src_list:
            if not abs_path.startswith(_JSON_PREFIX):
                print(
                    f"Skipping unexpected path (does not start with known prefix): {abs_path}"
                )
                continue

            # Derive the relative path inside the repository, e.g. "real/…/file.npz"
            rel_path = abs_path[len(_JSON_PREFIX) :]  # strip the prefix
            # The repository stores files under `test/<rel_path>`
            src_path = repo_root / "test" / rel_path

            _copy_file(src_path, dst_category_dir)


if __name__ == "__main__":
    main()
