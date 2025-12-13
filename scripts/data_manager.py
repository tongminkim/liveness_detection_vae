from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


def _is_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def select_device(prefer_mps: bool = True) -> torch.device:
    """
    Pick the best available device.

    Args:
        prefer_mps: If True, choose MPS over CUDA when both exist.
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    # Repository guideline: prefer mps; if unavailable, stay on CPU.
    return torch.device("cpu")


def resolve_dtype(use_half: bool = True) -> torch.dtype:
    """Return the desired tensor dtype."""
    return torch.float16 if use_half else torch.float32


def _candidate_zip_paths(base_dir: Path, dataset_name: str) -> list[Path]:
    """
    Return likely locations for a dataset zip, ordered by preference.

    This helps Colab users who often keep archives under ``datasets`` on Drive
    while the notebooks default to ``local_data``.
    """

    filename = f"{dataset_name}.zip"
    candidates = [
        base_dir / filename,
        base_dir / dataset_name / filename,
        base_dir.parent / filename,
        base_dir.parent / "datasets" / filename,
        base_dir.parent / "local_data" / filename,
    ]

    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


@dataclass
class DatasetLocalCache:
    """
    Manage moving a zipped dataset into fast local storage and unpacking it.

    The class exposes a single public method, :meth:`prepare`, that returns
    the directory containing the extracted dataset.
    """

    zip_path: Path
    extract_dir: Path

    def _find_data_dir(self) -> Optional[Path]:
        """Return a directory that already contains .npz files if present."""
        candidates = [
            # Common extraction targets
            self.extract_dir,
            self.extract_dir / self.zip_path.stem,
            # Sometimes data is already sitting next to the zip on Drive
            self.zip_path.parent / self.zip_path.stem,
            self.zip_path.parent / self.zip_path.stem / self.zip_path.stem,
            # Or directly under the parent of extract_dir (if names changed)
            self.extract_dir.parent / self.zip_path.stem,
            self.extract_dir.parent / self.zip_path.stem / self.zip_path.stem,
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.npz")):
                return candidate
        return None

    def _materialize_zip(self) -> Path:
        """
        Copy the zip to the extraction root if necessary and return the path
        we should unpack from.
        """
        if not self.zip_path.exists():
            dataset_name = self.zip_path.stem
            search_roots = [
                self.zip_path.parent,
                self.extract_dir,
                self.extract_dir.parent,
                self.extract_dir.parent.parent,
            ]
            candidates: list[Path] = []
            for root in search_roots:
                candidates.extend(_candidate_zip_paths(root, dataset_name))

            zip_fallback = next((p for p in candidates if p.exists()), None)
            if zip_fallback is None:
                searched = "\n".join(f"- {p}" for p in candidates)
                raise FileNotFoundError(
                    f"Zip file not found: {self.zip_path}\n"
                    f"Also tried:\n{searched}\n"
                    f"Place {dataset_name}.zip under one of the above paths "
                    f"or set BANDVAE_DATA_DIR to an extracted folder."
                )
            self.zip_path = zip_fallback

        # Keep the zip next to the extracted data (avoids scattering files
        # under the project root when extract_dir is already the target dir).
        local_zip = self.extract_dir / self.zip_path.name
        local_zip.parent.mkdir(parents=True, exist_ok=True)

        if local_zip.resolve() == self.zip_path.resolve():
            return local_zip

        if (
            not local_zip.exists()
            or self.zip_path.stat().st_mtime > local_zip.stat().st_mtime
        ):
            shutil.copy2(self.zip_path, local_zip)

        return local_zip

    def prepare(self, force: bool = False) -> Path:
        """
        Ensure the dataset is extracted locally.

        Args:
            force: Re-extract even if files already exist.
        """
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        existing = self._find_data_dir()
        if not force and existing:
            self.extract_dir = existing
            return existing

        zip_src = self._materialize_zip()
        shutil.unpack_archive(str(zip_src), str(self.extract_dir))
        extracted = self._find_data_dir()
        if extracted:
            self.extract_dir = extracted
            return extracted
        return self.extract_dir


def prepare_band_dataset(
    base_dir: Path,
    dataset_name: str = "20GBprocessed",
    local_root: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Prepare the band-vae dataset in local storage and return the path.

    Args:
        base_dir: Project root containing ``{dataset_name}.zip``.
    dataset_name: Dataset folder name inside the archive.
    local_root: Where to extract. Defaults to /content/local_data on Colab,
                otherwise under ``base_dir``.
    force: If True, always re-extract.
    """
    # Allow explicit override for quick fixes.
    env_override = os.environ.get("BANDVAE_DATA_DIR")
    if env_override:
        return Path(env_override)

    base_dir = Path(base_dir)
    if local_root is None:
        if _is_colab():
            local_root = Path("/content/local_data")
        else:
            local_root = base_dir

    zip_candidates = _candidate_zip_paths(base_dir, dataset_name)
    zip_path = next((p for p in zip_candidates if p.exists()), None)
    if zip_path is None:
        searched = "\n".join(f"- {p}" for p in zip_candidates)
        raise FileNotFoundError(
            f"Zip file not found for dataset '{dataset_name}'. "
            f"Checked the following locations:\n{searched}\n"
            f"Place {dataset_name}.zip under one of these paths "
            f"(commonly ProjectRoot/datasets) or set BANDVAE_DATA_DIR "
            f"to an extracted folder containing .npz files."
        )
    # Extract directly under local_root; most archives already contain a
    # top-level {dataset_name} folder, so adding another level creates
    # the annoying nested 20GBprocessed/20GBprocessed structure.
    extract_dir = Path(local_root)

    cache = DatasetLocalCache(zip_path=zip_path, extract_dir=extract_dir)
    existing = cache._find_data_dir()
    if existing and not force:
        return existing
    return cache.prepare(force=force)
