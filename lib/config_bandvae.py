from __future__ import annotations

from pathlib import Path

import torch

# Default dataset location: project_root/datasets (handles the
# common double-nested case 20GBprocessed/20GBprocessed).
DATASET_NAME = "20GBprocessed"
DATA_ROOT = Path(__file__).resolve().parent.parent / "datasets"


def _resolve_data_dir(
    dataset_name: str = DATASET_NAME, base_dir: Path = DATA_ROOT
) -> str:
    """
    Pick the dataset directory, handling the common double-nested case
    produced by some archive tools.
    """
    base = Path(base_dir) / dataset_name
    nested = base / dataset_name
    if nested.exists():
        return str(nested)
    if base.exists():
        return str(base)
    return str(nested)


class Config:
    """
    Central configuration container.

    Parameters
    ----------
    mode : {"full", "simple"}
        Selects the feature set:
        * ``full``   – position, velocity, acceleration, angle, angle‑rate
        * ``simple`` – position, velocity only

    The class sets all required attributes for data handling,
    model construction and training.  No inheritance or polymorphism is
    used; the configuration is a plain object with clearly named fields.
    """

    # ------------------------------------------------------------------
    # Shared defaults (common to both modes)
    # ------------------------------------------------------------------
    data_dir: str
    save_dir_root: str = "runs/bandvae"

    # Data processing (paper: T = 150 frames @ 30 fps)
    T_fixed: int = 150
    fps: int = 30

    # Butterworth filter bank (replaces FIR bank in the paper)
    fc_low: float = 2.0  # low‑pass cutoff (Hz)
    fc_high: float = 8.0  # high‑pass cutoff (Hz)
    filter_order: int = 4

    # Model architecture (paper Section 2.1)
    C_h: int = 48  # hidden channels inside TCN
    C_z: int = 12  # latent dimension
    dilations: list[int] = [1, 2, 4]

    # Training hyper‑parameters
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 20
    val_split: float = 0.1

    # KL‑weight per band (paper Section 2.3)
    beta_lf: float = 1.0
    beta_bp: float = 1.0
    beta_hf: float = 1.0

    # Fusion penalty (not used in the current implementation)
    alpha_fusion: float = 0.1

    # System
    device: str = "cuda"
    num_workers: int = 8
    pin_memory: bool = True
    dtype: torch.dtype = torch.float16

    # ------------------------------------------------------------------
    # Mode‑specific settings (filled in __init__)
    # ------------------------------------------------------------------
    feature_mode: str
    use_velocity: bool
    use_acceleration: bool
    use_angle: bool
    use_angle_rate: bool
    K_landmarks: int
    F_dim: int
    C_in_per_band: int
    save_dir: str

    # ------------------------------------------------------------------
    def __init__(self, mode: str = "simple", data_dir: str | None = None):
        mode = mode.lower()
        if mode not in {"full", "simple"}:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'full' or 'simple'.")

        # Data path (auto-resolves nested dataset layout)
        self.data_dir = data_dir or _resolve_data_dir()

        # ------------------------------------------------------------------
        # Mode‑specific flags
        # ------------------------------------------------------------------
        self.feature_mode = mode

        if mode == "full":
            # FULL version – 5 features per landmark (8 channels)
            self.use_velocity = True
            self.use_acceleration = True
            self.use_angle = True
            self.use_angle_rate = True
            self.F_dim = 8
            self.save_dir = f"{self.save_dir_root}_full"
        else:  # simple
            # SIMPLE version – position + velocity only (4 channels)
            self.use_velocity = True
            self.use_acceleration = False
            self.use_angle = False
            self.use_angle_rate = False
            self.F_dim = 4
            self.save_dir = f"{self.save_dir_root}_simple"

        # ------------------------------------------------------------------
        # Constants that are the same for both modes
        # ------------------------------------------------------------------
        self.K_landmarks = 40  # lips only
        self.C_in_per_band = self.K_landmarks * self.F_dim

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """Human‑readable summary of the configuration."""
        lines = [
            f"Config (mode={self.feature_mode})",
            f"  Data: {self.data_dir}",
            f"  Save dir: {self.save_dir}",
            f"  Device: {self.device}, dtype: {self.dtype}",
            f"  Frames per sample: {self.T_fixed} @ {self.fps} fps",
            f"  Features per landmark: {self.F_dim} ({'full' if self.feature_mode == 'full' else 'simple'})",
            f"  Input channels / band: {self.C_in_per_band}",
            f"  TCN hidden channels: {self.C_h}, latent dim: {self.C_z}",
            f"  Dilations: {self.dilations}",
            f"  Filter: Butterworth order={self.filter_order}, fc_low={self.fc_low} Hz, fc_high={self.fc_high} Hz",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Backward‑compatible alias (existing code may import get_config)
# ----------------------------------------------------------------------
def get_config(mode: str = "simple") -> Config:  # pragma: no cover
    """
    Compatibility wrapper kept for legacy imports.  Internally it just
    constructs a :class:`Config` instance.
    """
    return Config(mode)
