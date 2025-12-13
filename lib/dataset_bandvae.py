"""
Dataset for Band-Split VAE (논문 Section 2 구현)

논문의 전체 파이프라인:
1. Geometric Normalization (Umeyama)
2. Frequency Decomposition (Filter Bank → LF/BP/HF)
3. Per-Band Feature Engineering (derivatives 계산)
4. Return 3 feature tensors (lf, bp, hf)

본 구현:
- 논문: FIR filter bank
- 여기서는: Butterworth IIR filter (병목 해소, 10-100배 빠름)
"""

import glob
import os

import numpy as np
import torch
from model_bandvae import ButterworthFilterBank
from torch.utils.data import Dataset

# ============================================
# Feature Engineering Functions (재사용)
# ============================================


def central_diff(arr):
    """Central difference (length-preserving)"""
    d = np.empty_like(arr)
    d[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    d[0] = arr[1] - arr[0]
    d[-1] = arr[-1] - arr[-2]
    return d


def second_diff(arr):
    """Second-order difference (length-preserving)"""
    dd = np.empty_like(arr)
    dd[1:-1] = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    dd[0] = arr[1] - arr[0]
    dd[-1] = arr[-1] - arr[-2]
    return dd


def umeyama_similarity(X, Y):
    """Umeyama similarity transform"""
    Xc = X.mean(axis=0)
    Yc = Y.mean(axis=0)
    X0 = X - Xc
    Y0 = Y - Yc

    var = (X0**2).sum() / X0.shape[0] + 1e-12
    U, S, Vt = np.linalg.svd((Y0.T @ X0) / X0.shape[0])
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = S.sum() / var
    t = Yc - s * (R @ Xc)

    return s, R, t


def apply_similarity(P, s, R, t):
    """Apply similarity transform"""
    return (s * (P @ R.T)) + t


def normalize_landmarks_procrustes(landmarks, ref_frames=10):
    """Procrustes normalization"""
    T, K, _ = landmarks.shape
    ref_shape = landmarks[: min(ref_frames, T)].mean(axis=0)

    normalized = np.empty_like(landmarks)
    for t in range(T):
        try:
            s, R, tt = umeyama_similarity(landmarks[t], ref_shape)
            normalized[t] = apply_similarity(landmarks[t], s, R, tt)
        except (np.linalg.LinAlgError, ValueError):
            normalized[t] = landmarks[t]

    return normalized


def compute_band_features(
    x_band, fps=30, use_acceleration=False, use_angle=False, use_angle_rate=False
):
    """
    특정 주파수 대역의 position에서 derivatives 계산

    Args:
        x_band: [T, K, 2] - 특정 대역의 position
        fps: frames per second
        use_acceleration, use_angle, use_angle_rate: feature flags

    Returns:
        features: dict with enabled features
    """
    features = {}

    # 1. Position (always included)
    features["position"] = x_band  # [T, K, 2]

    # 2. Velocity (always included)
    features["velocity"] = central_diff(x_band) * fps  # [T, K, 2]

    # 3. Acceleration (optional)
    if use_acceleration:
        features["acceleration"] = second_diff(x_band) * (fps**2)  # [T, K, 2]

    # 4. Angle (optional)
    if use_angle or use_angle_rate:
        v = np.roll(x_band, -1, axis=1) - x_band  # [T, K, 2]
        angles = np.arctan2(v[..., 1], v[..., 0])  # [T, K]
        angles_unwrap = np.unwrap(angles, axis=0)[..., None]  # [T, K, 1]

        if use_angle:
            features["angle"] = angles_unwrap

        # 5. Angle-rate (optional)
        if use_angle_rate:
            features["angle_rate"] = central_diff(angles_unwrap) * fps  # [T, K, 1]

    return features


def features_to_array(features):
    """Convert feature dict to array"""
    arrays = []
    for key in ["position", "velocity", "acceleration", "angle", "angle_rate"]:
        if key in features:
            arrays.append(features[key])
    return np.concatenate(arrays, axis=2)


# ============================================
# Dataset Class
# ============================================


class LipLivenessBandDataset(Dataset):
    """
    Band-Split VAE용 Dataset (논문 Section 2 구현)

    논문의 파이프라인:
    1. Geometric normalization (Umeyama)
    2. Frequency decomposition (Filter Bank → LF/BP/HF)
    3. Per-band feature engineering (derivatives)
    4. Return 3 tensors (lf, bp, hf)

    본 구현:
    - Butterworth IIR filter 사용 (FIR 대신, 병목 해소)
    """

    def __init__(
        self,
        data_dir,
        T_fixed=150,
        fps=30,
        ref_frames=10,
        use_procrustes=True,
        use_acceleration=False,
        use_angle=False,
        use_angle_rate=False,
        # Filter bank parameters
        fc_low=2.0,
        fc_high=8.0,
        filter_order=4,  # Butterworth filter order (논문의 FIR numtaps 대신)
        random_crop=False,  # If True, randomly crop T_fixed frames instead of center crop
        dtype=torch.float32,
        skip_corrupted=True,  # If True, drop unreadable .npz files up front
        processing_dtype: torch.dtype = torch.float32,  # internal numeric stability
        output_dtype: torch.dtype | None = None,  # cast outputs before returning
    ):
        """
        Args:
            data_dir: .npz 파일 디렉토리
            T_fixed: 고정 시간 길이 (논문: T=150 frames @ 30fps)
            fps: FPS
            ref_frames: Procrustes reference 프레임 수
            use_procrustes: Procrustes 정규화 사용
            use_acceleration: Acceleration 사용 (논문: Δ²x)
            use_angle: Angle 사용 (논문: θ)
            use_angle_rate: Angle-rate 사용 (논문: Δθ)
            fc_low: Low-pass cutoff (Hz) - 논문: 2.0 Hz
            fc_high: High-pass cutoff (Hz) - 논문: 8.0 Hz
            filter_order: Butterworth filter order (4-6 일반적)
            skip_corrupted: True일 때 읽을 수 없는 .npz 파일을 사전에 제외
            processing_dtype: Filtering/normalization precision (keep at fp32)
            output_dtype: If set, tensors are cast to this dtype before return
        """
        self.data_dir = data_dir
        self.T_fixed = T_fixed
        self.fps = fps
        self.ref_frames = ref_frames
        self.use_procrustes = use_procrustes
        self.use_acceleration = use_acceleration
        self.use_angle = use_angle
        self.use_angle_rate = use_angle_rate
        self.random_crop = random_crop
        self.dtype = dtype  # kept for backward compatibility
        self.skip_corrupted = skip_corrupted
        self.processing_dtype = processing_dtype
        self.output_dtype = output_dtype

        # Butterworth Filter Bank (논문의 FIR filter 대신)
        self.filter_bank = ButterworthFilterBank(
            fps=fps, fc_low=fc_low, fc_high=fc_high, order=filter_order
        )

        # .npz 파일 목록
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if self.skip_corrupted:
            healthy_files = []
            dropped = 0
            for path in self.files:
                try:
                    with np.load(path) as data:
                        # Ensure required keys exist; loading also catches EOF.
                        _ = data["lips_outer"]
                        _ = data["lips_inner"]
                        _ = data["size"]
                    healthy_files.append(path)
                except Exception as exc:
                    dropped += 1
                    print(f"Warning: skipping corrupted sample '{path}': {exc}")
            self.files = healthy_files
            if dropped:
                print(
                    f"Skipped {dropped} corrupted files; {len(self.files)} remaining."
                )
                print(
                    f"Skipped {dropped} corrupted files; {len(self.files)} remaining."
                )
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        print(f"Found {len(self.files)} samples in {data_dir}")
        print(
            f"Butterworth Filter Bank: fc_low={fc_low}Hz, fc_high={fc_high}Hz, order={filter_order}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        논문 Section 2의 전체 파이프라인 구현

        Returns:
            x_lf, x_bp, x_hf: 각각 [C, T] tensor
                - C = K * F_dim
                - SIMPLE: F_dim=4 (position+velocity) → C=160
                - FULL: F_dim=8 (all features) → C=320
        """
        # 1. Load .npz file
        try:
            data = np.load(self.files[idx])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sample '{self.files[idx]}'. "
                "The file may be truncated; delete it or re-extract the dataset."
            ) from exc

        lips_outer = data["lips_outer"]  # [T, 21, 2]
        lips_inner = data["lips_inner"]  # [T, 21, 2]

        # 마지막 중복 제거 → K=40 landmarks
        lips = np.concatenate(
            [
                lips_outer[:, :-1, :],  # [T, 20, 2]
                lips_inner[:, :-1, :],  # [T, 20, 2]
            ],
            axis=1,
        )  # [T, 40, 2]

        size = data["size"]
        h, w = size

        # 2. Normalize by image size
        lips_norm = lips.copy()
        lips_norm[..., 0] /= w + 1e-8
        lips_norm[..., 1] /= h + 1e-8

        # 3. Crop/Pad to T_fixed (논문: T=150 frames)
        T = lips_norm.shape[0]
        if T > self.T_fixed:
            if self.random_crop:
                # Random crop for training
                max_start = T - self.T_fixed
                start = np.random.randint(0, max_start + 1)
            else:
                # Center crop for validation/test
                start = (T - self.T_fixed) // 2
            lips_norm = lips_norm[start : start + self.T_fixed]
        elif T < self.T_fixed:
            lips_norm = np.pad(
                lips_norm, ((0, self.T_fixed - T), (0, 0), (0, 0)), mode="edge"
            )

        # 4. Step 1: Geometric Normalization (Section 2.2)
        #    "apply geometric normalization (scale-rotation-translation
        #     via Umeyama alignment)"
        if self.use_procrustes:
            lips_norm = normalize_landmarks_procrustes(lips_norm, self.ref_frames)

        # 5. Step 2: Frequency Decomposition (Section 2.1)
        #    "A lightweight FIR filter bank separates the signal into
        #     low-, mid-, and high-frequency streams"
        #    본 구현: Butterworth IIR filter (병목 해소)
        lips_lf, lips_bp, lips_hf = self.filter_bank.apply(lips_norm)

        # 6. Step 3: Per-Band Feature Engineering (Section 2.2)
        #    "construct per-landmark channels: position, velocity, acceleration,
        #     angle/slope, angle-rate"
        features_lf = compute_band_features(
            lips_lf,
            fps=self.fps,
            use_acceleration=self.use_acceleration,
            use_angle=self.use_angle,
            use_angle_rate=self.use_angle_rate,
        )
        features_bp = compute_band_features(
            lips_bp,
            fps=self.fps,
            use_acceleration=self.use_acceleration,
            use_angle=self.use_angle,
            use_angle_rate=self.use_angle_rate,
        )
        features_hf = compute_band_features(
            lips_hf,
            fps=self.fps,
            use_acceleration=self.use_acceleration,
            use_angle=self.use_angle,
            use_angle_rate=self.use_angle_rate,
        )

        # 7. Convert to arrays [T, K, F]
        feat_lf = features_to_array(features_lf)
        feat_bp = features_to_array(features_bp)
        feat_hf = features_to_array(features_hf)

        # 8. Reshape to [C, T] where C = K × F_dim
        x_lf = (
            torch.tensor(feat_lf, dtype=self.processing_dtype)
            .permute(1, 2, 0)
            .reshape(-1, self.T_fixed)
        )
        x_bp = (
            torch.tensor(feat_bp, dtype=self.processing_dtype)
            .permute(1, 2, 0)
            .reshape(-1, self.T_fixed)
        )
        x_hf = (
            torch.tensor(feat_hf, dtype=self.processing_dtype)
            .permute(1, 2, 0)
            .reshape(-1, self.T_fixed)
        )

        # 9. Per-band normalization (zero mean, unit variance)
        #    Critical: 주파수 대역별로 스케일이 다르므로 독립적으로 정규화
        x_lf = (x_lf - x_lf.mean()) / (x_lf.std() + 1e-8)
        x_bp = (x_bp - x_bp.mean()) / (x_bp.std() + 1e-8)
        x_hf = (x_hf - x_hf.mean()) / (x_hf.std() + 1e-8)

        if self.output_dtype and self.output_dtype != self.processing_dtype:
            x_lf = x_lf.to(self.output_dtype)
            x_bp = x_bp.to(self.output_dtype)
            x_hf = x_hf.to(self.output_dtype)

        return x_lf, x_bp, x_hf


# ============================================
# Test code
# ============================================

if __name__ == "__main__":
    import sys

    print(
        "This module provides the LipLivenessBandDataset class. "
        "Run training scripts or notebooks to use it."
    )
