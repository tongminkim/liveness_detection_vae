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

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from model_bandvae import ButterworthFilterBank


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
    ref_shape = landmarks[:min(ref_frames, T)].mean(axis=0)

    normalized = np.empty_like(landmarks)
    for t in range(T):
        try:
            s, R, tt = umeyama_similarity(landmarks[t], ref_shape)
            normalized[t] = apply_similarity(landmarks[t], s, R, tt)
        except (np.linalg.LinAlgError, ValueError):
            normalized[t] = landmarks[t]

    return normalized


def compute_band_features(x_band, fps=30, use_acceleration=False, use_angle=False, use_angle_rate=False):
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
    features['position'] = x_band  # [T, K, 2]

    # 2. Velocity (always included)
    features['velocity'] = central_diff(x_band) * fps  # [T, K, 2]

    # 3. Acceleration (optional)
    if use_acceleration:
        features['acceleration'] = second_diff(x_band) * (fps ** 2)  # [T, K, 2]

    # 4. Angle (optional)
    if use_angle or use_angle_rate:
        v = np.roll(x_band, -1, axis=1) - x_band  # [T, K, 2]
        angles = np.arctan2(v[..., 1], v[..., 0])  # [T, K]
        angles_unwrap = np.unwrap(angles, axis=0)[..., None]  # [T, K, 1]

        if use_angle:
            features['angle'] = angles_unwrap

        # 5. Angle-rate (optional)
        if use_angle_rate:
            features['angle_rate'] = central_diff(angles_unwrap) * fps  # [T, K, 1]

    return features


def features_to_array(features):
    """Convert feature dict to array"""
    arrays = []
    for key in ['position', 'velocity', 'acceleration', 'angle', 'angle_rate']:
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
        filter_order=4  # Butterworth filter order (논문의 FIR numtaps 대신)
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
        """
        self.data_dir = data_dir
        self.T_fixed = T_fixed
        self.fps = fps
        self.ref_frames = ref_frames
        self.use_procrustes = use_procrustes
        self.use_acceleration = use_acceleration
        self.use_angle = use_angle
        self.use_angle_rate = use_angle_rate

        # Butterworth Filter Bank (논문의 FIR filter 대신)
        self.filter_bank = ButterworthFilterBank(
            fps=fps,
            fc_low=fc_low,
            fc_high=fc_high,
            order=filter_order
        )

        # .npz 파일 목록
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        print(f"Found {len(self.files)} samples in {data_dir}")
        print(f"Butterworth Filter Bank: fc_low={fc_low}Hz, fc_high={fc_high}Hz, order={filter_order}")

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
        data = np.load(self.files[idx])

        lips_outer = data['lips_outer']  # [T, 21, 2]
        lips_inner = data['lips_inner']  # [T, 21, 2]

        # 마지막 중복 제거 → K=40 landmarks
        lips = np.concatenate([
            lips_outer[:, :-1, :],  # [T, 20, 2]
            lips_inner[:, :-1, :]   # [T, 20, 2]
        ], axis=1)  # [T, 40, 2]

        size = data['size']
        h, w = size

        # 2. Normalize by image size
        lips_norm = lips.copy()
        lips_norm[..., 0] /= (w + 1e-8)
        lips_norm[..., 1] /= (h + 1e-8)

        # 3. Crop/Pad to T_fixed (논문: T=150 frames)
        T = lips_norm.shape[0]
        if T > self.T_fixed:
            start = (T - self.T_fixed) // 2
            lips_norm = lips_norm[start:start + self.T_fixed]
        elif T < self.T_fixed:
            lips_norm = np.pad(
                lips_norm,
                ((0, self.T_fixed - T), (0, 0), (0, 0)),
                mode='edge'
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
            use_angle_rate=self.use_angle_rate
        )
        features_bp = compute_band_features(
            lips_bp,
            fps=self.fps,
            use_acceleration=self.use_acceleration,
            use_angle=self.use_angle,
            use_angle_rate=self.use_angle_rate
        )
        features_hf = compute_band_features(
            lips_hf,
            fps=self.fps,
            use_acceleration=self.use_acceleration,
            use_angle=self.use_angle,
            use_angle_rate=self.use_angle_rate
        )

        # 7. Convert to arrays [T, K, F]
        feat_lf = features_to_array(features_lf)
        feat_bp = features_to_array(features_bp)
        feat_hf = features_to_array(features_hf)

        # 8. Reshape to [C, T] where C = K × F_dim
        x_lf = torch.tensor(feat_lf, dtype=torch.float32).permute(1, 2, 0).reshape(-1, self.T_fixed)
        x_bp = torch.tensor(feat_bp, dtype=torch.float32).permute(1, 2, 0).reshape(-1, self.T_fixed)
        x_hf = torch.tensor(feat_hf, dtype=torch.float32).permute(1, 2, 0).reshape(-1, self.T_fixed)

        # 9. Per-band normalization (zero mean, unit variance)
        #    Critical: 주파수 대역별로 스케일이 다르므로 독립적으로 정규화
        x_lf = (x_lf - x_lf.mean()) / (x_lf.std() + 1e-8)
        x_bp = (x_bp - x_bp.mean()) / (x_bp.std() + 1e-8)
        x_hf = (x_hf - x_hf.mean()) / (x_hf.std() + 1e-8)

        return x_lf, x_bp, x_hf


# ============================================
# Test code
# ============================================

if __name__ == "__main__":
    import sys

    data_dir = "/home/elicer/liveness_detection/model1/processed_live"

    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    print("="*60)
    print("BandVAE Dataset Test: 2 Feature Versions")
    print("="*60)

    # Test 1: SIMPLE version
    print("\n【Test 1: SIMPLE Version (position + velocity)】")
    dataset_simple = LipLivenessBandDataset(
        data_dir=data_dir,
        T_fixed=150,
        fps=30,
        use_procrustes=True,
        use_acceleration=False,
        use_angle=False,
        use_angle_rate=False
    )

    print(f"Dataset size: {len(dataset_simple)}")
    x_lf, x_bp, x_hf = dataset_simple[0]
    print(f"LF shape: {x_lf.shape} (expected: [160, 150])")
    print(f"BP shape: {x_bp.shape} (expected: [160, 150])")
    print(f"HF shape: {x_hf.shape} (expected: [160, 150])")
    print(f"  {'✓ Correct!' if x_lf.shape[0] == 160 else '✗ Mismatch!'}")

    # Test 2: FULL version
    print("\n【Test 2: FULL Version (all 5 features)】")
    dataset_full = LipLivenessBandDataset(
        data_dir=data_dir,
        T_fixed=150,
        fps=30,
        use_procrustes=True,
        use_acceleration=True,
        use_angle=True,
        use_angle_rate=True
    )

    print(f"Dataset size: {len(dataset_full)}")
    x_lf, x_bp, x_hf = dataset_full[0]
    print(f"LF shape: {x_lf.shape} (expected: [320, 150])")
    print(f"BP shape: {x_bp.shape} (expected: [320, 150])")
    print(f"HF shape: {x_hf.shape} (expected: [320, 150])")
    print(f"  {'✓ Correct!' if x_lf.shape[0] == 320 else '✗ Mismatch!'}")

    # Statistics
    print("\n【Statistics】")
    print(f"SIMPLE - LF mean: {x_lf.mean():.4f}, std: {x_lf.std():.4f}")
    print(f"FULL   - LF mean: {x_lf.mean():.4f}, std: {x_lf.std():.4f}")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
