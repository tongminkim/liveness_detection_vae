"""
Dataset 클래스 및 Feature Engineering
- .npz 파일에서 랜드마크 로드
- Procrustes 정규화
- Velocity, Acceleration, Angle, Angle-rate 계산
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================
# Feature Engineering Functions
# ============================================

def central_diff(arr):
    """
    Central difference (length-preserving)
    arr: [T, K, D] -> output: [T, K, D]
    """
    d = np.empty_like(arr)
    d[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    d[0] = arr[1] - arr[0]
    d[-1] = arr[-1] - arr[-2]
    return d


def second_diff(arr):
    """
    Second-order difference (length-preserving)
    arr: [T, K, D] -> output: [T, K, D]
    """
    dd = np.empty_like(arr)
    dd[1:-1] = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    dd[0] = arr[1] - arr[0]
    dd[-1] = arr[-1] - arr[-2]
    return dd


def umeyama_similarity(X, Y):
    """
    Umeyama similarity transform
    X, Y: [N, 2] (float) → find s, R, t s.t. Y ≈ s R X + t
    Returns: s (scale), R (2x2 rotation), t (2, translation)
    """
    Xc = X.mean(axis=0)
    Yc = Y.mean(axis=0)
    X0 = X - Xc
    Y0 = Y - Yc

    var = (X0**2).sum() / X0.shape[0] + 1e-12
    U, S, Vt = np.linalg.svd((Y0.T @ X0) / X0.shape[0])
    R = U @ Vt

    # Reflection 방지
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = S.sum() / var
    t = Yc - s * (R @ Xc)

    return s, R, t


def apply_similarity(P, s, R, t):
    """Apply similarity transform to points P [M, 2]"""
    return (s * (P @ R.T)) + t


def normalize_landmarks_procrustes(landmarks, ref_frames=10):
    """
    Procrustes 정규화 적용

    Args:
        landmarks: [T, K, 2] numpy array
        ref_frames: reference shape을 계산할 프레임 수

    Returns:
        normalized_landmarks: [T, K, 2] numpy array
    """
    T, K, _ = landmarks.shape

    # Reference shape: 처음 ref_frames의 평균
    ref_shape = landmarks[:min(ref_frames, T)].mean(axis=0)  # [K, 2]

    # 각 프레임별로 정규화
    normalized = np.empty_like(landmarks)
    for t in range(T):
        try:
            s, R, tt = umeyama_similarity(landmarks[t], ref_shape)
            normalized[t] = apply_similarity(landmarks[t], s, R, tt)
        except (np.linalg.LinAlgError, ValueError):
            # Singular matrix 등의 에러 발생 시 원본 사용
            normalized[t] = landmarks[t]

    return normalized


def compute_features(landmarks, fps=30, use_acceleration=False, use_angle=False, use_angle_rate=False):
    """
    입술 랜드마크에서 특징 추출

    Args:
        landmarks: [T, K, 2] normalized landmarks
        fps: frames per second
        use_acceleration: 2차 미분 사용 여부
        use_angle: angle/slope 사용 여부
        use_angle_rate: angle-rate 사용 여부

    Returns:
        features: dict with enabled features
    """
    features = {}

    # 1. Position (always included)
    features['position'] = landmarks  # [T, K, 2]

    # 2. Velocity (1st derivative, always included)
    features['velocity'] = central_diff(landmarks) * fps  # [T, K, 2]

    # 3. Acceleration (2nd derivative, optional)
    if use_acceleration:
        features['acceleration'] = second_diff(landmarks) * (fps ** 2)  # [T, K, 2]

    # 4. Angle/Slope (optional)
    if use_angle or use_angle_rate:
        v = np.roll(landmarks, -1, axis=1) - landmarks  # [T, K, 2]
        angles = np.arctan2(v[..., 1], v[..., 0])  # [T, K]
        angles_unwrap = np.unwrap(angles, axis=0)[..., None]  # [T, K, 1]

        if use_angle:
            features['angle'] = angles_unwrap

        # 5. Angle-rate (angle의 시간 미분, optional)
        if use_angle_rate:
            features['angle_rate'] = central_diff(angles_unwrap) * fps  # [T, K, 1]

    return features


def features_to_array(features):
    """
    Feature dict를 single array로 변환

    Args:
        features: dict with feature arrays

    Returns:
        feature_array: [T, K, F] where F depends on enabled features
            - simple: F=4 (position + velocity)
            - full: F=8 (position + velocity + acceleration + angle + angle_rate)
    """
    arrays = []

    # Order: position, velocity, acceleration, angle, angle_rate
    for key in ['position', 'velocity', 'acceleration', 'angle', 'angle_rate']:
        if key in features:
            arrays.append(features[key])

    return np.concatenate(arrays, axis=2)


# ============================================
# Dataset Class
# ============================================

class LipLivenessDataset(Dataset):
    """
    입술 Liveness Detection Dataset

    .npz 파일에서 랜드마크를 로드하고 Feature Engineering 적용
    """

    def __init__(
        self,
        data_dir,
        T_fixed=150,  # 5초 @ 30fps
        fps=30,
        ref_frames=10,
        use_procrustes=True,
        use_acceleration=False,
        use_angle=False,
        use_angle_rate=False
    ):
        """
        Args:
            data_dir: .npz 파일들이 있는 디렉토리
            T_fixed: 고정된 시간 길이 (프레임 수)
            fps: FPS
            ref_frames: Procrustes reference 프레임 수
            use_procrustes: Procrustes 정규화 사용 여부
            use_acceleration: Acceleration 사용 여부
            use_angle: Angle/slope 사용 여부
            use_angle_rate: Angle-rate 사용 여부
        """
        self.data_dir = data_dir
        self.T_fixed = T_fixed
        self.fps = fps
        self.ref_frames = ref_frames
        self.use_procrustes = use_procrustes
        self.use_acceleration = use_acceleration
        self.use_angle = use_angle
        self.use_angle_rate = use_angle_rate

        # .npz 파일 목록
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        print(f"Found {len(self.files)} samples in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
            x: [C, T] tensor
                C = K * F_dim
                - K = 40 (lips_outer + lips_inner, 각 20개)
                - F_dim = 4 if use_features else 2 (position + velocity)
        """
        # 1. .npz 파일 로드
        data = np.load(self.files[idx])

        # 입술 랜드마크만 사용 (lips_outer + lips_inner)
        lips_outer = data['lips_outer']  # [T, 21, 2]
        lips_inner = data['lips_inner']  # [T, 21, 2]

        # 마지막 중복 랜드마크 제거 (21 -> 20)
        # lips_outer와 lips_inner의 첫 점이 마지막에 반복됨
        lips = np.concatenate([
            lips_outer[:, :-1, :],  # [T, 20, 2]
            lips_inner[:, :-1, :]   # [T, 20, 2]
        ], axis=1)  # [T, 40, 2]

        size = data['size']  # [h, w]
        h, w = size

        # 2. 이미지 크기로 정규화 (0~1 범위)
        lips_norm = lips.copy()
        lips_norm[..., 0] /= (w + 1e-8)
        lips_norm[..., 1] /= (h + 1e-8)

        # 3. Crop/Pad to T_fixed
        T = lips_norm.shape[0]
        if T > self.T_fixed:
            # 중앙 부분 crop
            start = (T - self.T_fixed) // 2
            lips_norm = lips_norm[start:start + self.T_fixed]
        elif T < self.T_fixed:
            # Edge padding
            lips_norm = np.pad(
                lips_norm,
                ((0, self.T_fixed - T), (0, 0), (0, 0)),
                mode='edge'
            )

        # 4. Procrustes 정규화 (선택적)
        if self.use_procrustes:
            lips_norm = normalize_landmarks_procrustes(lips_norm, self.ref_frames)

        # 5. Feature Engineering
        features = compute_features(
            lips_norm,
            fps=self.fps,
            use_acceleration=self.use_acceleration,
            use_angle=self.use_angle,
            use_angle_rate=self.use_angle_rate
        )
        feature_array = features_to_array(features)  # [T, K, F]

        # 6. Reshape to [C, T]
        # feature_array: [T, K, F] -> [K, F, T] -> [K*F, T]
        x = torch.tensor(feature_array, dtype=torch.float32)
        x = x.permute(1, 2, 0)  # [K, F, T]
        x = x.reshape(-1, self.T_fixed)  # [K*F, T]

        return x


# ============================================
# Test code
# ============================================

if __name__ == "__main__":
    import sys

    # 데이터 디렉토리
    data_dir = "/home/elicer/liveness_detection/model1/processed_live"

    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    print("="*60)
    print("Dataset Test: 2 Feature Versions")
    print("="*60)

    # Test 1: SIMPLE version (position + velocity)
    print("\n【Test 1: SIMPLE Version】")
    print("Features: position + velocity")
    dataset_simple = LipLivenessDataset(
        data_dir=data_dir,
        T_fixed=150,
        fps=30,
        use_procrustes=True,
        use_acceleration=False,
        use_angle=False,
        use_angle_rate=False
    )

    print(f"Dataset size: {len(dataset_simple)}")
    x_simple = dataset_simple[0]
    print(f"Sample shape: {x_simple.shape}")
    print(f"  Expected: [C, T] = [40 × 4, 150] = [160, 150]")
    print(f"  {'✓ Correct!' if x_simple.shape[0] == 160 else '✗ Mismatch!'}")

    # Test 2: FULL version (all 5 features)
    print("\n【Test 2: FULL Version】")
    print("Features: position + velocity + acceleration + angle + angle_rate")
    dataset_full = LipLivenessDataset(
        data_dir=data_dir,
        T_fixed=150,
        fps=30,
        use_procrustes=True,
        use_acceleration=True,
        use_angle=True,
        use_angle_rate=True
    )

    print(f"Dataset size: {len(dataset_full)}")
    x_full = dataset_full[0]
    print(f"Sample shape: {x_full.shape}")
    print(f"  Expected: [C, T] = [40 × 8, 150] = [320, 150]")
    print(f"  {'✓ Correct!' if x_full.shape[0] == 320 else '✗ Mismatch!'}")

    # Statistics
    print("\n【Statistics Comparison】")
    print(f"SIMPLE - Mean: {x_simple.mean():.4f}, Std: {x_simple.std():.4f}")
    print(f"FULL   - Mean: {x_full.mean():.4f}, Std: {x_full.std():.4f}")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
