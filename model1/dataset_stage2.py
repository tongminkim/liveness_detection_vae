"""
Stage 2 Dataset: Real + Fake data from data_split.json
- Loads pre-split train/test data
- Returns (x_lf, x_bp, x_hf, label) where label=0 for Real, label=1 for Fake
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt


class Stage2Dataset(Dataset):
    """Dataset for Stage 2 with Real + Fake labels"""

    def __init__(
        self,
        split_json_path,
        split_name,  # 'stage2_train' or 'final_test'
        T_fixed=300,
        fps=30,
        use_acceleration=True,
        use_angle=True,
        use_angle_rate=True,
        fc_low=2.0,
        fc_high=8.0,
        filter_order=4,
        random_crop=True
    ):
        """
        Args:
            split_json_path: Path to data_split.json
            split_name: 'stage2_train' or 'final_test'
            T_fixed: Fixed sequence length
            fps: Frame rate
            use_acceleration: Use acceleration features
            use_angle: Use angle features
            use_angle_rate: Use angle rate features
            fc_low: Low cutoff frequency (Hz)
            fc_high: High cutoff frequency (Hz)
            filter_order: Butterworth filter order
            random_crop: Random crop during training
        """
        self.T_fixed = T_fixed
        self.fps = fps
        self.use_acceleration = use_acceleration
        self.use_angle = use_angle
        self.use_angle_rate = use_angle_rate
        self.fc_low = fc_low
        self.fc_high = fc_high
        self.filter_order = filter_order
        self.random_crop = random_crop

        # Load split data
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)

        if split_name not in split_data:
            raise ValueError(f"split_name must be 'stage2_train' or 'final_test', got {split_name}")

        # Collect files and labels
        self.files = []
        self.labels = []

        split = split_data[split_name]

        # Real files (label=0)
        for file_path in split['real']:
            self.files.append(file_path)
            self.labels.append(0)

        # Fake files (label=1)
        for file_path in split['fake']:
            self.files.append(file_path)
            self.labels.append(1)

        print(f"Butterworth Filter Bank: fc_low={fc_low}Hz, fc_high={fc_high}Hz, order={filter_order}")
        print(f"Loaded {len(self.files)} samples ({split['real'].__len__()} real, {split['fake'].__len__()} fake)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Load and process a single sample"""
        # Load npz file
        data = np.load(self.files[idx])

        # Combine lips_outer and lips_inner to create 40 landmarks
        lips_outer = data['lips_outer']  # (T, 21, 2)
        lips_inner = data['lips_inner']  # (T, 21, 2)

        # Remove last duplicate points and concatenate
        landmarks = np.concatenate([
            lips_outer[:, :-1, :],  # (T, 20, 2)
            lips_inner[:, :-1, :]   # (T, 20, 2)
        ], axis=1)  # (T, 40, 2)

        # Normalize by image size
        size = data['size']
        h, w = size
        landmarks = landmarks.copy()
        landmarks[..., 0] /= (w + 1e-8)
        landmarks[..., 1] /= (h + 1e-8)

        label = self.labels[idx]

        # Build features
        features = self._build_features(landmarks)  # (T, 40, C_feature)

        # Crop/pad to T_fixed
        features = self._crop_or_pad(features)  # (T_fixed, 40, C_feature)

        # Apply band-split filtering
        x_lf, x_bp, x_hf = self._apply_filters(features)  # Each: (T_fixed, C_in_per_band)

        # Convert to tensors
        x_lf = torch.from_numpy(x_lf).float()
        x_bp = torch.from_numpy(x_bp).float()
        x_hf = torch.from_numpy(x_hf).float()
        label = torch.tensor(label, dtype=torch.long)

        return x_lf, x_bp, x_hf, label

    def _build_features(self, landmarks):
        """Build feature vector from landmarks (position + velocity + acceleration + angle + angle_rate)"""
        T, N, _ = landmarks.shape  # (T, 40, 2)

        # 1) Position (x, y)
        position = landmarks  # (T, 40, 2)

        # 2) Velocity (Δx, Δy)
        velocity = np.zeros_like(position)
        velocity[1:] = np.diff(position, axis=0) * self.fps

        # 3) Acceleration (Δ²x, Δ²y)
        if self.use_acceleration:
            acceleration = np.zeros_like(position)
            acceleration[1:] = np.diff(velocity, axis=0) * self.fps

        # 4) Angle (θ) and 5) Angle rate (Δθ)
        if self.use_angle or self.use_angle_rate:
            angle = np.arctan2(landmarks[:, :, 1], landmarks[:, :, 0])  # (T, 40)
            angle = np.expand_dims(angle, axis=-1)  # (T, 40, 1)

            if self.use_angle_rate:
                angle_rate = np.zeros_like(angle)
                angle_rate[1:] = np.diff(angle, axis=0) * self.fps

        # Concatenate features
        feature_list = [position, velocity]

        if self.use_acceleration:
            feature_list.append(acceleration)

        if self.use_angle:
            feature_list.append(angle)

        if self.use_angle_rate:
            feature_list.append(angle_rate)

        features = np.concatenate(feature_list, axis=-1)  # (T, 40, C_feature)

        return features

    def _crop_or_pad(self, features):
        """Crop or pad to T_fixed"""
        T, N, C = features.shape

        if T == self.T_fixed:
            return features
        elif T > self.T_fixed:
            # Crop
            if self.random_crop:
                start = np.random.randint(0, T - self.T_fixed + 1)
            else:
                start = 0  # Deterministic crop from beginning
            return features[start:start + self.T_fixed]
        else:
            # Pad
            pad_width = ((0, self.T_fixed - T), (0, 0), (0, 0))
            return np.pad(features, pad_width, mode='edge')

    def _apply_filters(self, features):
        """Apply Butterworth band-pass filters to create LF, BP, HF bands"""
        T, N, C = features.shape
        nyquist = self.fps / 2.0

        # LF: [0, fc_low] Hz
        b_lf, a_lf = butter(self.filter_order, self.fc_low / nyquist, btype='low')

        # BP: [fc_low, fc_high] Hz
        b_bp, a_bp = butter(self.filter_order, [self.fc_low / nyquist, self.fc_high / nyquist], btype='band')

        # HF: [fc_high, nyquist] Hz
        b_hf, a_hf = butter(self.filter_order, self.fc_high / nyquist, btype='high')

        # Apply filters to each channel
        x_lf = np.zeros_like(features)
        x_bp = np.zeros_like(features)
        x_hf = np.zeros_like(features)

        for n in range(N):
            for c in range(C):
                signal = features[:, n, c]
                x_lf[:, n, c] = filtfilt(b_lf, a_lf, signal)
                x_bp[:, n, c] = filtfilt(b_bp, a_bp, signal)
                x_hf[:, n, c] = filtfilt(b_hf, a_hf, signal)

        # Reshape to (N*C, T) to match model input format [C, T]
        x_lf = x_lf.transpose(1, 2, 0).reshape(N * C, T)
        x_bp = x_bp.transpose(1, 2, 0).reshape(N * C, T)
        x_hf = x_hf.transpose(1, 2, 0).reshape(N * C, T)

        # Per-band normalization (zero mean, unit variance)
        # Critical: Different frequency bands have different scales
        x_lf = (x_lf - x_lf.mean()) / (x_lf.std() + 1e-8)
        x_bp = (x_bp - x_bp.mean()) / (x_bp.std() + 1e-8)
        x_hf = (x_hf - x_hf.mean()) / (x_hf.std() + 1e-8)

        return x_lf, x_bp, x_hf
