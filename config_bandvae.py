"""
Configuration for Band-Split VAE (논문 Section 2 구현)

논문의 핵심 아키텍처:
- Frequency-Decoupled Feature-Space VAE
- 3-band split (LF/BP/HF)
- Per-band feature engineering
- Learnable weighted fusion

본 구현의 2가지 Feature 버전:
1. FULL: position + velocity + acceleration + angle + angle-rate (8 features per landmark)
2. SIMPLE: position + velocity only (4 features per landmark)

필터 변경:
- 논문: FIR filter bank
- 본 구현: Butterworth IIR filter (병목 해소, 10-100배 빠름)
"""


class BaseConfig:
    """
    Base configuration for Band-Split VAE

    논문 Section 2의 핵심 파라미터 기반
    """

    # Data paths
    data_dir = "/home/elicer/liveness_detection/model1/processed_live"
    save_dir = "runs/bandvae"

    # Data processing (논문: T=150 frames @ 30fps)
    T_fixed = 150  # 5초 @ 30fps
    fps = 30

    # Butterworth Filter Bank (논문의 FIR filter 대신)
    # 논문: "A lightweight FIR filter bank separates the signal..."
    # 본 구현: Butterworth IIR (10-100배 빠름, 핵심 아이디어 유지)
    fc_low = 2.0  # Low-pass cutoff (Hz) - 논문과 동일
    fc_high = 8.0  # High-pass cutoff (Hz) - 논문과 동일
    filter_order = 4  # Butterworth order (4-6 일반적, FIR numtaps 대신)

    # Model architecture (논문 Section 2.1)
    C_h = 48  # Hidden channels (TCN 내부 채널)
    C_z = 12  # Latent dimension (compact latent space)
    dilations = [1, 2, 4]  # TCN dilation factors

    # Training
    batch_size = 64
    lr = 1e-3
    epochs = 20
    val_split = 0.1

    # Per-band KL weights (논문 Section 2.3: β_b)
    beta_lf = 1.0
    beta_bp = 1.0
    beta_hf = 1.0

    # Fusion penalty (논문: "small fusion penalty L_mix")
    # 본 구현에서는 사용하지 않음 (alpha_fusion=0.0 in training)
    alpha_fusion = 0.1

    # System
    device = "cuda"  # or "cpu"
    num_workers = 8  # Parallel data loading (increased from 0 for speed)
    pin_memory = True


class FullFeatureConfig(BaseConfig):
    """
    FULL 버전: 5가지 feature 모두 사용 + 3-band split
    - Position (x, y)
    - Velocity (Δx, Δy)
    - Acceleration (Δ²x, Δ²y)
    - Angle (θ)
    - Angle-rate (Δθ)

    총 8 features per landmark
    """

    feature_mode = "full"

    # Feature configuration
    use_velocity = True
    use_acceleration = True
    use_angle = True
    use_angle_rate = True

    # Model input
    K_landmarks = 40  # lips only
    F_dim = 8  # features per landmark
    C_in_per_band = K_landmarks * F_dim  # = 320 channels per band

    # Save directory
    save_dir = "runs/bandvae_full"

    @staticmethod
    def get_feature_names():
        return ["position", "velocity", "acceleration", "angle", "angle_rate"]

    def __repr__(self):
        return (
            f"FullFeatureConfig (Band-Split VAE)\\n"
            f"  features=5 (position, velocity, acceleration, angle, angle_rate)\\n"
            f"  C_in_per_band={self.C_in_per_band} (40 landmarks × 8 features)\\n"
            f"  bands=3 (LF/BP/HF)\\n"
            f"  total_params_per_band≈60K × 3 = 180K\\n"
        )


class SimpleFeatureConfig(BaseConfig):
    """
    SIMPLE 버전: 기본 feature만 사용 + 3-band split
    - Position (x, y)
    - Velocity (Δx, Δy)

    총 4 features per landmark
    """

    feature_mode = "simple"

    # Feature configuration
    use_velocity = True
    use_acceleration = False
    use_angle = False
    use_angle_rate = False

    # Model input
    K_landmarks = 40  # lips only
    F_dim = 4  # features per landmark
    C_in_per_band = K_landmarks * F_dim  # = 160 channels per band

    # Save directory
    save_dir = "runs/bandvae_simple"

    @staticmethod
    def get_feature_names():
        return ["position", "velocity"]

    def __repr__(self):
        return (
            f"SimpleFeatureConfig (Band-Split VAE)\\n"
            f"  features=2 (position, velocity)\\n"
            f"  C_in_per_band={self.C_in_per_band} (40 landmarks × 4 features)\\n"
            f"  bands=3 (LF/BP/HF)\\n"
            f"  total_params_per_band≈60K × 3 = 180K\\n"
        )


# Config factory
def get_config(mode="simple"):
    """
    Get configuration by mode

    Args:
        mode: "full" or "simple"

    Returns:
        Config instance
    """
    if mode == "full":
        return FullFeatureConfig()
    elif mode == "simple":
        return SimpleFeatureConfig()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'full' or 'simple'")


if __name__ == "__main__":
    print("=" * 60)
    print("Band-Split VAE Configuration Comparison")
    print("=" * 60)

    print("\\n【FULL Feature Version】")
    cfg_full = get_config("full")
    print(cfg_full)
    print(f"  Filter: {cfg_full.fc_low}Hz ~ {cfg_full.fc_high}Hz")

    print("\\n【SIMPLE Feature Version】")
    cfg_simple = get_config("simple")
    print(cfg_simple)
    print(f"  Filter: {cfg_simple.fc_low}Hz ~ {cfg_simple.fc_high}Hz")

    print("\\n" + "=" * 60)
