"""
Configuration for Liveness Detection Models

2가지 Feature 버전:
1. FULL: position + velocity + acceleration + angle + angle-rate (8 features)
2. SIMPLE: position + velocity only (4 features)
"""


class BaseConfig:
    """Base configuration"""
    # Data paths
    data_dir = "/home/elicer/liveness_detection/model1/processed_live"
    save_dir = "runs/vae"

    # Data processing
    T_fixed = 150  # 5초 @ 30fps
    fps = 30

    # Model architecture
    C_h = 48  # Hidden channels
    C_z = 12  # Latent dimension
    dilations = [1, 2, 4]

    # Training
    batch_size = 64
    lr = 1e-3
    epochs = 20
    beta = 1.0  # KL weight
    val_split = 0.1

    # System
    device = "cuda"  # or "cpu"
    num_workers = 4
    pin_memory = True
    save_every = 5


class FullFeatureConfig(BaseConfig):
    """
    FULL 버전: 5가지 feature 모두 사용
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
    C_in = K_landmarks * F_dim  # = 320 channels

    # Save directory
    save_dir = "runs/vae_full"

    @staticmethod
    def get_feature_names():
        return ['position', 'velocity', 'acceleration', 'angle', 'angle_rate']

    def __repr__(self):
        return (
            f"FullFeatureConfig(\n"
            f"  features=5 (position, velocity, acceleration, angle, angle_rate)\n"
            f"  C_in={self.C_in} (40 landmarks × 8 features)\n"
            f"  total_vars_per_sample={self.C_in * self.T_fixed}\n"
            f")"
        )


class SimpleFeatureConfig(BaseConfig):
    """
    SIMPLE 버전: 기본 feature만 사용
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
    C_in = K_landmarks * F_dim  # = 160 channels

    # Save directory
    save_dir = "runs/vae_simple"

    @staticmethod
    def get_feature_names():
        return ['position', 'velocity']

    def __repr__(self):
        return (
            f"SimpleFeatureConfig(\n"
            f"  features=2 (position, velocity)\n"
            f"  C_in={self.C_in} (40 landmarks × 4 features)\n"
            f"  total_vars_per_sample={self.C_in * self.T_fixed}\n"
            f")"
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
    print("="*60)
    print("Configuration Comparison")
    print("="*60)

    print("\n【FULL Feature Version】")
    cfg_full = get_config("full")
    print(cfg_full)
    print(f"  Expected parameters: ~75,000")
    print(f"  Params/InputVars ratio: ~1.56")

    print("\n【SIMPLE Feature Version】")
    cfg_simple = get_config("simple")
    print(cfg_simple)
    print(f"  Expected parameters: ~60,000")
    print(f"  Params/InputVars ratio: ~2.5")

    print("\n" + "="*60)
