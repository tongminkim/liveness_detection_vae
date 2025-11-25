"""
Real vs Fake Frequency Decomposition Visualization

Real 영상과 Fake 영상의 주파수 분리를 비교하여 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
import os
import glob

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


class ButterworthFilterBank:
    """Butterworth IIR filter로 신호를 3개 주파수 대역으로 분리"""

    def __init__(self, fps=30, fc_low=2.0, fc_high=8.0, order=4):
        self.fps = fps
        self.fc_low = fc_low
        self.fc_high = fc_high
        nyquist = fps / 2.0

        wn_low = fc_low / nyquist
        wn_high = fc_high / nyquist

        self.sos_lf = signal.butter(order, wn_low, btype='lowpass', output='sos')
        self.sos_bp = signal.butter(order, [wn_low, wn_high], btype='bandpass', output='sos')
        self.sos_hf = signal.butter(order, wn_high, btype='highpass', output='sos')

    def apply(self, x):
        T, K, D = x.shape
        x_flat = x.transpose(1, 2, 0).reshape(K * D, T)

        x_lf_flat = signal.sosfiltfilt(self.sos_lf, x_flat, axis=1)
        x_bp_flat = signal.sosfiltfilt(self.sos_bp, x_flat, axis=1)
        x_hf_flat = signal.sosfiltfilt(self.sos_hf, x_flat, axis=1)

        x_lf = x_lf_flat.reshape(K, D, T).transpose(2, 0, 1)
        x_bp = x_bp_flat.reshape(K, D, T).transpose(2, 0, 1)
        x_hf = x_hf_flat.reshape(K, D, T).transpose(2, 0, 1)

        return x_lf, x_bp, x_hf


def load_sample_data(data_path, T_max=300):
    """Load landmark data from .npz file"""
    data = np.load(data_path)

    lips_outer = data['lips_outer']
    lips_inner = data['lips_inner']

    lips = np.concatenate([
        lips_outer[:, :-1, :],
        lips_inner[:, :-1, :]
    ], axis=1)

    size = data['size']
    h, w = size

    lips_norm = lips.copy()
    lips_norm[..., 0] /= (w + 1e-8)
    lips_norm[..., 1] /= (h + 1e-8)

    T = lips_norm.shape[0]
    if T > T_max:
        start = (T - T_max) // 2
        lips_norm = lips_norm[start:start + T_max]

    return lips_norm


def visualize_real_vs_fake_comparison(
    real_landmarks,
    fake_landmarks,
    fps=30,
    fc_low=2.0,
    fc_high=8.0,
    landmark_idx=13,
    save_path='real_vs_fake_decomposition.png'
):
    """
    Real vs Fake 주파수 분리 비교 시각화
    """
    # Apply filters
    filter_bank = ButterworthFilterBank(fps=fps, fc_low=fc_low, fc_high=fc_high)

    # Real
    real_lf, real_bp, real_hf = filter_bank.apply(real_landmarks)
    # Fake
    fake_lf, fake_bp, fake_hf = filter_bank.apply(fake_landmarks)

    T = min(real_landmarks.shape[0], fake_landmarks.shape[0])
    time = np.arange(T) / fps

    # Extract specific landmark (Y-coordinate)
    real_original = real_landmarks[:T, landmark_idx, 1]
    real_lf_sig = real_lf[:T, landmark_idx, 1]
    real_bp_sig = real_bp[:T, landmark_idx, 1]
    real_hf_sig = real_hf[:T, landmark_idx, 1]

    fake_original = fake_landmarks[:T, landmark_idx, 1]
    fake_lf_sig = fake_lf[:T, landmark_idx, 1]
    fake_bp_sig = fake_bp[:T, landmark_idx, 1]
    fake_hf_sig = fake_hf[:T, landmark_idx, 1]

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(18, 10), sharex=True)

    # Colors
    color_real = '#2C3E50'
    color_fake = '#E74C3C'
    color_lf = '#3498DB'
    color_bp = '#E67E22'
    color_hf = '#2ECC71'

    # ========== Left Column: REAL ==========

    # 1. Real - Original
    axes[0, 0].plot(time, real_original, color=color_real, linewidth=2)
    axes[0, 0].set_ylabel('Position', fontweight='bold', fontsize=11)
    axes[0, 0].set_title('REAL: Input Signal', fontweight='bold', fontsize=14,
                          color=color_real, pad=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(time[0], time[-1])

    # 2. Real - Low-Frequency
    axes[1, 0].plot(time, real_lf_sig, color=color_lf, linewidth=2)
    axes[1, 0].set_ylabel('Position', fontweight='bold', fontsize=11)
    axes[1, 0].set_title(f'Low-Freq (0-{fc_low}Hz): Head Pose',
                          fontweight='bold', fontsize=12, color=color_lf, pad=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(time[0], time[-1])

    # 3. Real - Band-Pass
    axes[2, 0].plot(time, real_bp_sig, color=color_bp, linewidth=2)
    axes[2, 0].set_ylabel('Position', fontweight='bold', fontsize=11)
    axes[2, 0].set_title(f'Band-Pass ({fc_low}-{fc_high}Hz): Speech Motion',
                          fontweight='bold', fontsize=12, color=color_bp, pad=8)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim(time[0], time[-1])

    # 4. Real - High-Frequency
    axes[3, 0].plot(time, real_hf_sig, color=color_hf, linewidth=2)
    axes[3, 0].set_ylabel('Position', fontweight='bold', fontsize=11)
    axes[3, 0].set_title(f'High-Freq ({fc_high}Hz+): Micro-Expressions',
                          fontweight='bold', fontsize=12, color=color_hf, pad=8)
    axes[3, 0].set_xlabel('Time (seconds)', fontweight='bold', fontsize=11)
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].set_xlim(time[0], time[-1])

    # ========== Right Column: FAKE ==========

    # 1. Fake - Original
    axes[0, 1].plot(time, fake_original, color=color_fake, linewidth=2)
    axes[0, 1].set_ylabel('Position', fontweight='bold', fontsize=11)
    axes[0, 1].set_title('FAKE: Input Signal', fontweight='bold', fontsize=14,
                          color=color_fake, pad=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(time[0], time[-1])

    # 2. Fake - Low-Frequency
    axes[1, 1].plot(time, fake_lf_sig, color=color_lf, linewidth=2)
    axes[1, 1].set_ylabel('Position', fontweight='bold', fontsize=11)
    axes[1, 1].set_title(f'Low-Freq (0-{fc_low}Hz): Head Pose',
                          fontweight='bold', fontsize=12, color=color_lf, pad=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(time[0], time[-1])

    # 3. Fake - Band-Pass
    axes[2, 1].plot(time, fake_bp_sig, color=color_bp, linewidth=2)
    axes[2, 1].set_ylabel('Position', fontweight='bold', fontsize=11)
    axes[2, 1].set_title(f'Band-Pass ({fc_low}-{fc_high}Hz): Speech Motion',
                          fontweight='bold', fontsize=12, color=color_bp, pad=8)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlim(time[0], time[-1])

    # 4. Fake - High-Frequency
    axes[3, 1].plot(time, fake_hf_sig, color=color_hf, linewidth=2)
    axes[3, 1].set_ylabel('Position', fontweight='bold', fontsize=11)
    axes[3, 1].set_title(f'High-Freq ({fc_high}Hz+): Micro-Expressions',
                          fontweight='bold', fontsize=12, color=color_hf, pad=8)
    axes[3, 1].set_xlabel('Time (seconds)', fontweight='bold', fontsize=11)
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].set_xlim(time[0], time[-1])

    # Main title
    fig.suptitle('Frequency Decomposition: REAL vs FAKE Comparison',
                 fontsize=18, fontweight='bold', y=0.995)

    # Footer
    fig.text(0.5, 0.01,
             f'Method: 4th-order Butterworth IIR Filters | '
             f'Landmark: Upper Lip Center (#{landmark_idx}) | '
             f'FPS: {fps} | Cutoffs: {fc_low}Hz, {fc_high}Hz',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    plt.close()


def visualize_overlay_comparison(
    real_landmarks,
    fake_landmarks,
    fps=30,
    fc_low=2.0,
    fc_high=8.0,
    landmark_idx=13,
    save_path='real_vs_fake_overlay.png'
):
    """
    Real과 Fake를 overlay하여 차이를 명확히 보여주는 시각화
    """
    filter_bank = ButterworthFilterBank(fps=fps, fc_low=fc_low, fc_high=fc_high)

    # Real
    real_lf, real_bp, real_hf = filter_bank.apply(real_landmarks)
    # Fake
    fake_lf, fake_bp, fake_hf = filter_bank.apply(fake_landmarks)

    T = min(real_landmarks.shape[0], fake_landmarks.shape[0])
    time = np.arange(T) / fps

    # Extract signals
    real_original = real_landmarks[:T, landmark_idx, 1]
    real_lf_sig = real_lf[:T, landmark_idx, 1]
    real_bp_sig = real_bp[:T, landmark_idx, 1]
    real_hf_sig = real_hf[:T, landmark_idx, 1]

    fake_original = fake_landmarks[:T, landmark_idx, 1]
    fake_lf_sig = fake_lf[:T, landmark_idx, 1]
    fake_bp_sig = fake_bp[:T, landmark_idx, 1]
    fake_hf_sig = fake_hf[:T, landmark_idx, 1]

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    color_real = '#3498DB'  # Blue for real
    color_fake = '#E74C3C'  # Red for fake

    # 1. Original Signals
    axes[0].plot(time, real_original, color=color_real, linewidth=2,
                 label='Real', alpha=0.8)
    axes[0].plot(time, fake_original, color=color_fake, linewidth=2,
                 label='Fake', alpha=0.8, linestyle='--')
    axes[0].set_ylabel('Position', fontweight='bold', fontsize=12)
    axes[0].set_title('Input Signals: Real vs Fake', fontweight='bold', fontsize=14, pad=10)
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(time[0], time[-1])

    # 2. Low-Frequency
    axes[1].plot(time, real_lf_sig, color=color_real, linewidth=2.5,
                 label='Real', alpha=0.8)
    axes[1].plot(time, fake_lf_sig, color=color_fake, linewidth=2.5,
                 label='Fake', alpha=0.8, linestyle='--')
    axes[1].set_ylabel('Position', fontweight='bold', fontsize=12)
    axes[1].set_title(f'Low-Frequency (0-{fc_low}Hz): Head Pose & Global Motion',
                      fontweight='bold', fontsize=13, pad=10)
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(time[0], time[-1])

    # 3. Band-Pass
    axes[2].plot(time, real_bp_sig, color=color_real, linewidth=2.5,
                 label='Real', alpha=0.8)
    axes[2].plot(time, fake_bp_sig, color=color_fake, linewidth=2.5,
                 label='Fake', alpha=0.8, linestyle='--')
    axes[2].set_ylabel('Position', fontweight='bold', fontsize=12)
    axes[2].set_title(f'Band-Pass ({fc_low}-{fc_high}Hz): Speech-Related Motion',
                      fontweight='bold', fontsize=13, pad=10)
    axes[2].legend(loc='upper right', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(time[0], time[-1])

    # 4. High-Frequency
    axes[3].plot(time, real_hf_sig, color=color_real, linewidth=2,
                 label='Real', alpha=0.8)
    axes[3].plot(time, fake_hf_sig, color=color_fake, linewidth=2,
                 label='Fake', alpha=0.8, linestyle='--')
    axes[3].set_ylabel('Position', fontweight='bold', fontsize=12)
    axes[3].set_title(f'High-Frequency ({fc_high}Hz+): Micro-Expressions',
                      fontweight='bold', fontsize=13, pad=10)
    axes[3].set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    axes[3].legend(loc='upper right', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(time[0], time[-1])

    fig.suptitle('Real vs Fake: Frequency Band Comparison (Overlay)',
                 fontsize=18, fontweight='bold', y=0.995)

    fig.text(0.5, 0.01,
             f'Blue (solid) = Real | Red (dashed) = Fake | '
             f'4th-order Butterworth Filters | Cutoffs: {fc_low}Hz, {fc_high}Hz',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    plt.close()


if __name__ == "__main__":
    # Paths
    real_dir = "/home/elicer/liveness_detection/model1/test_balanced_npz/real"
    fake_dir = "/home/elicer/liveness_detection/model1/test_balanced_npz/fake"
    output_dir = "/home/elicer/liveness_detection/model1/paper_figures"

    # Find real samples
    real_files = []
    for subdir in ['원본1', '원본2']:
        real_path = os.path.join(real_dir, subdir)
        if os.path.exists(real_path):
            real_files.extend(glob.glob(os.path.join(real_path, "*.npz")))

    # Find fake samples (audio-driven)
    fake_files = glob.glob(os.path.join(fake_dir, "audio-driven", "*.npz"))

    if not real_files:
        print("Error: No real files found!")
        exit(1)
    if not fake_files:
        print("Error: No fake files found!")
        exit(1)

    # Select samples
    real_sample = real_files[0]
    fake_sample = fake_files[0]

    print(f"Real sample: {os.path.basename(real_sample)}")
    print(f"Fake sample: {os.path.basename(fake_sample)}")

    # Load data
    print("\nLoading landmark data...")
    real_landmarks = load_sample_data(real_sample, T_max=300)
    fake_landmarks = load_sample_data(fake_sample, T_max=300)

    print(f"Real landmarks shape: {real_landmarks.shape}")
    print(f"Fake landmarks shape: {fake_landmarks.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    print("\n=== Generating Visualizations ===")

    print("\n1. Side-by-side comparison...")
    visualize_real_vs_fake_comparison(
        real_landmarks,
        fake_landmarks,
        fps=30,
        fc_low=2.0,
        fc_high=8.0,
        landmark_idx=13,
        save_path=os.path.join(output_dir, 'real_vs_fake_sidebyside.png')
    )

    print("\n2. Overlay comparison...")
    visualize_overlay_comparison(
        real_landmarks,
        fake_landmarks,
        fps=30,
        fc_low=2.0,
        fc_high=8.0,
        landmark_idx=13,
        save_path=os.path.join(output_dir, 'real_vs_fake_overlay.png')
    )

    print("\n✓ All visualizations generated successfully!")
    print(f"\nOutput directory: {output_dir}")
