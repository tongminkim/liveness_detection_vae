"""
Frequency Decomposition Visualization for Presentation

Butterworth 필터를 통한 주파수 대역 분리 시각화
- Input: Original landmark signal
- Output: 3개 주파수 대역 (LF, BP, HF)로 분리된 신호
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


class ButterworthFilterBank:
    """Butterworth IIR filter로 신호를 3개 주파수 대역으로 분리"""

    def __init__(self, fps=30, fc_low=2.0, fc_high=8.0, order=4):
        """
        Args:
            fps: frames per second
            fc_low: low cutoff frequency (Hz)
            fc_high: high cutoff frequency (Hz)
            order: filter order
        """
        self.fps = fps
        self.fc_low = fc_low
        self.fc_high = fc_high
        nyquist = fps / 2.0

        # Normalize cutoff frequencies
        wn_low = fc_low / nyquist
        wn_high = fc_high / nyquist

        # Design Butterworth filters
        self.sos_lf = signal.butter(order, wn_low, btype='lowpass', output='sos')
        self.sos_bp = signal.butter(order, [wn_low, wn_high], btype='bandpass', output='sos')
        self.sos_hf = signal.butter(order, wn_high, btype='highpass', output='sos')

    def apply(self, x):
        """
        Apply Butterworth filter bank

        Args:
            x: [T, K, 2] position landmarks

        Returns:
            x_lf, x_bp, x_hf: each [T, K, 2]
        """
        T, K, D = x.shape

        # Reshape to [K*D, T] for vectorized filtering
        x_flat = x.transpose(1, 2, 0).reshape(K * D, T)  # [K*D, T]

        # Apply Butterworth filters
        x_lf_flat = signal.sosfiltfilt(self.sos_lf, x_flat, axis=1)
        x_bp_flat = signal.sosfiltfilt(self.sos_bp, x_flat, axis=1)
        x_hf_flat = signal.sosfiltfilt(self.sos_hf, x_flat, axis=1)

        # Reshape back to [T, K, 2]
        x_lf = x_lf_flat.reshape(K, D, T).transpose(2, 0, 1)
        x_bp = x_bp_flat.reshape(K, D, T).transpose(2, 0, 1)
        x_hf = x_hf_flat.reshape(K, D, T).transpose(2, 0, 1)

        return x_lf, x_bp, x_hf


def load_sample_data(data_path, T_max=300):
    """
    Load landmark data from .npz file

    Args:
        data_path: path to .npz file
        T_max: maximum temporal length

    Returns:
        landmarks: [T, 40, 2] normalized landmarks
    """
    data = np.load(data_path)

    lips_outer = data['lips_outer']  # [T, 21, 2]
    lips_inner = data['lips_inner']  # [T, 21, 2]

    # Concatenate and remove duplicates -> 40 landmarks
    lips = np.concatenate([
        lips_outer[:, :-1, :],  # [T, 20, 2]
        lips_inner[:, :-1, :]   # [T, 20, 2]
    ], axis=1)  # [T, 40, 2]

    size = data['size']
    h, w = size

    # Normalize by image size
    lips_norm = lips.copy()
    lips_norm[..., 0] /= (w + 1e-8)
    lips_norm[..., 1] /= (h + 1e-8)

    # Crop to T_max frames
    T = lips_norm.shape[0]
    if T > T_max:
        start = (T - T_max) // 2
        lips_norm = lips_norm[start:start + T_max]

    return lips_norm


def visualize_frequency_decomposition(
    landmarks,
    fps=30,
    fc_low=2.0,
    fc_high=8.0,
    landmark_idx=13,  # Upper lip center
    save_path='frequency_decomposition.png'
):
    """
    Visualize frequency decomposition using Butterworth filters

    Args:
        landmarks: [T, K, 2] landmark positions
        fps: frames per second
        fc_low: low cutoff frequency (Hz)
        fc_high: high cutoff frequency (Hz)
        landmark_idx: which landmark to visualize
        save_path: output file path
    """
    # Apply Butterworth filter bank
    filter_bank = ButterworthFilterBank(fps=fps, fc_low=fc_low, fc_high=fc_high)
    lf, bp, hf = filter_bank.apply(landmarks)

    T = landmarks.shape[0]
    time = np.arange(T) / fps

    # Extract specific landmark for visualization
    original = landmarks[:, landmark_idx, :]  # [T, 2]
    lf_signal = lf[:, landmark_idx, :]
    bp_signal = bp[:, landmark_idx, :]
    hf_signal = hf[:, landmark_idx, :]

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Color scheme
    color_original = '#2C3E50'
    color_lf = '#3498DB'  # Blue
    color_bp = '#E74C3C'  # Red
    color_hf = '#2ECC71'  # Green

    # Title
    fig.suptitle('Frequency Decomposition via Butterworth Filters',
                 fontsize=16, fontweight='bold', y=0.98)

    # ========== Left Column: X-coordinate ==========

    # 1. Original signal (X)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, original[:, 0], color=color_original, linewidth=1.5, label='Original')
    ax1.set_ylabel('X Position (normalized)', fontweight='bold')
    ax1.set_title('Input Signal (X-coordinate)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim(time[0], time[-1])

    # 2. Low-frequency (X)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, original[:, 0], color=color_original, linewidth=0.5, alpha=0.3, label='Original')
    ax2.plot(time, lf_signal[:, 0], color=color_lf, linewidth=2, label=f'Low-Freq (0-{fc_low}Hz)')
    ax2.set_ylabel('X Position', fontweight='bold')
    ax2.set_title(f'Low-Frequency Band (0-{fc_low}Hz): Head Pose & Global Motion',
                  fontweight='bold', color=color_lf)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim(time[0], time[-1])

    # 3. Band-pass (X)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time, original[:, 0], color=color_original, linewidth=0.5, alpha=0.3, label='Original')
    ax3.plot(time, bp_signal[:, 0], color=color_bp, linewidth=2, label=f'Band-Pass ({fc_low}-{fc_high}Hz)')
    ax3.set_ylabel('X Position', fontweight='bold')
    ax3.set_title(f'Band-Pass ({fc_low}-{fc_high}Hz): Speech-Related Motion',
                  fontweight='bold', color=color_bp)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    ax3.set_xlim(time[0], time[-1])

    # 4. High-frequency (X)
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(time, original[:, 0], color=color_original, linewidth=0.5, alpha=0.3, label='Original')
    ax4.plot(time, hf_signal[:, 0], color=color_hf, linewidth=2, label=f'High-Freq ({fc_high}-15Hz)')
    ax4.set_ylabel('X Position', fontweight='bold')
    ax4.set_title(f'High-Frequency Band ({fc_high}Hz+): Micro-Expressions & Noise',
                  fontweight='bold', color=color_hf)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    ax4.set_xlim(time[0], time[-1])

    # 5. Reconstruction (X)
    ax5 = fig.add_subplot(gs[4, 0])
    reconstructed_x = lf_signal[:, 0] + bp_signal[:, 0] + hf_signal[:, 0]
    ax5.plot(time, original[:, 0], color=color_original, linewidth=1.5, label='Original', alpha=0.7)
    ax5.plot(time, reconstructed_x, color='#9B59B6', linewidth=1.5, linestyle='--',
             label='Reconstructed (LF+BP+HF)')
    ax5.set_xlabel('Time (seconds)', fontweight='bold')
    ax5.set_ylabel('X Position', fontweight='bold')
    ax5.set_title('Reconstruction: Sum of All Bands', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right')
    ax5.set_xlim(time[0], time[-1])

    # ========== Right Column: Y-coordinate ==========

    # 1. Original signal (Y)
    ax1r = fig.add_subplot(gs[0, 1])
    ax1r.plot(time, original[:, 1], color=color_original, linewidth=1.5, label='Original')
    ax1r.set_ylabel('Y Position (normalized)', fontweight='bold')
    ax1r.set_title('Input Signal (Y-coordinate)', fontweight='bold')
    ax1r.grid(True, alpha=0.3)
    ax1r.legend(loc='upper right')
    ax1r.set_xlim(time[0], time[-1])

    # 2. Low-frequency (Y)
    ax2r = fig.add_subplot(gs[1, 1])
    ax2r.plot(time, original[:, 1], color=color_original, linewidth=0.5, alpha=0.3, label='Original')
    ax2r.plot(time, lf_signal[:, 1], color=color_lf, linewidth=2, label=f'Low-Freq (0-{fc_low}Hz)')
    ax2r.set_ylabel('Y Position', fontweight='bold')
    ax2r.set_title(f'Low-Frequency Band (0-{fc_low}Hz): Head Pose & Global Motion',
                   fontweight='bold', color=color_lf)
    ax2r.grid(True, alpha=0.3)
    ax2r.legend(loc='upper right')
    ax2r.set_xlim(time[0], time[-1])

    # 3. Band-pass (Y)
    ax3r = fig.add_subplot(gs[2, 1])
    ax3r.plot(time, original[:, 1], color=color_original, linewidth=0.5, alpha=0.3, label='Original')
    ax3r.plot(time, bp_signal[:, 1], color=color_bp, linewidth=2, label=f'Band-Pass ({fc_low}-{fc_high}Hz)')
    ax3r.set_ylabel('Y Position', fontweight='bold')
    ax3r.set_title(f'Band-Pass ({fc_low}-{fc_high}Hz): Speech-Related Motion',
                   fontweight='bold', color=color_bp)
    ax3r.grid(True, alpha=0.3)
    ax3r.legend(loc='upper right')
    ax3r.set_xlim(time[0], time[-1])

    # 4. High-frequency (Y)
    ax4r = fig.add_subplot(gs[3, 1])
    ax4r.plot(time, original[:, 1], color=color_original, linewidth=0.5, alpha=0.3, label='Original')
    ax4r.plot(time, hf_signal[:, 1], color=color_hf, linewidth=2, label=f'High-Freq ({fc_high}-15Hz)')
    ax4r.set_ylabel('Y Position', fontweight='bold')
    ax4r.set_title(f'High-Frequency Band ({fc_high}Hz+): Micro-Expressions & Noise',
                   fontweight='bold', color=color_hf)
    ax4r.grid(True, alpha=0.3)
    ax4r.legend(loc='upper right')
    ax4r.set_xlim(time[0], time[-1])

    # 5. Reconstruction (Y)
    ax5r = fig.add_subplot(gs[4, 1])
    reconstructed_y = lf_signal[:, 1] + bp_signal[:, 1] + hf_signal[:, 1]
    ax5r.plot(time, original[:, 1], color=color_original, linewidth=1.5, label='Original', alpha=0.7)
    ax5r.plot(time, reconstructed_y, color='#9B59B6', linewidth=1.5, linestyle='--',
              label='Reconstructed (LF+BP+HF)')
    ax5r.set_xlabel('Time (seconds)', fontweight='bold')
    ax5r.set_ylabel('Y Position', fontweight='bold')
    ax5r.set_title('Reconstruction: Sum of All Bands', fontweight='bold')
    ax5r.grid(True, alpha=0.3)
    ax5r.legend(loc='upper right')
    ax5r.set_xlim(time[0], time[-1])

    # Add method description
    fig.text(0.5, 0.01,
             f'Method: 4th-order Butterworth IIR Filters | '
             f'Landmark: Upper Lip Center (#{landmark_idx}) | '
             f'FPS: {fps} | Cutoffs: {fc_low}Hz, {fc_high}Hz',
             ha='center', fontsize=9, style='italic', color='gray')

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    plt.close()


def visualize_compact_decomposition(
    landmarks,
    fps=30,
    fc_low=2.0,
    fc_high=8.0,
    landmark_idx=13,
    save_path='frequency_decomposition_compact.png'
):
    """
    Create a compact version for presentation slides
    """
    # Apply Butterworth filter bank
    filter_bank = ButterworthFilterBank(fps=fps, fc_low=fc_low, fc_high=fc_high)
    lf, bp, hf = filter_bank.apply(landmarks)

    T = landmarks.shape[0]
    time = np.arange(T) / fps

    # Extract specific landmark (Y-coordinate only for clarity)
    original = landmarks[:, landmark_idx, 1]
    lf_signal = lf[:, landmark_idx, 1]
    bp_signal = bp[:, landmark_idx, 1]
    hf_signal = hf[:, landmark_idx, 1]

    # Create compact figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)

    color_original = '#2C3E50'
    color_lf = '#3498DB'
    color_bp = '#E74C3C'
    color_hf = '#2ECC71'

    # 1. Original
    axes[0].plot(time, original, color=color_original, linewidth=2)
    axes[0].set_ylabel('Position', fontweight='bold', fontsize=12)
    axes[0].set_title('Input Signal', fontweight='bold', fontsize=13, pad=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(time[0], time[-1])

    # 2. Low-Frequency
    axes[1].plot(time, lf_signal, color=color_lf, linewidth=2)
    axes[1].set_ylabel('Position', fontweight='bold', fontsize=12)
    axes[1].set_title(f'Low-Frequency (0-{fc_low}Hz): Head Pose & Global Motion',
                      fontweight='bold', fontsize=13, color=color_lf, pad=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(time[0], time[-1])

    # 3. Band-Pass
    axes[2].plot(time, bp_signal, color=color_bp, linewidth=2)
    axes[2].set_ylabel('Position', fontweight='bold', fontsize=12)
    axes[2].set_title(f'Band-Pass ({fc_low}-{fc_high}Hz): Speech-Related Motion',
                      fontweight='bold', fontsize=13, color=color_bp, pad=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(time[0], time[-1])

    # 4. High-Frequency
    axes[3].plot(time, hf_signal, color=color_hf, linewidth=2)
    axes[3].set_ylabel('Position', fontweight='bold', fontsize=12)
    axes[3].set_title(f'High-Frequency ({fc_high}Hz+): Micro-Expressions',
                      fontweight='bold', fontsize=13, color=color_hf, pad=10)
    axes[3].set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(time[0], time[-1])

    plt.suptitle('Frequency Decomposition via Butterworth Filters',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    plt.close()


if __name__ == "__main__":
    # Configuration
    data_dir = "/home/elicer/liveness_detection/model1/20GBprocessed"
    output_dir = "/home/elicer/liveness_detection/model1/paper_figures"

    # Find a sample file
    sample_files = [f for f in os.listdir(data_dir) if f.endswith('.npz') and not f.startswith('._')]
    if not sample_files:
        print("Error: No .npz files found!")
        exit(1)

    sample_path = os.path.join(data_dir, sample_files[0])
    print(f"Using sample: {sample_files[0]}")

    # Load data
    print("Loading landmark data...")
    landmarks = load_sample_data(sample_path, T_max=300)
    print(f"Loaded landmarks shape: {landmarks.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    print("\n=== Generating Visualizations ===")

    print("\n1. Full visualization (both X and Y coordinates)...")
    visualize_frequency_decomposition(
        landmarks,
        fps=30,
        fc_low=2.0,
        fc_high=8.0,
        landmark_idx=13,
        save_path=os.path.join(output_dir, 'frequency_decomposition_full.png')
    )

    print("\n2. Compact visualization (Y-coordinate only)...")
    visualize_compact_decomposition(
        landmarks,
        fps=30,
        fc_low=2.0,
        fc_high=8.0,
        landmark_idx=13,
        save_path=os.path.join(output_dir, 'frequency_decomposition_compact.png')
    )

    print("\n✓ All visualizations generated successfully!")
    print(f"\nOutput directory: {output_dir}")
