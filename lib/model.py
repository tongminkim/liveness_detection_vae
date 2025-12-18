"""
Frequency-Decoupled Feature-Space VAE (Band-Split Version)

논문 Section 2 구현 (Figure 1):
- 3-band frequency decomposition (LF/BP/HF)
- Per-band feature-space VAE
- Learnable weighted fusion
- Per-band reconstruction + KL loss

논문의 핵심 아이디어:
1. "A lightweight FIR filter bank separates the signal into
    low-, mid-, and high-frequency streams" (Section 2.1)
   → 본 구현: Butterworth IIR filter (병목 해소, 10-100배 빠름)

2. "For each band b ∈ {L, B, H}, the encoder fb takes the
    concatenated feature tensor and outputs (μb, log σ²b)" (Section 2.1)
   → 각 대역별 독립적인 TCN-VAE

3. "A learnable projection-based fusion combines per-band
    reconstructions" (Section 2.1)
   → Learnable weighted fusion (softmax normalized)

4. "Per-band feature-space reconstruction with derivative-aware
    consistency" (Section 2.3)
   → L1 reconstruction + KL divergence per band
"""

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

# ============================================
# Butterworth Filter Bank (논문 Section 2.1)
# ============================================


class ButterworthFilterBank:
    """
    Butterworth IIR filter로 position signal을 3개 주파수 대역으로 분리

    논문 Section 2.1:
    "A lightweight FIR filter bank separates the signal into
     low-, mid-, and high-frequency streams, each modeled by a
     compact encoder-decoder operating directly in the feature space."

    본 구현:
    - FIR filter 대신 Butterworth IIR filter 사용 (10-100배 빠름)
    - 핵심 아이디어(주파수 대역 분리)는 동일하게 유지
    - SOS (second-order sections) format으로 numerical stability 보장

    Frequency Bands:
    - Low-Frequency (LF): 0 ~ fc_low Hz (전체적인 움직임, 대화 리듬)
    - Band-Pass (BP): fc_low ~ fc_high Hz (주요 발화 모션)
    - High-Frequency (HF): fc_high ~ Nyquist Hz (미세한 떨림, artifact)
    """

    def __init__(self, fps=30, fc_low=2.0, fc_high=8.0, order=4):
        """
        Args:
            fps: frames per second
            fc_low: low cutoff frequency (Hz)
            fc_high: high cutoff frequency (Hz)
            order: filter order (4-6 is typical, higher = sharper cutoff)
        """
        self.fps = fps
        nyquist = fps / 2.0

        # Normalize cutoff frequencies
        wn_low = fc_low / nyquist
        wn_high = fc_high / nyquist

        # Design Butterworth filters (SOS format for numerical stability)
        self.sos_lf = signal.butter(order, wn_low, btype="lowpass", output="sos")
        self.sos_bp = signal.butter(
            order, [wn_low, wn_high], btype="bandpass", output="sos"
        )
        self.sos_hf = signal.butter(order, wn_high, btype="highpass", output="sos")

    def apply(self, x):
        """
        Apply Butterworth filter bank (VERY FAST!)

        Args:
            x: [T, K, 2] position landmarks

        Returns:
            x_lf, x_bp, x_hf: each [T, K, 2]
        """
        T, K, D = x.shape

        # Reshape to [K*D, T] for vectorized filtering
        x_flat = x.transpose(1, 2, 0).reshape(K * D, T)  # [K*D, T]

        # Apply Butterworth filters (sosfiltfilt: zero-phase, very fast)
        x_lf_flat = signal.sosfiltfilt(self.sos_lf, x_flat, axis=1)
        x_bp_flat = signal.sosfiltfilt(self.sos_bp, x_flat, axis=1)
        x_hf_flat = signal.sosfiltfilt(self.sos_hf, x_flat, axis=1)

        # Reshape back to [T, K, 2]
        x_lf = x_lf_flat.reshape(K, D, T).transpose(2, 0, 1)
        x_bp = x_bp_flat.reshape(K, D, T).transpose(2, 0, 1)
        x_hf = x_hf_flat.reshape(K, D, T).transpose(2, 0, 1)

        return x_lf, x_bp, x_hf


# ============================================
# TCN Building Blocks (재사용)
# ============================================


class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block with residual connection"""

    def __init__(self, C_in, C_out, dilation=1, kernel_size=3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            C_in, C_out, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.gn = nn.GroupNorm(1, C_out)
        self.act = nn.SiLU()

        # Residual connection
        self.res = nn.Conv1d(C_in, C_out, 1) if C_in != C_out else nn.Identity()

    def forward(self, x):
        y = self.act(self.gn(self.conv(x)))
        return y + self.res(x)


# ============================================
# Single-Band VAE (논문 Section 2.1)
# ============================================


class SingleBandVAE(nn.Module):
    """
    하나의 주파수 대역에 대한 Feature-Space VAE

    논문 Section 2.1:
    "For each band b ∈ {L, B, H}, the encoder fb takes the
     concatenated feature tensor [xb, Δxb, Δ²xb, θb, Δθb] and
     outputs (μb, log σ²b) for a latent zb."

    구조:
    - TCN Encoder: [xb, Δxb, Δ²xb, θb, Δθb] → (μb, σ²b)
    - Reparameterization: zb = μb + σb ⊙ ε, ε ~ N(0, I)
    - TCN Decoder: zb → [x̂b, Δx̂b, Δ²x̂b, θ̂b, Δθ̂b]
    """

    def __init__(self, C_in, C_h=48, C_z=12, dilations=[1, 2, 4]):
        """
        Args:
            C_in: Input channels (K × F_dim)
                - SIMPLE: K=40 × F_dim=4 = 160
                - FULL: K=40 × F_dim=8 = 320
            C_h: Hidden channels (논문: lightweight, 본 구현 48)
            C_z: Latent dimension (논문: compact, 본 구현 12)
            dilations: TCN dilation factors (default [1,2,4])
        """
        super().__init__()

        self.C_in = C_in
        self.C_h = C_h
        self.C_z = C_z

        # Encoder
        self.enc_inp = nn.Conv1d(C_in, C_h, 1)
        enc_blocks = []
        for d in dilations:
            enc_blocks.append(TCNBlock(C_h, C_h, dilation=d))
        self.encoder = nn.Sequential(*enc_blocks)
        self.enc_out = nn.Conv1d(C_h, C_z * 2, 1)

        # Decoder
        self.dec_inp = nn.Conv1d(C_z, C_h, 1)
        dec_blocks = []
        for d in reversed(dilations):
            dec_blocks.append(TCNBlock(C_h, C_h, dilation=d))
        self.decoder = nn.Sequential(*dec_blocks)
        self.dec_out = nn.Conv1d(C_h, C_in, 1)

    def encode(self, x):
        """Encode to latent distribution"""
        h = self.encoder(self.enc_inp(x))
        mu, logvar = torch.chunk(self.enc_out(h), 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from latent"""
        h = self.decoder(self.dec_inp(z))
        return self.dec_out(h)

    def forward(self, x):
        """
        Args:
            x: [B, C_in, T]

        Returns:
            x_hat: [B, C_in, T]
            mu: [B, C_z, T]
            logvar: [B, C_z, T]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# ============================================
# Band-Split VAE (3-band) - 논문의 핵심 아키텍처
# ============================================


class BandSplitVAE(nn.Module):
    """
    Frequency-Decoupled Feature-Space VAE (논문 Figure 1)

    논문 Section 2.1:
    "We decompose the temporal lip landmark signal into low-, mid-,
     and high-frequency bands, and for each band we train a compact
     encoder-decoder that operates on [xb, Δxb, Δ²xb, θb, Δθb] with
     KL regularization and derivative-consistency."

    핵심 구조:
    1. 3개의 독립적인 VAE (LF/BP/HF)
       - 각 VAE는 자신의 주파수 대역만 학습
       - 병렬 처리, 독립적인 loss

    2. Learnable Weighted Fusion (Section 2.1)
       - "A learnable projection-based fusion combines per-band
          reconstructions"
       - Softmax normalized weights: W = softmax([w_L, w_B, w_H])
       - x̂_fused = Σ_b W_b · x̂_b

    3. Per-Band Loss (Section 2.3)
       - Reconstruction: L_rec = Σ_b ‖Λ_b ⊙ (F_b - F̂_b)‖₁
       - KL divergence: L_KL = Σ_b β_b · KL(q_b(z_b|F_b) ‖ p(z))
    """

    def __init__(self, C_in_per_band, C_h=48, C_z=12, dilations=[1, 2, 4]):
        """
        Args:
            C_in_per_band: Input channels per band (K × F_dim)
                - SIMPLE: K=40 × F_dim=4 = 160
                - FULL: K=40 × F_dim=8 = 320
            C_h: Hidden channels (default 48)
            C_z: Latent dimension (default 12)
            dilations: TCN dilation factors (default [1,2,4])
        """
        super().__init__()

        self.C_in_per_band = C_in_per_band

        # 3 independent VAEs
        self.vae_lf = SingleBandVAE(C_in_per_band, C_h, C_z, dilations)
        self.vae_bp = SingleBandVAE(C_in_per_band, C_h, C_z, dilations)
        self.vae_hf = SingleBandVAE(C_in_per_band, C_h, C_z, dilations)

        # Learnable fusion weights (initialized to 1/3)
        self.register_parameter("fusion_weights", nn.Parameter(torch.ones(3) / 3.0))

    def forward(self, x_lf, x_bp, x_hf):
        """
        Args:
            x_lf: [B, C_in, T] - Low-frequency band features
            x_bp: [B, C_in, T] - Band-pass features
            x_hf: [B, C_in, T] - High-frequency features

        Returns:
            recons: dict with per-band reconstructions
            mus: dict with per-band means
            logvars: dict with per-band logvars
            fused: [B, C_in, T] - Weighted fusion
        """
        # Forward through each VAE
        x_hat_lf, mu_lf, logvar_lf = self.vae_lf(x_lf)
        x_hat_bp, mu_bp, logvar_bp = self.vae_bp(x_bp)
        x_hat_hf, mu_hf, logvar_hf = self.vae_hf(x_hf)

        # Softmax normalization of fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)

        # Weighted fusion
        x_hat_fused = (
            weights[0] * x_hat_lf + weights[1] * x_hat_bp + weights[2] * x_hat_hf
        )

        # Pack results
        recons = {"lf": x_hat_lf, "bp": x_hat_bp, "hf": x_hat_hf}

        mus = {"lf": mu_lf, "bp": mu_bp, "hf": mu_hf}

        logvars = {"lf": logvar_lf, "bp": logvar_bp, "hf": logvar_hf}

        return recons, mus, logvars, x_hat_fused


# ============================================
# Loss Functions (논문 Section 2.3)
# ============================================


def kl_divergence(mu, logvar):
    """
    KL divergence for single band

    논문 Section 2.3:
    L_KL = Σ_b β_b · KL(q_b(z_b|F_b) ‖ p(z))
    where p(z) = N(0, I)
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def band_split_vae_loss(
    recons,
    mus,
    logvars,
    targets,
    x_hat_fused,
    x_target_full,
    betas={"lf": 1.0, "bp": 1.0, "hf": 1.0},
    alpha_fusion=0.1,
):
    """
    Band-Split VAE Loss (논문 Section 2.3)

    논문의 Loss 구조:
    L = L_rec + L_KL + L_mix + L_decor

    Per-band feature-space reconstruction:
    L_rec = Σ_b ‖Λ_b ⊙ (F_b - F̂_b)‖₁

    KL regularization:
    L_KL = Σ_b β_b · KL(q_b(z_b|F_b) ‖ p(z))

    본 구현에서는 L_mix, L_decor는 사용하지 않음
    (논문에서도 "small fusion penalty"로 optional)

    Args:
        recons: dict with 'lf', 'bp', 'hf' reconstructions
        mus: dict with 'lf', 'bp', 'hf' means
        logvars: dict with 'lf', 'bp', 'hf' logvars
        targets: dict with 'lf', 'bp', 'hf' target features
        x_hat_fused: [B, C, T] fused reconstruction
        x_target_full: [B, C, T] full-band target (for fusion penalty)
        betas: dict with per-band KL weights (β_b)
        alpha_fusion: fusion penalty weight (본 구현에서는 0.0 사용)

    Returns:
        total_loss, loss_dict
    """
    loss_dict = {}
    total_loss = 0.0

    # Per-band losses
    for band in ["lf", "bp", "hf"]:
        # Reconstruction loss (L1)
        recon_loss = F.l1_loss(recons[band], targets[band])

        # KL divergence
        kl_loss = kl_divergence(mus[band], logvars[band])

        # Band loss
        band_loss = recon_loss + betas[band] * kl_loss

        loss_dict[f"recon_{band}"] = recon_loss.item()
        loss_dict[f"kl_{band}"] = kl_loss.item()
        loss_dict[f"total_{band}"] = band_loss.item()

        total_loss += band_loss

    # Fusion penalty (optional)
    if alpha_fusion > 0 and x_target_full is not None:
        fusion_loss = F.l1_loss(x_hat_fused, x_target_full)
        loss_dict["fusion"] = fusion_loss.item()
        total_loss += alpha_fusion * fusion_loss

    loss_dict["total"] = total_loss.item()

    return total_loss, loss_dict


# ============================================
# Anomaly Scoring (논문 Section 2.4)
# ============================================


@torch.no_grad()
def compute_band_split_anomaly_score(
    model, x_lf, x_bp, x_hf, betas={"lf": 1.0, "bp": 1.0, "hf": 1.0}
):
    """
    Compute anomaly score for band-split VAE

    논문 Section 2.4:
    "We score using a compact feature-space norm and KL:
     S_ano = Σ_b [α·‖Λ_b ⊙ (F_b - F̂_b)‖₁ + β·KL_b]"

    핵심 아이디어:
    - Live 데이터로만 학습 (one-class learning)
    - Fake는 smoother, low-variance motion → 낮은 reconstruction error
    - 논문 결과: Fake (0.208±0.07) < Live (0.319±0.10)
    - 전략: 비정상적으로 낮은 reconstruction = spoof
      (rec < τ₂₀ ⇒ fake, where τ₂₀ is 20th percentile of live)

    Args:
        model: BandSplitVAE model
        x_lf, x_bp, x_hf: Input features per band
        betas: KL weights per band (β_b)

    Returns:
        score: Anomaly score (scalar or [B])
            - Higher score = more anomalous
            - BUT in practice, LOWER score may indicate fake
              (due to smoother motion)
    """
    model.eval()

    # Add batch dimension if needed
    if x_lf.dim() == 2:
        x_lf = x_lf.unsqueeze(0)
        x_bp = x_bp.unsqueeze(0)
        x_hf = x_hf.unsqueeze(0)

    # Forward
    recons, mus, logvars, _ = model(x_lf, x_bp, x_hf)

    score = 0.0

    # Per-band scoring
    for band, x_in in zip(["lf", "bp", "hf"], [x_lf, x_bp, x_hf]):
        # Reconstruction error
        recon = F.l1_loss(recons[band], x_in, reduction="none").mean(dim=[1, 2])

        # KL divergence
        kl = (-0.5 * (1 + logvars[band] - mus[band].pow(2) - logvars[band].exp())).mean(
            dim=[1, 2]
        )

        # Weighted sum
        score += recon + betas[band] * kl

    return score.squeeze() if score.size(0) == 1 else score


# ============================================
# Helpers for notebooks
# ============================================


def _has_npz(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*.npz"))


def _prepare_dataset(name: str, drive_root: Path, local_data: Path) -> Optional[Path]:
    drive_zip = drive_root / f"{name}.zip"
    local_zip = local_data / f"{name}.zip"
    extract_dir = local_data / name
    nested_dir = extract_dir / name

    if drive_zip.exists():
        if (
            not local_zip.exists()
            or drive_zip.stat().st_mtime > local_zip.stat().st_mtime
        ):
            shutil.copy2(drive_zip, local_zip)
            print(f"Copied {drive_zip.name} from Drive to local cache.")

    if not _has_npz(extract_dir) and not _has_npz(nested_dir) and local_zip.exists():
        shutil.unpack_archive(str(local_zip), str(local_data))
        try:
            local_zip.unlink()
            print(f"Removed {local_zip} after extraction.")
        except OSError as exc:
            print(f"Warning: could not remove {local_zip}: {exc}")

    if _has_npz(nested_dir):
        return nested_dir
    if _has_npz(extract_dir):
        return extract_dir
    return extract_dir if extract_dir.exists() else None


def _prepare_dataset_with_fallback(
    names: List[str], drive_root: Path, local_data: Path
) -> Tuple[Optional[Path], Optional[str]]:
    for candidate in names:
        path = _prepare_dataset(candidate, drive_root, local_data)
        if path is not None and _has_npz(path):
            return path, candidate
    return None, None


def _gather_pairs(root: Path, split: str) -> Optional[List[Tuple[str, int]]]:
    split_dir = root / split
    real_dir = split_dir / "real"
    fake_dir = split_dir / "fake"
    real = sorted(real_dir.glob("*.npz")) if real_dir.exists() else []
    fake = sorted(fake_dir.glob("*.npz")) if fake_dir.exists() else []
    if real and fake:
        return [(str(p), 0) for p in real] + [(str(p), 1) for p in fake]
    return None


def setup_data_paths(
    pretrain_dataset: str,
    finetune_dataset: str,
    drive_root: Path,
    local_data: Path,
) -> Tuple[Path, Path, List[Tuple[str, int]], List[Tuple[str, int]]]:
    pretrain_root, pretrain_used = _prepare_dataset_with_fallback(
        [pretrain_dataset], drive_root, local_data
    )
    finetune_candidates = [finetune_dataset, "preprocessed"]
    if pretrain_dataset not in finetune_candidates:
        finetune_candidates.append(pretrain_dataset)
    finetune_root, finetune_used = _prepare_dataset_with_fallback(
        finetune_candidates, drive_root, local_data
    )

    if pretrain_root is None or not _has_npz(pretrain_root):
        raise FileNotFoundError(
            f"No pretrain dataset '{pretrain_dataset}' found under {drive_root} or {local_data}"
        )

    if finetune_root is None:
        raise FileNotFoundError(
            f"No finetune dataset '{finetune_dataset}' found under {drive_root} or {local_data}"
        )

    pretrain_data_dir = pretrain_root
    if (pretrain_root / "train" / "real").exists():
        pretrain_data_dir = pretrain_root / "train" / "real"

    finetune_train_pairs = _gather_pairs(finetune_root, "train")
    finetune_val_pairs = _gather_pairs(finetune_root, "test") or _gather_pairs(
        finetune_root, "val"
    )

    if finetune_train_pairs is None:
        raise FileNotFoundError(
            f"Expected '{finetune_dataset}' to provide train/real and train/fake splits."
        )
    if finetune_val_pairs is None:
        split_idx = max(1, len(finetune_train_pairs) // 10)
        finetune_train_pairs, finetune_val_pairs = (
            finetune_train_pairs[:-split_idx],
            finetune_train_pairs[-split_idx:],
        )

    if finetune_used and finetune_used != finetune_dataset:
        print(
            f"Finetune dataset '{finetune_dataset}' missing; using '{finetune_used}' instead."
        )

    return pretrain_data_dir, finetune_root, finetune_train_pairs, finetune_val_pairs


def validate_stage1(model, loader, device, train_dtype, use_amp):
    model.eval()
    total_loss = lf_rec = bp_rec = hf_rec = 0.0
    n_batches = len(loader)
    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device, dtype=train_dtype)
        x_bp = x_bp.to(device, dtype=train_dtype)
        x_hf = x_hf.to(device, dtype=train_dtype)
        with torch.cuda.amp.autocast(enabled=use_amp):
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)
            targets = {"lf": x_lf, "bp": x_bp, "hf": x_hf}
            betas = {"lf": 1.0, "bp": 1.0, "hf": 1.0}
            loss, loss_dict = band_split_vae_loss(
                recons,
                mus,
                logvars,
                targets,
                x_hat_fused,
                None,
                betas=betas,
                alpha_fusion=0.0,
            )
        total_loss += loss_dict["total"]
        lf_rec += loss_dict["recon_lf"]
        bp_rec += loss_dict["recon_bp"]
        hf_rec += loss_dict["recon_hf"]
    return {
        "total": total_loss / n_batches,
        "lf_rec": lf_rec / n_batches,
        "bp_rec": bp_rec / n_batches,
        "hf_rec": hf_rec / n_batches,
    }


def train_epoch_stage1(model, loader, optimizer, device, train_dtype, use_amp, scaler):
    model.train()
    total_loss = lf_rec = bp_rec = hf_rec = 0.0
    n_batches = len(loader)
    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device, dtype=train_dtype)
        x_bp = x_bp.to(device, dtype=train_dtype)
        x_hf = x_hf.to(device, dtype=train_dtype)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)
            targets = {"lf": x_lf, "bp": x_bp, "hf": x_hf}
            betas = {"lf": 1.0, "bp": 1.0, "hf": 1.0}
            loss, loss_dict = band_split_vae_loss(
                recons,
                mus,
                logvars,
                targets,
                x_hat_fused,
                None,
                betas=betas,
                alpha_fusion=0.0,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss_dict["total"]
        lf_rec += loss_dict["recon_lf"]
        bp_rec += loss_dict["recon_bp"]
        hf_rec += loss_dict["recon_hf"]
    return {
        "total": total_loss / n_batches,
        "lf_rec": lf_rec / n_batches,
        "bp_rec": bp_rec / n_batches,
        "hf_rec": hf_rec / n_batches,
    }


def margin_loss(real_loss, fake_loss, margin=0.5, lambda_margin=1.0):
    loss_fake = torch.clamp(margin - fake_loss, min=0.0)
    return real_loss + lambda_margin * loss_fake


def validate_margin(model, loader, device, train_dtype, use_amp, margin):
    model.eval()
    total_loss = real_rec = fake_rec = 0.0
    n_real = n_fake = 0
    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device, dtype=train_dtype)
        x_bp = x_bp.to(device, dtype=train_dtype)
        x_hf = x_hf.to(device, dtype=train_dtype)
        labels = labels.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)
            targets = {"lf": x_lf, "bp": x_bp, "hf": x_hf}
            betas = {"lf": 1.0, "bp": 1.0, "hf": 1.0}
            for i in range(x_lf.size(0)):
                single_recons = {k: v[i : i + 1] for k, v in recons.items()}
                single_mus = {k: v[i : i + 1] for k, v in mus.items()}
                single_logvars = {k: v[i : i + 1] for k, v in logvars.items()}
                single_targets = {k: v[i : i + 1] for k, v in targets.items()}
                single_x_hat = (
                    x_hat_fused[i : i + 1] if x_hat_fused is not None else None
                )
                _, loss_dict = band_split_vae_loss(
                    single_recons,
                    single_mus,
                    single_logvars,
                    single_targets,
                    single_x_hat,
                    None,
                    betas=betas,
                    alpha_fusion=0.0,
                )
                if labels[i] == 0:
                    real_rec += loss_dict["total"]
                    n_real += 1
                else:
                    fake_rec += loss_dict["total"]
                    n_fake += 1
    real_mean = real_rec / max(n_real, 1)
    fake_mean = fake_rec / max(n_fake, 1)
    total = margin_loss(real_mean, fake_mean, margin)
    return {
        "total": float(total),
        "real_rec": float(real_mean),
        "fake_rec": float(fake_mean),
        "separation": float(fake_mean - real_mean),
    }


def train_epoch_margin(
    model,
    loader,
    optimizer,
    device,
    train_dtype,
    use_amp,
    margin,
    lambda_margin,
    scaler,
):
    model.train()
    total_loss = real_rec = fake_rec = 0.0
    n_real = n_fake = 0
    n_batches = len(loader)
    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device, dtype=train_dtype)
        x_bp = x_bp.to(device, dtype=train_dtype)
        x_hf = x_hf.to(device, dtype=train_dtype)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        real_mask = labels == 0
        fake_mask = labels == 1
        with torch.cuda.amp.autocast(enabled=use_amp):
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)
            targets = {"lf": x_lf, "bp": x_bp, "hf": x_hf}
            betas = {"lf": 1.0, "bp": 1.0, "hf": 1.0}
            if real_mask.any():
                real_loss, real_dict = band_split_vae_loss(
                    {k: v[real_mask] for k, v in recons.items()},
                    {k: v[real_mask] for k, v in mus.items()},
                    {k: v[real_mask] for k, v in logvars.items()},
                    {k: v[real_mask] for k, v in targets.items()},
                    x_hat_fused[real_mask] if x_hat_fused is not None else None,
                    None,
                    betas=betas,
                    alpha_fusion=0.0,
                )
            else:
                real_loss, real_dict = torch.zeros(1, device=device), {"total": 0.0}
            if fake_mask.any():
                fake_loss, fake_dict = band_split_vae_loss(
                    {k: v[fake_mask] for k, v in recons.items()},
                    {k: v[fake_mask] for k, v in mus.items()},
                    {k: v[fake_mask] for k, v in logvars.items()},
                    {k: v[fake_mask] for k, v in targets.items()},
                    x_hat_fused[fake_mask] if x_hat_fused is not None else None,
                    None,
                    betas=betas,
                    alpha_fusion=0.0,
                )
            else:
                fake_loss, fake_dict = torch.zeros(1, device=device), {"total": 0.0}
            loss = margin_loss(real_loss, fake_loss, margin, lambda_margin)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        real_rec += float(real_dict["total"])
        fake_rec += float(fake_dict["total"])
        n_real += real_mask.sum().item()
        n_fake += fake_mask.sum().item()
    return {
        "total": total_loss / n_batches,
        "real_rec": real_rec / n_batches,
        "fake_rec": fake_rec / n_batches,
        "separation": (fake_rec - real_rec) / n_batches,
    }


def validate_disc(
    model, discriminator, loader, device, train_dtype, use_amp, criterion
):
    model.eval()
    discriminator.eval()
    total_loss = rec_loss = cls_loss = 0.0
    n_batches = len(loader)
    correct = total = 0
    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device, dtype=train_dtype)
        x_bp = x_bp.to(device, dtype=train_dtype)
        x_hf = x_hf.to(device, dtype=train_dtype)
        labels = labels.to(device).float()
        with torch.cuda.amp.autocast(enabled=use_amp):
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)
            targets = {"lf": x_lf, "bp": x_bp, "hf": x_hf}
            betas = {"lf": 1.0, "bp": 1.0, "hf": 1.0}
            loss, loss_dict = band_split_vae_loss(
                recons,
                mus,
                logvars,
                targets,
                x_hat_fused,
                None,
                betas=betas,
                alpha_fusion=0.0,
            )
            logits = discriminator(mus["lf"], mus["bp"], mus["hf"]).squeeze(1)
            cls = criterion(logits, labels)
            total_batch = loss + cls
        total_loss += total_batch.item()
        rec_loss += loss_dict["total"]
        cls_loss += cls.item()
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)
    acc = correct / total if total else 0.0
    return {
        "total": total_loss / n_batches,
        "rec": rec_loss / n_batches,
        "cls": cls_loss / n_batches,
        "accuracy": acc,
    }


def train_epoch_disc(
    model,
    discriminator,
    loader,
    optimizer,
    device,
    train_dtype,
    use_amp,
    criterion,
    scaler,
):
    model.train()
    discriminator.train()
    total_loss = rec_loss = cls_loss = 0.0
    n_batches = len(loader)
    correct = total = 0
    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device, dtype=train_dtype)
        x_bp = x_bp.to(device, dtype=train_dtype)
        x_hf = x_hf.to(device, dtype=train_dtype)
        labels = labels.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)
            targets = {"lf": x_lf, "bp": x_bp, "hf": x_hf}
            betas = {"lf": 1.0, "bp": 1.0, "hf": 1.0}
            loss, loss_dict = band_split_vae_loss(
                recons,
                mus,
                logvars,
                targets,
                x_hat_fused,
                None,
                betas=betas,
                alpha_fusion=0.0,
            )
            logits = discriminator(mus["lf"], mus["bp"], mus["hf"]).squeeze(1)
            cls = criterion(logits, labels)
            total_batch = loss + cls
        scaler.scale(total_batch).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += total_batch.item()
        rec_loss += loss_dict["total"]
        cls_loss += cls.item()
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)
    acc = correct / total if total else 0.0
    return {
        "total": total_loss / n_batches,
        "rec": rec_loss / n_batches,
        "cls": cls_loss / n_batches,
        "accuracy": acc,
    }


class LatentDiscriminator(nn.Module):
    def __init__(self, C_z, num_bands=3, hidden_dim=128):
        super().__init__()
        input_dim = C_z * 2 * num_bands
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_lf, z_bp, z_hf):
        def pool(z):
            if z.dim() == 3:
                z_avg = z.mean(dim=2)
                z_max = z.max(dim=2)[0]
                return torch.cat([z_avg, z_max], dim=1)
            return z

        z_concat = torch.cat([pool(z_lf), pool(z_bp), pool(z_hf)], dim=1)
        return self.net(z_concat)


# ============================================
# Test code
# ============================================

if __name__ == "__main__":
    print("Testing BandSplitVAE...")

    # Parameters
    K = 40  # landmarks
    T = 150  # frames
    B = 8  # batch size

    # SIMPLE version: [(x,y), (Δx,Δy)] → F_dim=4 → C_in=160
    F_dim_simple = 4
    C_in_simple = K * F_dim_simple

    # FULL version: [(x,y), (Δx,Δy), (Δ²x,Δ²y), θ, Δθ] → F_dim=8 → C_in=320
    F_dim_full = 8
    C_in_full = K * F_dim_full

    print("\n【Test 1: SIMPLE Version (160 channels per band)】")
    model_simple = BandSplitVAE(C_in_per_band=C_in_simple, C_h=48, C_z=12)

    # Dummy inputs (3 bands)
    x_lf = torch.randn(B, C_in_simple, T)
    x_bp = torch.randn(B, C_in_simple, T)
    x_hf = torch.randn(B, C_in_simple, T)

    recons, mus, logvars, fused = model_simple(x_lf, x_bp, x_hf)

    print(f"Input shapes: LF={x_lf.shape}, BP={x_bp.shape}, HF={x_hf.shape}")
    print(
        f"Recon shapes: LF={recons['lf'].shape}, BP={recons['bp'].shape}, HF={recons['hf'].shape}"
    )
    print(f"Fused shape: {fused.shape}")
    print(
        f"Fusion weights: {F.softmax(model_simple.fusion_weights, dim=0).detach().numpy()}"
    )

    # Count parameters
    n_params = sum(p.numel() for p in model_simple.parameters())
    print(f"Total parameters: {n_params:,}")

    print("\n【Test 2: FULL Version (320 channels per band)】")
    model_full = BandSplitVAE(C_in_per_band=C_in_full, C_h=48, C_z=12)

    # Dummy inputs (3 bands)
    x_lf = torch.randn(B, C_in_full, T)
    x_bp = torch.randn(B, C_in_full, T)
    x_hf = torch.randn(B, C_in_full, T)

    recons, mus, logvars, fused = model_full(x_lf, x_bp, x_hf)

    print(f"Input shapes: LF={x_lf.shape}, BP={x_bp.shape}, HF={x_hf.shape}")
    print(
        f"Recon shapes: LF={recons['lf'].shape}, BP={recons['bp'].shape}, HF={recons['hf'].shape}"
    )
    print(f"Fused shape: {fused.shape}")

    n_params = sum(p.numel() for p in model_full.parameters())
    print(f"Total parameters: {n_params:,}")

    print("\n✓ BandSplitVAE test passed!")
