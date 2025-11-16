"""
Stage 1: Pre-training with Real/Live Data Only
- Dataset: /home/elicer/liveness_detection/model1/20GBprocessed
- Method: One-Class Learning (VAE on Real data only)
- Hyperparameters: 논문 (PROJECT.md) 기준
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset_bandvae import LipLivenessBandDataset
from model_bandvae import BandSplitVAE, band_split_vae_loss
from config_bandvae import FullFeatureConfig


# ========================================
# Configuration
# ========================================

# Data
TRAIN_DIR = "/home/elicer/liveness_detection/model1/20GBprocessed"

# Hyperparameters (논문 기준, T_fixed만 300으로 변경)
config = FullFeatureConfig()
config.T_fixed = 300  # Changed from 150 to 300
config.fc_low = 2.0  # 논문
config.fc_high = 8.0  # 논문
config.filter_order = 4  # 논문
config.C_h = 48  # 논문
config.C_z = 12  # 논문
config.dilations = [1, 2, 4]  # 논문
config.lr = 1e-3  # 논문
config.epochs = 20  # 논문
config.batch_size = 64  # 논문
config.num_workers = 8

# Output
OUTPUT_DIR = "runs/stage1_pretrain"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# Training Functions
# ========================================

@torch.no_grad()
def validate(model, loader, device):
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    lf_rec_loss = 0.0
    bp_rec_loss = 0.0
    hf_rec_loss = 0.0
    n_batches = len(loader)

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
        betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

        loss, loss_dict = band_split_vae_loss(
            recons, mus, logvars, targets,
            x_hat_fused, None, betas=betas, alpha_fusion=0.0
        )

        total_loss += loss_dict['total']
        lf_rec_loss += loss_dict['recon_lf']
        bp_rec_loss += loss_dict['recon_bp']
        hf_rec_loss += loss_dict['recon_hf']

    return {
        'total': total_loss / n_batches,
        'lf_rec': lf_rec_loss / n_batches,
        'bp_rec': bp_rec_loss / n_batches,
        'hf_rec': hf_rec_loss / n_batches
    }


def train_epoch(model, loader, optimizer, device, use_amp=True):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    lf_rec_loss = 0.0
    bp_rec_loss = 0.0
    hf_rec_loss = 0.0
    n_batches = len(loader)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

                targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
                betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

                loss, loss_dict = band_split_vae_loss(
                    recons, mus, logvars, targets,
                    x_hat_fused, None, betas=betas, alpha_fusion=0.0
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

            targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
            betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

            loss, loss_dict = band_split_vae_loss(
                recons, mus, logvars, targets,
                x_hat_fused, None, betas=betas, alpha_fusion=0.0
            )

            loss.backward()
            optimizer.step()

        total_loss += loss_dict['total']
        lf_rec_loss += loss_dict['recon_lf']
        bp_rec_loss += loss_dict['recon_bp']
        hf_rec_loss += loss_dict['recon_hf']

    return {
        'total': total_loss / n_batches,
        'lf_rec': lf_rec_loss / n_batches,
        'bp_rec': bp_rec_loss / n_batches,
        'hf_rec': hf_rec_loss / n_batches
    }


# ========================================
# Main Training
# ========================================

def main():
    print("="*80)
    print("Stage 1: Pre-training with Real/Live Data Only")
    print("="*80)
    print(f"Dataset: {TRAIN_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Hyperparameters (논문 기준):")
    print(f"  T_fixed: {config.T_fixed}")
    print(f"  fc_low: {config.fc_low} Hz, fc_high: {config.fc_high} Hz")
    print(f"  C_h: {config.C_h}, C_z: {config.C_z}")
    print(f"  dilations: {config.dilations}")
    print(f"  lr: {config.lr}, epochs: {config.epochs}, batch_size: {config.batch_size}")
    print(f"  Feature: FULL (position + velocity + acceleration + angle + angle_rate)")
    print("="*80)
    print()

    # Load dataset
    print("Loading dataset...")
    full_dataset = LipLivenessBandDataset(
        data_dir=TRAIN_DIR,
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order,
        random_crop=True  # Random crop for training
    )

    # Split into train/val (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )

    print(f"Train size: {train_size}, Val size: {val_size}")
    print()

    # Create model
    print("Creating model...")
    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    print("Starting training...")
    print("="*80)

    best_val_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, use_amp=True)

        # Validate
        val_loss = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start_time

        # Log
        print(
            f"Epoch {epoch}/{config.epochs} - "
            f"train_loss: {train_loss['total']:.4f} "
            f"(lf:{train_loss['lf_rec']:.4f}, bp:{train_loss['bp_rec']:.4f}, hf:{train_loss['hf_rec']:.4f}), "
            f"val_loss: {val_loss['total']:.4f} "
            f"(lf:{val_loss['lf_rec']:.4f}, bp:{val_loss['bp_rec']:.4f}, hf:{val_loss['hf_rec']:.4f}) "
            f"[{epoch_time/60:.1f}min]"
        )

        # Save best model
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, os.path.join(OUTPUT_DIR, "stage1_pretrained.pt"))
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

    print("="*80)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(OUTPUT_DIR, 'stage1_pretrained.pt')}")
    print("="*80)


if __name__ == "__main__":
    main()
