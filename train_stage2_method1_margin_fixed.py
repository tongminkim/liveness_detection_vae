"""
Stage 2: Method 1 - Margin Loss (Fixed Version)
- Load pre-trained model from Stage 1
- Fine-tune with margin-based contrastive loss
- Real samples: minimize reconstruction loss (as in Stage 1)
- Fake samples: push reconstruction loss above margin threshold
- FIXED: Proper tensor handling in validate() to avoid gradient errors
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE, band_split_vae_loss
from config_bandvae import FullFeatureConfig


# ========================================
# Configuration
# ========================================

# Data
SPLIT_JSON = "/home/elicer/liveness_detection/model1/data_split.json"
PRETRAINED_MODEL = "runs/stage1_pretrain/stage1_pretrained.pt"

# Hyperparameters (논문 기준)
config = FullFeatureConfig()
config.T_fixed = 300
config.fc_low = 2.0
config.fc_high = 8.0
config.filter_order = 4
config.C_h = 48
config.C_z = 12
config.dilations = [1, 2, 4]
config.lr = 1e-4  # Lower learning rate for fine-tuning
config.epochs = 10  # Fewer epochs for fine-tuning
config.batch_size = 64
config.num_workers = 8

# Margin Loss parameters
MARGIN = 0.5  # Minimum reconstruction loss for fake samples
LAMBDA_MARGIN = 1.0  # Weight for margin loss

# Output
OUTPUT_DIR = "runs/stage2_method1_margin_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# Margin Loss Function
# ========================================

def margin_loss(real_loss, fake_loss, margin=0.5):
    """
    Margin-based contrastive loss
    - Real samples: minimize reconstruction loss
    - Fake samples: push reconstruction loss above margin

    Args:
        real_loss: Reconstruction loss for real samples (tensor)
        fake_loss: Reconstruction loss for fake samples (tensor)
        margin: Minimum reconstruction loss threshold for fake samples

    Returns:
        loss: Combined loss
    """
    # Real loss: minimize (as in Stage 1)
    loss_real = real_loss

    # Fake loss: push above margin using hinge loss
    # If fake_loss < margin, penalize; otherwise, no penalty
    loss_fake = torch.clamp(margin - fake_loss, min=0.0)

    return loss_real + LAMBDA_MARGIN * loss_fake


# ========================================
# Training Functions
# ========================================

@torch.no_grad()
def validate(model, loader, device, margin):
    """Validate the model - FIXED version with proper tensor handling"""
    model.eval()

    real_rec_loss = 0.0
    fake_rec_loss = 0.0
    n_real = 0
    n_fake = 0

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)
        labels = labels.to(device)

        # Separate real and fake samples
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
        betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

        # Compute reconstruction loss for real samples
        if real_mask.sum() > 0:
            real_recons = {k: v[real_mask] for k, v in recons.items()}
            real_mus = {k: v[real_mask] for k, v in mus.items()}
            real_logvars = {k: v[real_mask] for k, v in logvars.items()}
            real_targets = {k: v[real_mask] for k, v in targets.items()}
            real_x_hat_fused = x_hat_fused[real_mask] if x_hat_fused is not None else None

            loss, loss_dict = band_split_vae_loss(
                real_recons, real_mus, real_logvars, real_targets,
                real_x_hat_fused, None, betas=betas, alpha_fusion=0.0
            )

            real_rec_loss += loss.item()  # Accumulate as float
            n_real += real_mask.sum().item()

        # Compute reconstruction loss for fake samples
        if fake_mask.sum() > 0:
            fake_recons = {k: v[fake_mask] for k, v in recons.items()}
            fake_mus = {k: v[fake_mask] for k, v in mus.items()}
            fake_logvars = {k: v[fake_mask] for k, v in logvars.items()}
            fake_targets = {k: v[fake_mask] for k, v in targets.items()}
            fake_x_hat_fused = x_hat_fused[fake_mask] if x_hat_fused is not None else None

            loss, loss_dict = band_split_vae_loss(
                fake_recons, fake_mus, fake_logvars, fake_targets,
                fake_x_hat_fused, None, betas=betas, alpha_fusion=0.0
            )

            fake_rec_loss += loss.item()  # Accumulate as float
            n_fake += fake_mask.sum().item()

    # Average losses
    real_rec_loss = real_rec_loss / n_real if n_real > 0 else 0.0
    fake_rec_loss = fake_rec_loss / n_fake if n_fake > 0 else 0.0

    # Compute margin loss for total (use tensors for proper computation)
    real_loss_tensor = torch.tensor(real_rec_loss, device=device)
    fake_loss_tensor = torch.tensor(fake_rec_loss, device=device)
    total_loss = margin_loss(real_loss_tensor, fake_loss_tensor, margin)

    return {
        'total': total_loss.item(),
        'real_rec': real_rec_loss,
        'fake_rec': fake_rec_loss,
        'separation': fake_rec_loss - real_rec_loss
    }


def train_epoch(model, loader, optimizer, device, margin, use_amp=True):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    real_rec_loss = 0.0
    fake_rec_loss = 0.0
    n_real = 0
    n_fake = 0
    n_batches = len(loader)

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Separate real and fake samples
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        # Forward pass
        if use_amp and torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

                targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
                betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

                # Compute reconstruction loss for real samples
                if real_mask.sum() > 0:
                    real_recons = {k: v[real_mask] for k, v in recons.items()}
                    real_mus = {k: v[real_mask] for k, v in mus.items()}
                    real_logvars = {k: v[real_mask] for k, v in logvars.items()}
                    real_targets = {k: v[real_mask] for k, v in targets.items()}
                    real_x_hat_fused = x_hat_fused[real_mask] if x_hat_fused is not None else None

                    real_loss, real_loss_dict = band_split_vae_loss(
                        real_recons, real_mus, real_logvars, real_targets,
                        real_x_hat_fused, None, betas=betas, alpha_fusion=0.0
                    )
                else:
                    real_loss = torch.zeros(1, device=device)
                    real_loss_dict = {'total': 0.0}

                # Compute reconstruction loss for fake samples
                if fake_mask.sum() > 0:
                    fake_recons = {k: v[fake_mask] for k, v in recons.items()}
                    fake_mus = {k: v[fake_mask] for k, v in mus.items()}
                    fake_logvars = {k: v[fake_mask] for k, v in logvars.items()}
                    fake_targets = {k: v[fake_mask] for k, v in targets.items()}
                    fake_x_hat_fused = x_hat_fused[fake_mask] if x_hat_fused is not None else None

                    fake_loss, fake_loss_dict = band_split_vae_loss(
                        fake_recons, fake_mus, fake_logvars, fake_targets,
                        fake_x_hat_fused, None, betas=betas, alpha_fusion=0.0
                    )
                else:
                    fake_loss = torch.zeros(1, device=device)
                    fake_loss_dict = {'total': 0.0}

                # Margin loss (use loss tensors, not dict values)
                loss = margin_loss(real_loss, fake_loss, margin)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

            targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
            betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

            # Compute reconstruction loss for real samples
            if real_mask.sum() > 0:
                real_recons = {k: v[real_mask] for k, v in recons.items()}
                real_mus = {k: v[real_mask] for k, v in mus.items()}
                real_logvars = {k: v[real_mask] for k, v in logvars.items()}
                real_targets = {k: v[real_mask] for k, v in targets.items()}
                real_x_hat_fused = x_hat_fused[real_mask] if x_hat_fused is not None else None

                real_loss, real_loss_dict = band_split_vae_loss(
                    real_recons, real_mus, real_logvars, real_targets,
                    real_x_hat_fused, None, betas=betas, alpha_fusion=0.0
                )
            else:
                real_loss = torch.zeros(1, device=device)
                real_loss_dict = {'total': 0.0}

            # Compute reconstruction loss for fake samples
            if fake_mask.sum() > 0:
                fake_recons = {k: v[fake_mask] for k, v in recons.items()}
                fake_mus = {k: v[fake_mask] for k, v in mus.items()}
                fake_logvars = {k: v[fake_mask] for k, v in logvars.items()}
                fake_targets = {k: v[fake_mask] for k, v in targets.items()}
                fake_x_hat_fused = x_hat_fused[fake_mask] if x_hat_fused is not None else None

                fake_loss, fake_loss_dict = band_split_vae_loss(
                    fake_recons, fake_mus, fake_logvars, fake_targets,
                    fake_x_hat_fused, None, betas=betas, alpha_fusion=0.0
                )
            else:
                fake_loss = torch.zeros(1, device=device)
                fake_loss_dict = {'total': 0.0}

            # Margin loss (use loss tensors, not dict values)
            loss = margin_loss(real_loss, fake_loss, margin)

            loss.backward()
            optimizer.step()

        # Accumulate losses for logging
        total_loss += loss.item()
        real_rec_loss += real_loss.item()
        fake_rec_loss += fake_loss.item()
        n_real += real_mask.sum().item()
        n_fake += fake_mask.sum().item()

    # Compute averages
    real_rec_loss_avg = real_rec_loss / n_batches if n_batches > 0 else 0.0
    fake_rec_loss_avg = fake_rec_loss / n_batches if n_batches > 0 else 0.0

    return {
        'total': total_loss / n_batches,
        'real_rec': real_rec_loss_avg,
        'fake_rec': fake_rec_loss_avg,
        'separation': fake_rec_loss_avg - real_rec_loss_avg
    }


# ========================================
# Main Training
# ========================================

def main():
    print("="*80)
    print("Stage 2: Method 1 - Margin Loss (Fixed Version)")
    print("="*80)
    print(f"Split JSON: {SPLIT_JSON}")
    print(f"Pre-trained model: {PRETRAINED_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Hyperparameters:")
    print(f"  Margin: {MARGIN}, Lambda: {LAMBDA_MARGIN}")
    print(f"  lr: {config.lr}, epochs: {config.epochs}, batch_size: {config.batch_size}")
    print("="*80)
    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = Stage2Dataset(
        split_json_path=SPLIT_JSON,
        split_name='stage2_train',
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order,
        random_crop=True
    )

    val_dataset = Stage2Dataset(
        split_json_path=SPLIT_JSON,
        split_name='final_test',
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order,
        random_crop=False  # No random crop for validation
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

    print()

    # Load pre-trained model
    print("Loading pre-trained model...")
    checkpoint = torch.load(PRETRAINED_MODEL, map_location=device, weights_only=False)

    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    print("Starting fine-tuning...")
    print("="*80)

    best_separation = -float('inf')

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, MARGIN, use_amp=True)

        # Validate
        val_loss = validate(model, val_loader, device, MARGIN)

        epoch_time = time.time() - epoch_start_time

        # Log
        print(
            f"Epoch {epoch}/{config.epochs} - "
            f"train: total={train_loss['total']:.4f}, real={train_loss['real_rec']:.4f}, "
            f"fake={train_loss['fake_rec']:.4f}, sep={train_loss['separation']:.4f} | "
            f"val: total={val_loss['total']:.4f}, real={val_loss['real_rec']:.4f}, "
            f"fake={val_loss['fake_rec']:.4f}, sep={val_loss['separation']:.4f} "
            f"[{epoch_time/60:.1f}min]"
        )

        # Save best model (based on separation)
        if val_loss['separation'] > best_separation:
            best_separation = val_loss['separation']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_separation': best_separation,
                'val_real_loss': val_loss['real_rec'],
                'val_fake_loss': val_loss['fake_rec'],
                'config': config,
                'margin': MARGIN,
                'lambda_margin': LAMBDA_MARGIN
            }, os.path.join(OUTPUT_DIR, "stage2_method1_best.pt"))
            print(f"  → Saved best model (separation={best_separation:.4f})")

    print("="*80)
    print(f"Fine-tuning complete! Best separation: {best_separation:.4f}")
    print(f"Model saved to: {os.path.join(OUTPUT_DIR, 'stage2_method1_best.pt')}")
    print("="*80)


if __name__ == "__main__":
    main()
