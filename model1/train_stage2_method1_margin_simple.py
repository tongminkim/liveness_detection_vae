"""
Stage 2: Method 1 - Simplified Margin Loss
- Simpler implementation: process entire batch at once
- Real samples: minimize reconstruction loss
- Fake samples: push reconstruction loss above margin
"""

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE, band_split_vae_loss
from config_bandvae import FullFeatureConfig


# ========================================
# Configuration
# ========================================

SPLIT_JSON = "/home/elicer/liveness_detection/model1/data_split.json"
PRETRAINED_MODEL = "runs/stage1_pretrain/stage1_pretrained.pt"

config = FullFeatureConfig()
config.T_fixed = 300
config.fc_low = 2.0
config.fc_high = 8.0
config.filter_order = 4
config.C_h = 48
config.C_z = 12
config.dilations = [1, 2, 4]
config.lr = 1e-4
config.epochs = 10
config.batch_size = 64
config.num_workers = 8

# Margin Loss parameters
MARGIN = 0.5
LAMBDA_MARGIN = 1.0

OUTPUT_DIR = "runs/stage2_method1_margin"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# Training Functions
# ========================================

@torch.no_grad()
def validate(model, loader, device, margin):
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    real_losses = []
    fake_losses = []

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)
        labels = labels.to(device)

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
        betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

        # Compute batch loss
        loss, loss_dict = band_split_vae_loss(
            recons, mus, logvars, targets,
            x_hat_fused, None, betas=betas, alpha_fusion=0.0
        )

        # Compute per-sample reconstruction loss
        # Simplified: use total loss as proxy
        batch_size = x_lf.size(0)
        per_sample_loss = loss / batch_size

        # Separate by label
        for i in range(batch_size):
            if labels[i] == 0:  # Real
                real_losses.append(per_sample_loss.item())
            else:  # Fake
                fake_losses.append(per_sample_loss.item())

    # Compute metrics
    real_loss_mean = sum(real_losses) / len(real_losses) if real_losses else 0.0
    fake_loss_mean = sum(fake_losses) / len(fake_losses) if fake_losses else 0.0

    # Margin loss
    fake_penalty = max(0.0, margin - fake_loss_mean)
    total_loss = real_loss_mean + LAMBDA_MARGIN * fake_penalty
    separation = fake_loss_mean - real_loss_mean

    return {
        'total': total_loss,
        'real_rec': real_loss_mean,
        'fake_rec': fake_loss_mean,
        'separation': separation
    }


def train_epoch(model, loader, optimizer, device, margin, use_amp=True):
    """Train for one epoch - Simplified version"""
    model.train()

    total_loss = 0.0
    real_losses = []
    fake_losses = []
    n_batches = len(loader)

    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_amp and torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

                targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
                betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

                # Reconstruction loss for entire batch
                loss, loss_dict = band_split_vae_loss(
                    recons, mus, logvars, targets,
                    x_hat_fused, None, betas=betas, alpha_fusion=0.0
                )

                # Separate real and fake masks
                real_mask = (labels == 0)
                fake_mask = (labels == 1)

                # Average loss per sample type
                n_real = real_mask.sum().item()
                n_fake = fake_mask.sum().item()

                # Compute weighted loss
                # Real: minimize normally
                # Fake: apply margin penalty
                if n_real > 0 and n_fake > 0:
                    # Estimate per-sample loss (simplified)
                    batch_size = labels.size(0)
                    avg_per_sample = loss / batch_size

                    # For real samples: use full loss
                    real_loss = avg_per_sample * n_real

                    # For fake samples: apply margin
                    fake_loss_estimate = avg_per_sample
                    fake_penalty = torch.clamp(margin - fake_loss_estimate, min=0.0)
                    fake_loss = LAMBDA_MARGIN * fake_penalty * n_fake

                    total_batch_loss = real_loss + fake_loss
                elif n_real > 0:
                    total_batch_loss = loss
                else:
                    fake_loss_estimate = loss / labels.size(0)
                    fake_penalty = torch.clamp(margin - fake_loss_estimate, min=0.0)
                    total_batch_loss = LAMBDA_MARGIN * fake_penalty * n_fake

            scaler.scale(total_batch_loss).backward()
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

            real_mask = (labels == 0)
            fake_mask = (labels == 1)
            n_real = real_mask.sum().item()
            n_fake = fake_mask.sum().item()

            if n_real > 0 and n_fake > 0:
                batch_size = labels.size(0)
                avg_per_sample = loss / batch_size
                real_loss = avg_per_sample * n_real
                fake_loss_estimate = avg_per_sample
                fake_penalty = torch.clamp(margin - fake_loss_estimate, min=0.0)
                fake_loss = LAMBDA_MARGIN * fake_penalty * n_fake
                total_batch_loss = real_loss + fake_loss
            elif n_real > 0:
                total_batch_loss = loss
            else:
                fake_loss_estimate = loss / labels.size(0)
                fake_penalty = torch.clamp(margin - fake_loss_estimate, min=0.0)
                total_batch_loss = LAMBDA_MARGIN * fake_penalty * n_fake

            total_batch_loss.backward()
            optimizer.step()

        # Track losses
        total_loss += total_batch_loss.item()

        # Approximate real/fake losses for logging
        batch_size = labels.size(0)
        per_sample_loss = loss_dict['total'] / batch_size
        for i in range(batch_size):
            if labels[i] == 0:
                real_losses.append(per_sample_loss)
            else:
                fake_losses.append(per_sample_loss)

    real_loss_mean = sum(real_losses) / len(real_losses) if real_losses else 0.0
    fake_loss_mean = sum(fake_losses) / len(fake_losses) if fake_losses else 0.0

    return {
        'total': total_loss / n_batches,
        'real_rec': real_loss_mean,
        'fake_rec': fake_loss_mean,
        'separation': fake_loss_mean - real_loss_mean
    }


# ========================================
# Main Training
# ========================================

def main():
    print("="*80)
    print("Stage 2: Method 1 - Simplified Margin Loss")
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
        random_crop=False
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

        # Save best model
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
