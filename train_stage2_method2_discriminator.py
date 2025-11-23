"""
Stage 2: Method 2 - Discriminator in Latent Space
- Load pre-trained model from Stage 1
- Add discriminator head on latent representations
- Train with reconstruction loss + classification loss
- Discriminator classifies Real (0) vs Fake (1)
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

# Discriminator parameters
LAMBDA_CLS = 1.0  # Weight for classification loss

# Output
OUTPUT_DIR = "runs/stage2_method2_discriminator"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# Discriminator Model
# ========================================

class LatentDiscriminator(nn.Module):
    """
    Discriminator in latent space
    - Input: Concatenated latent vectors from LF, BP, HF bands
    - Output: Binary classification (Real=0, Fake=1)
    """

    def __init__(self, C_z, num_bands=3, hidden_dim=128):
        """
        Args:
            C_z: Latent dimension per band
            num_bands: Number of frequency bands (default: 3 for LF, BP, HF)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Input: Max + Average pooling for each band
        # Each band contributes C_z*2 features (C_z for avg, C_z for max)
        input_dim = C_z * 2 * num_bands  # Total: C_z * 2 * 3 = C_z * 6

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification
        )

    def forward(self, z_lf, z_bp, z_hf):
        """
        Args:
            z_lf: Latent vector for LF band (B, C_z, T) or (B, C_z)
            z_bp: Latent vector for BP band (B, C_z, T) or (B, C_z)
            z_hf: Latent vector for HF band (B, C_z, T) or (B, C_z)

        Returns:
            logits: Classification logits (B, 1)
        """
        # Pool temporal dimension if present (B, C_z, T) -> (B, C_z*2)
        # Use both max and average pooling for richer representation
        if len(z_lf.shape) == 3:
            z_lf_avg = z_lf.mean(dim=2)  # Average pooling (global pattern)
            z_lf_max = z_lf.max(dim=2)[0]  # Max pooling (salient features)
            z_lf = torch.cat([z_lf_avg, z_lf_max], dim=1)  # (B, C_z*2)

            z_bp_avg = z_bp.mean(dim=2)
            z_bp_max = z_bp.max(dim=2)[0]
            z_bp = torch.cat([z_bp_avg, z_bp_max], dim=1)  # (B, C_z*2)

            z_hf_avg = z_hf.mean(dim=2)
            z_hf_max = z_hf.max(dim=2)[0]
            z_hf = torch.cat([z_hf_avg, z_hf_max], dim=1)  # (B, C_z*2)

        # Concatenate latent vectors from all bands
        z_concat = torch.cat([z_lf, z_bp, z_hf], dim=1)  # (B, C_z*2*3 = C_z*6)

        # Pass through discriminator
        logits = self.net(z_concat)  # (B, 1)

        return logits


# ========================================
# Training Functions
# ========================================

@torch.no_grad()
def validate(model, discriminator, loader, device):
    """Validate the model"""
    model.eval()
    discriminator.eval()

    total_loss = 0.0
    rec_loss = 0.0
    cls_loss = 0.0
    n_batches = len(loader)

    correct = 0
    total = 0

    criterion_cls = nn.BCEWithLogitsLoss()

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)
        labels = labels.to(device).float()

        # Forward pass (VAE)
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
        betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

        # Reconstruction loss
        loss, loss_dict = band_split_vae_loss(
            recons, mus, logvars, targets,
            x_hat_fused, None, betas=betas, alpha_fusion=0.0
        )

        # Discriminator classification
        z_lf = mus['lf']  # (B, C_z)
        z_bp = mus['bp']  # (B, C_z)
        z_hf = mus['hf']  # (B, C_z)

        logits = discriminator(z_lf, z_bp, z_hf).squeeze(1)  # (B,)
        cls_loss_batch = criterion_cls(logits, labels)

        # Total loss
        total_loss_batch = loss + LAMBDA_CLS * cls_loss_batch

        # Accumulate
        total_loss += total_loss_batch.item()
        rec_loss += loss_dict['total']
        cls_loss += cls_loss_batch.item()

        # Accuracy
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0

    return {
        'total': total_loss / n_batches,
        'rec': rec_loss / n_batches,
        'cls': cls_loss / n_batches,
        'accuracy': accuracy
    }


def train_epoch(model, discriminator, loader, optimizer, device, use_amp=True):
    """Train for one epoch"""
    model.train()
    discriminator.train()

    total_loss = 0.0
    rec_loss = 0.0
    cls_loss = 0.0
    n_batches = len(loader)

    correct = 0
    total = 0

    criterion_cls = nn.BCEWithLogitsLoss()

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        # Forward pass
        if use_amp and torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                # VAE forward
                recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

                targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
                betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

                # Reconstruction loss
                loss, loss_dict = band_split_vae_loss(
                    recons, mus, logvars, targets,
                    x_hat_fused, None, betas=betas, alpha_fusion=0.0
                )

                # Discriminator classification
                z_lf = mus['lf']  # (B, C_z)
                z_bp = mus['bp']  # (B, C_z)
                z_hf = mus['hf']  # (B, C_z)

                logits = discriminator(z_lf, z_bp, z_hf).squeeze(1)  # (B,)
                cls_loss_batch = criterion_cls(logits, labels)

                # Total loss
                total_loss_batch = loss + LAMBDA_CLS * cls_loss_batch

            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # VAE forward
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

            targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
            betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

            # Reconstruction loss
            loss, loss_dict = band_split_vae_loss(
                recons, mus, logvars, targets,
                x_hat_fused, None, betas=betas, alpha_fusion=0.0
            )

            # Discriminator classification
            z_lf = mus['lf']  # (B, C_z)
            z_bp = mus['bp']  # (B, C_z)
            z_hf = mus['hf']  # (B, C_z)

            logits = discriminator(z_lf, z_bp, z_hf).squeeze(1)  # (B,)
            cls_loss_batch = criterion_cls(logits, labels)

            # Total loss
            total_loss_batch = loss + LAMBDA_CLS * cls_loss_batch

            total_loss_batch.backward()
            optimizer.step()

        # Accumulate
        total_loss += total_loss_batch.item()
        rec_loss += loss_dict['total']
        cls_loss += cls_loss_batch.item()

        # Accuracy
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0

    return {
        'total': total_loss / n_batches,
        'rec': rec_loss / n_batches,
        'cls': cls_loss / n_batches,
        'accuracy': accuracy
    }


# ========================================
# Main Training
# ========================================

def main():
    print("="*80)
    print("Stage 2: Method 2 - Discriminator in Latent Space")
    print("="*80)
    print(f"Split JSON: {SPLIT_JSON}")
    print(f"Pre-trained model: {PRETRAINED_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Hyperparameters:")
    print(f"  Lambda_cls: {LAMBDA_CLS}")
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
    checkpoint = torch.load(PRETRAINED_MODEL, map_location=device)

    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    # Create discriminator
    discriminator = LatentDiscriminator(
        C_z=config.C_z,
        num_bands=3,
        hidden_dim=128
    ).to(device)

    n_disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Discriminator parameters: {n_disc_params:,}")
    print()

    # Optimizer (both VAE and discriminator)
    optimizer = optim.Adam(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=config.lr
    )

    # Training loop
    print("Starting fine-tuning...")
    print("="*80)

    best_accuracy = 0.0

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, discriminator, train_loader, optimizer, device, use_amp=True)

        # Validate
        val_loss = validate(model, discriminator, val_loader, device)

        epoch_time = time.time() - epoch_start_time

        # Log
        print(
            f"Epoch {epoch}/{config.epochs} - "
            f"train: total={train_loss['total']:.4f}, rec={train_loss['rec']:.4f}, "
            f"cls={train_loss['cls']:.4f}, acc={train_loss['accuracy']:.4f} | "
            f"val: total={val_loss['total']:.4f}, rec={val_loss['rec']:.4f}, "
            f"cls={val_loss['cls']:.4f}, acc={val_loss['accuracy']:.4f} "
            f"[{epoch_time/60:.1f}min]"
        )

        # Save best model (based on accuracy)
        if val_loss['accuracy'] > best_accuracy:
            best_accuracy = val_loss['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_accuracy,
                'val_rec_loss': val_loss['rec'],
                'val_cls_loss': val_loss['cls'],
                'config': config,
                'lambda_cls': LAMBDA_CLS
            }, os.path.join(OUTPUT_DIR, "stage2_method2_best.pt"))
            print(f"  → Saved best model (accuracy={best_accuracy:.4f})")

    print("="*80)
    print(f"Fine-tuning complete! Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {os.path.join(OUTPUT_DIR, 'stage2_method2_best.pt')}")
    print("="*80)


if __name__ == "__main__":
    main()
