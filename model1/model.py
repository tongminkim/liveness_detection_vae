"""
Frequency-Decoupled Feature-Space VAE for Lip Liveness Detection

POC: Single-stream TCN-VAE (논문의 간소화 버전)
- TCN encoder/decoder
- Feature-space reconstruction
- KL + L1 reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# TCN Building Blocks
# ============================================

class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block with residual connection"""

    def __init__(self, C_in, C_out, dilation=1, kernel_size=3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            C_in, C_out,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.gn = nn.GroupNorm(1, C_out)  # Layer normalization
        self.act = nn.SiLU()

        # Residual connection
        self.res = nn.Conv1d(C_in, C_out, 1) if C_in != C_out else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, C_in, T]
        Returns:
            y: [B, C_out, T]
        """
        y = self.act(self.gn(self.conv(x)))
        return y + self.res(x)


# ============================================
# Encoder & Decoder
# ============================================

class TCNEncoder(nn.Module):
    """TCN Encoder for VAE"""

    def __init__(self, C_in, C_h=48, C_z=12, dilations=[1, 2, 4]):
        """
        Args:
            C_in: Input channels
            C_h: Hidden channels
            C_z: Latent dimension
            dilations: Dilation factors for TCN blocks
        """
        super().__init__()

        self.inp = nn.Conv1d(C_in, C_h, 1)

        # TCN blocks with increasing dilation
        tcn_blocks = []
        for d in dilations:
            tcn_blocks.append(TCNBlock(C_h, C_h, dilation=d))
        self.tcn = nn.Sequential(*tcn_blocks)

        # Output: mu and logvar
        self.out = nn.Conv1d(C_h, C_z * 2, 1)

    def forward(self, x):
        """
        Args:
            x: [B, C_in, T]
        Returns:
            mu: [B, C_z, T]
            logvar: [B, C_z, T]
        """
        h = self.tcn(self.inp(x))
        mu, logvar = torch.chunk(self.out(h), 2, dim=1)
        return mu, logvar


class TCNDecoder(nn.Module):
    """TCN Decoder for VAE"""

    def __init__(self, C_z=12, C_h=48, C_out=320, dilations=[4, 2, 1]):
        """
        Args:
            C_z: Latent dimension
            C_h: Hidden channels
            C_out: Output channels (should match input)
            dilations: Dilation factors (reversed from encoder)
        """
        super().__init__()

        self.fc = nn.Conv1d(C_z, C_h, 1)

        # TCN blocks with decreasing dilation
        tcn_blocks = []
        for d in dilations:
            tcn_blocks.append(TCNBlock(C_h, C_h, dilation=d))
        self.tcn = nn.Sequential(*tcn_blocks)

        self.out = nn.Conv1d(C_h, C_out, 1)

    def forward(self, z):
        """
        Args:
            z: [B, C_z, T]
        Returns:
            x_hat: [B, C_out, T]
        """
        h = self.tcn(self.fc(z))
        return self.out(h)


# ============================================
# VAE Model
# ============================================

class FeatureSpaceVAE(nn.Module):
    """
    Feature-Space VAE for One-Class Lip Liveness Detection

    - Operates directly on feature space (no latent temporal conv)
    - Reconstructs engineered features: position + velocity only
    - KL + L1 reconstruction loss
    """

    def __init__(self, C_in=160, C_h=48, C_z=12, dilations=[1, 2, 4]):
        """
        Args:
            C_in: Input channels (K × F_dim, e.g., 40 × 4 = 160)
                  - K=40 landmarks (lips)
                  - F_dim=4 features (position + velocity)
            C_h: Hidden channels
            C_z: Latent dimension
            dilations: Dilation factors for TCN
        """
        super().__init__()

        self.C_in = C_in
        self.C_h = C_h
        self.C_z = C_z

        self.encoder = TCNEncoder(C_in, C_h, C_z, dilations)
        self.decoder = TCNDecoder(C_z, C_h, C_in, list(reversed(dilations)))

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick

        Args:
            mu: [B, C_z, T]
            logvar: [B, C_z, T]
        Returns:
            z: [B, C_z, T]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [B, C_in, T]
        Returns:
            x_hat: [B, C_in, T] - reconstructed features
            mu: [B, C_z, T]
            logvar: [B, C_z, T]
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def encode(self, x):
        """Encode to latent space (deterministic)"""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)


# ============================================
# Loss Functions
# ============================================

def reconstruction_loss(x_hat, x, reduction='mean'):
    """L1 reconstruction loss"""
    return F.l1_loss(x_hat, x, reduction=reduction)


def kl_divergence(mu, logvar, reduction='mean'):
    """KL divergence loss: KL(q(z|x) || p(z))"""
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    elif reduction == 'none':
        return kl
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    """
    Combined VAE loss

    Args:
        x_hat: Reconstructed input [B, C, T]
        x: Original input [B, C, T]
        mu: Latent mean [B, C_z, T]
        logvar: Latent log variance [B, C_z, T]
        beta: KL weight (beta-VAE)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon = reconstruction_loss(x_hat, x)
    kl = kl_divergence(mu, logvar)

    total = recon + beta * kl

    return total, recon, kl


# ============================================
# Anomaly Scoring
# ============================================

@torch.no_grad()
def compute_anomaly_score(model, x, beta=1.0):
    """
    Compute anomaly score for input

    Args:
        model: VAE model
        x: Input [B, C, T] or [C, T]
        beta: KL weight

    Returns:
        score: Anomaly score (scalar or [B])
    """
    model.eval()

    # Add batch dimension if needed
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1, C, T]

    x_hat, mu, logvar = model(x)

    # Per-sample reconstruction error
    recon = F.l1_loss(x_hat, x, reduction='none').mean(dim=[1, 2])  # [B]

    # Per-sample KL divergence
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=[1, 2])  # [B]

    # Combined score
    score = recon + beta * kl

    return score.squeeze() if score.size(0) == 1 else score


# ============================================
# Test code
# ============================================

if __name__ == "__main__":
    print("Testing FeatureSpaceVAE...")

    # Hyperparameters
    C_in = 160  # 40 landmarks × 4 features (position + velocity)
    T = 150     # 5초 @ 30fps
    B = 8       # Batch size
    C_h = 48
    C_z = 12

    # Create model
    model = FeatureSpaceVAE(C_in=C_in, C_h=C_h, C_z=C_z)

    print(f"\nModel created:")
    print(f"  Input channels: {C_in}")
    print(f"  Hidden channels: {C_h}")
    print(f"  Latent dim: {C_z}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Test forward pass
    x = torch.randn(B, C_in, T)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")

    x_hat, mu, logvar = model(x)

    print(f"  Output shape: {x_hat.shape}")
    print(f"  Latent mu shape: {mu.shape}")
    print(f"  Latent logvar shape: {logvar.shape}")

    # Test loss
    total, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=1.0)
    print(f"\nLoss values:")
    print(f"  Total: {total.item():.4f}")
    print(f"  Recon: {recon.item():.4f}")
    print(f"  KL: {kl.item():.4f}")

    # Test anomaly scoring
    score = compute_anomaly_score(model, x[0])  # Single sample
    print(f"\nAnomaly score (single sample): {score.item():.4f}")

    batch_scores = compute_anomaly_score(model, x)  # Batch
    print(f"Anomaly scores (batch): {batch_scores.shape}")

    print("\n✓ Model test passed!")
