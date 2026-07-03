"""
Self-Supervised EEG Foundation Models
Implements MAE, Contrastive, and JEPA objectives for pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class EEGPatchEmbedding(nn.Module):
    """Convert EEG signal to patch embeddings."""
    
    def __init__(
        self,
        n_channels: int,
        patch_size: int = 32,
        embed_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Project patches to embedding dimension
        self.projection = nn.Linear(patch_size, embed_dim)
        
        # Channel embedding
        self.channel_embedding = nn.Embedding(n_channels, embed_dim)
        
        # Positional embedding for time
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1000, embed_dim) * 0.02
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_channels, time)
        Returns:
            patches: (batch, n_patches, embed_dim)
            patch_indices: (batch, n_patches, 2) for reconstruction
        """
        batch, n_channels, time = x.shape
        
        # Create patches: (batch, n_channels, n_patches, patch_size)
        n_patches = time // self.patch_size
        x = x[:, :, :n_patches * self.patch_size]
        patches = x.reshape(batch, n_channels, n_patches, self.patch_size)
        
        # Project each patch
        # (batch, n_channels, n_patches, embed_dim)
        patch_embeds = self.projection(patches)
        
        # Add channel embeddings
        # (1, n_channels, 1, embed_dim)
        ch_embeds = self.channel_embedding(
            torch.arange(n_channels, device=x.device)
        ).unsqueeze(0).unsqueeze(2)
        
        patch_embeds = patch_embeds + ch_embeds
        
        # Add positional embeddings
        pos_embeds = self.pos_embedding[:, :n_patches, :].unsqueeze(1)
        patch_embeds = patch_embeds + pos_embeds
        
        # Reshape: (batch, n_channels * n_patches, embed_dim)
        patches = patch_embeds.reshape(batch, n_channels * n_patches, -1)
        
        # Store patch indices for reconstruction
        patch_indices = torch.stack([
            torch.arange(n_channels).unsqueeze(1).expand(-1, n_patches).flatten(),
            torch.arange(n_patches).repeat(n_channels)
        ], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        
        return self.dropout(patches), patch_indices


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for EEG (MAE).
    
    Randomly masks patches and reconstructs the masked regions.
    """
    
    def __init__(
        self,
        n_channels: int,
        time_steps: int,
        patch_size: int = 32,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        decoder_depth: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        self.n_patches = time_steps // patch_size
        self.total_patches = n_channels * self.n_patches
        
        # Patch embedding
        self.patch_embed = EEGPatchEmbedding(
            n_channels, patch_size, embed_dim, dropout
        )
        
        # Encoder (transformer)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=encoder_depth
        )
        
        # Decoder
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=decoder_depth
        )
        
        # Reconstruction head
        self.reconstruction_head = nn.Linear(embed_dim, patch_size)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_channels, time)
            mask: (batch, total_patches) boolean mask
        Returns:
            reconstructed: (batch, n_channels, time)
            latent: (batch, n_visible, embed_dim)
            mask: (batch, total_patches)
        """
        batch = x.shape[0]
        
        # Get patch embeddings
        patches, _ = self.patch_embed(x)
        
        # Generate random mask if not provided
        if mask is None:
            n_masked = int(self.total_patches * self.mask_ratio)
            mask = torch.zeros(batch, self.total_patches, device=x.device, dtype=torch.bool)
            for i in range(batch):
                mask[i, torch.randperm(self.total_patches)[:n_masked]] = True
        
        # Create masked patches
        visible_patches = patches.clone()
        # Expand mask to match patch dimensions
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        # Expand mask token to match batch size
        mask_token_expanded = self.mask_token.expand(batch, -1, -1)
        # Apply mask token where mask is True
        visible_patches = torch.where(mask_expanded, mask_token_expanded, visible_patches)
        
        # Encode
        latent = self.encoder(visible_patches)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Project back to patch space
        reconstructed = self.reconstruction_head(reconstructed)
        
        # Reshape to (batch, n_channels, n_patches, patch_size)
        reconstructed = reconstructed.reshape(
            batch, self.n_channels, self.n_patches, self.patch_size
        )
        
        # Flatten to (batch, n_channels, time)
        reconstructed = reconstructed.reshape(batch, self.n_channels, -1)
        
        return reconstructed, latent, mask
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from encoder (for downstream tasks)."""
        patches, _ = self.patch_embed(x)
        latent = self.encoder(patches)
        return latent.mean(dim=1)  # Global average pooling


class ContrastiveModel(nn.Module):
    """
    Contrastive Learning for EEG (SimCLR-style).
    
    Learns representations by maximizing agreement between
    augmented views of the same sample.
    """
    
    def __init__(
        self,
        n_channels: int,
        time_steps: int,
        embed_dim: int = 256,
        proj_dim: int = 128,
        encoder_depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        temperature: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.temperature = temperature
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_channels * time_steps, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_channels, time)
        Returns:
            embedding: (batch, embed_dim)
            projection: (batch, proj_dim)
        """
        batch = x.shape[0]
        
        # Flatten
        x_flat = x.reshape(batch, -1)
        
        # Encode
        embedding = self.encoder(x_flat)
        
        # Project
        projection = self.projection_head(embedding)
        
        return embedding, projection
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from encoder."""
        x_flat = x.reshape(x.shape[0], -1)
        return self.encoder(x_flat)


class JEPAModel(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA).
    
    Predicts latent representations of target patches from context patches.
    No waveform reconstruction - operates entirely in latent space.
    """
    
    def __init__(
        self,
        n_channels: int,
        time_steps: int,
        patch_size: int = 32,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        predictor_depth: int = 2,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        context_ratio: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.context_ratio = context_ratio
        
        self.n_patches = time_steps // patch_size
        self.total_patches = n_channels * self.n_patches
        
        # Context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=encoder_depth
        )
        
        # Target encoder (momentum updated)
        self.target_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=encoder_depth
        )
        
        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Predictor
        self.predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=predictor_depth
        )
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1000, embed_dim) * 0.02
        )
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_channels, time)
        Returns:
            pred_target: (batch, n_target_patches, embed_dim)
            context_emb: (batch, embed_dim)
            target_emb: (batch, embed_dim)
        """
        batch = x.shape[0]
        
        # Create patches
        n_patches = self.time_steps // self.patch_size
        x_patches = x[:, :, :n_patches * self.patch_size]
        x_patches = x_patches.reshape(batch, self.n_channels, n_patches, self.patch_size)
        x_patches = self.patch_embed(x_patches)
        
        # Flatten patches first
        x_patches = x_patches.reshape(batch, self.total_patches, -1)
        
        # Add positional embeddings
        pos_emb = self.pos_embedding[:, :self.total_patches, :]
        if pos_emb.shape[1] != x_patches.shape[1]:
            # Interpolate positional embeddings if needed
            pos_emb = pos_emb.transpose(1, 2)
            pos_emb = nn.functional.interpolate(pos_emb, size=x_patches.shape[1], mode='linear')
            pos_emb = pos_emb.transpose(1, 2)
        x_patches = x_patches + pos_emb
        
        # Select context and target patches
        n_context = int(self.total_patches * self.context_ratio)
        n_target = self.total_patches - n_context
        
        context_indices = torch.randperm(self.total_patches, device=x.device)[:n_context]
        target_indices = torch.randperm(self.total_patches, device=x.device)[:n_target]
        
        context_patches = x_patches[:, context_indices]
        target_patches = x_patches[:, target_indices]
        
        # Encode context
        context_emb = self.context_encoder(context_patches)
        
        # Encode target (no grad)
        with torch.no_grad():
            target_emb = self.target_encoder(target_patches)
        
        # Predict target embeddings from context
        pred_target = self.predictor(context_emb)
        
        return pred_target, context_emb.mean(dim=1), target_emb.mean(dim=1)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from context encoder."""
        batch = x.shape[0]
        n_patches = self.time_steps // self.patch_size
        x_patches = x[:, :, :n_patches * self.patch_size]
        x_patches = x_patches.reshape(batch, self.n_channels, n_patches, self.patch_size)
        x_patches = self.patch_embed(x_patches)
        
        # Flatten patches first
        x_patches = x_patches.reshape(batch, self.total_patches, -1)
        
        # Add positional embeddings
        pos_emb = self.pos_embedding[:, :self.total_patches, :]
        if pos_emb.shape[1] != x_patches.shape[1]:
            # Interpolate if needed
            pos_emb = pos_emb.transpose(1, 2)
            pos_emb = nn.functional.interpolate(pos_emb, size=x_patches.shape[1], mode='linear')
            pos_emb = pos_emb.transpose(1, 2)
        x_patches = x_patches + pos_emb
        
        # Use all patches as context
        latent = self.context_encoder(x_patches)
        return latent.mean(dim=1)


def mae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 32,
    n_channels: int = 19
) -> torch.Tensor:
    """
    Compute MAE reconstruction loss only on masked patches.
    
    Args:
        reconstructed: (batch, n_channels, time)
        original: (batch, n_channels, time)
        mask: (batch, total_patches) boolean mask
        patch_size: size of each patch
        n_channels: number of channels
    """
    batch, n_ch, time = reconstructed.shape
    n_patches = time // patch_size
    
    # Reshape mask to (batch, n_channels, n_patches)
    mask_patches = mask.reshape(batch, n_channels, n_patches)
    
    # Expand mask to match time dimension
    mask_expanded = mask_patches.unsqueeze(-1).expand(-1, -1, -1, patch_size)
    mask_expanded = mask_expanded.reshape(batch, n_channels, -1)
    
    # Only compute loss on masked regions
    loss = F.mse_loss(reconstructed[mask_expanded], original[mask_expanded])
    return loss


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    InfoNCE loss for contrastive learning.
    
    Args:
        z1, z2: (batch, proj_dim) normalized embeddings
        temperature: scaling factor
    """
    batch_size = z1.shape[0]
    
    # Normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Concatenate
    z = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    # Labels: positive pairs are (i, i+batch_size)
    labels = torch.cat([
        torch.arange(batch_size, device=z.device) + batch_size,
        torch.arange(batch_size, device=z.device)
    ])
    
    # Cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


def jepa_loss(
    pred_target: torch.Tensor,
    target_emb: torch.Tensor
) -> torch.Tensor:
    """
    JEPA loss: MSE between predicted and actual target embeddings.
    
    Args:
        pred_target: (batch, n_target, embed_dim)
        target_emb: (batch, embed_dim) - mean of target embeddings
    """
    pred_mean = pred_target.mean(dim=1)
    return F.mse_loss(pred_mean, target_emb)