"""
Adaptive wrappers for foundation models with architectural constraints
"""

import torch
import torch.nn as nn
import numpy as np


class AdaptiveBENDR(nn.Module):
    """
    Adaptive wrapper for BENDR that handles arbitrary channel counts.
    BENDR requires 20 channels with SCALE, but we can adapt it.
    """
    def __init__(self, base_model, n_chans, n_times, n_outputs=256):
        super().__init__()
        self.base_model = base_model
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_outputs = n_outputs
        
        # Add a channel adapter if needed
        if n_chans != 20:
            self.channel_adapter = nn.Linear(n_chans, 20)
        else:
            self.channel_adapter = nn.Identity()
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        # Adapt channels to 20
        if self.n_chans != 20:
            # Reshape for linear layer
            x_reshaped = x.permute(0, 2, 1)  # (batch, n_times, n_chans)
            x_adapted = self.channel_adapter(x_reshaped)  # (batch, n_times, 20)
            x = x_adapted.permute(0, 2, 1)  # (batch, 20, n_times)
        
        # Add SCALE channel (mean of all channels)
        scale_channel = x.mean(dim=1, keepdim=True)  # (batch, 1, n_times)
        x_with_scale = torch.cat([x, scale_channel], dim=1)  # (batch, 21, n_times)
        
        # Forward through base model
        return self.base_model(x_with_scale)
    
    def get_embeddings(self, x):
        """Get embeddings without classification head."""
        # Adapt channels
        if self.n_chans != 20:
            x_reshaped = x.permute(0, 2, 1)
            x_adapted = self.channel_adapter(x_reshaped)
            x = x_adapted.permute(0, 2, 1)
        
        # Add SCALE channel
        scale_channel = x.mean(dim=1, keepdim=True)
        x_with_scale = torch.cat([x, scale_channel], dim=1)
        
        # Get embeddings from base model
        return self.base_model.get_embeddings(x_with_scale)


class AdaptiveCBraMod(nn.Module):
    """
    Adaptive wrapper for CBraMod that handles patch_size constraints.
    """
    def __init__(self, base_model, n_chans, n_times, n_outputs=256):
        super().__init__()
        self.base_model = base_model
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_outputs = n_outputs
        
        # Calculate compatible patch size
        # Find largest patch_size that divides n_times and is <= 200
        for ps in [64, 32, 16, 8]:
            if n_times % ps == 0:
                self.patch_size = ps
                break
        else:
            self.patch_size = 64  # default
        
        # Update base model's patch_size if possible
        if hasattr(self.base_model, 'patch_size'):
            self.base_model.patch_size = self.patch_size
        
        # Time dimension adapter
        self.time_adapter = nn.Linear(n_times, (n_times // self.patch_size) * self.patch_size)
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        # Ensure time dimension is compatible
        target_size = (x.shape[2] // self.patch_size) * self.patch_size
        if x.shape[2] != target_size:
            # Pad or truncate to compatible size
            if x.shape[2] < target_size:
                # Pad
                pad_size = target_size - x.shape[2]
                x = torch.nn.functional.pad(x, (0, pad_size))
            else:
                # Truncate
                x = x[:, :, :target_size]
        
        # Update patch embedding if needed
        if hasattr(self.base_model, 'patch_embedding'):
            # Recreate patch embedding with correct patch size
            pass  # Let the base model handle it
        
        return self.base_model(x)
    
    def get_embeddings(self, x):
        """Get embeddings without classification head."""
        # Ensure compatible size
        if x.shape[2] != (x.shape[2] // self.patch_size) * self.patch_size:
            target_size = (x.shape[2] // self.patch_size) * self.patch_size
            if x.shape[2] < target_size:
                pad_size = target_size - x.shape[2]
                x = torch.nn.functional.pad(x, (0, pad_size))
            else:
                x = x[:, :, :target_size]
        
        return self.base_model.get_embeddings(x)


class AdaptiveLaBraM(nn.Module):
    """
    Adaptive wrapper for LaBraM that handles channel count mismatch.
    Uses learned embeddings for missing channels.
    """
    def __init__(self, base_model, n_chans, n_times, n_outputs=256):
        super().__init__()
        self.base_model = base_model
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_outputs = n_outputs
        self.target_chans = 128
        
        # Learnable embeddings for missing channels
        self.missing_channel_embeddings = nn.Parameter(
            torch.randn(self.target_chans - n_chans, n_times) * 0.02
        )
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        # Add learned embeddings for missing channels
        if self.n_chans < self.target_chans:
            # Expand learned embeddings to batch size
            batch_size = x.shape[0]
            missing_emb = self.missing_channel_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            # Concatenate
            x = torch.cat([x, missing_emb], dim=1)
        
        return self.base_model(x)
    
    def get_embeddings(self, x):
        """Get embeddings without classification head."""
        if self.n_chans < self.target_chans:
            batch_size = x.shape[0]
            missing_emb = self.missing_channel_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            x = torch.cat([x, missing_emb], dim=1)
        
        return self.base_model.get_embeddings(x)


class AdaptiveSignalJEPA(nn.Module):
    """
    Adaptive wrapper for SignalJEPA that handles self-supervised nature.
    Extracts features without requiring target-based loss.
    """
    def __init__(self, base_model, n_chans, n_times, n_outputs=256):
        super().__init__()
        self.base_model = base_model
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_outputs = n_outputs
        
        # Add a simple projection head for classification
        self.projection_head = nn.Sequential(
            nn.Linear(n_outputs, n_outputs),
            nn.ReLU(),
            nn.Linear(n_outputs, n_outputs)
        )
    
    def forward(self, x):
        # Get features from self-supervised model (no mask argument)
        features = self.base_model(x, return_features=True)
        
        if isinstance(features, dict):
            x = features.get('features', features.get('cls_token', x))
        else:
            x = features
        
        # Project to output dimension
        x = self.projection_head(x)
        
        # Return embeddings and logits
        embeddings = x.mean(dim=-1) if x.dim() > 2 else x
        logits = embeddings
        
        return embeddings, logits
    
    def get_embeddings(self, x):
        """Get embeddings without classification head."""
        with torch.no_grad():
            features = self.base_model(x, return_features=True)
            if isinstance(features, dict):
                x = features.get('features', features.get('cls_token', x))
            else:
                x = features
            x = self.projection_head(x)
            embeddings = x.mean(dim=-1) if x.dim() > 2 else x
        return embeddings