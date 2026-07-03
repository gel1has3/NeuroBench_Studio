"""
Braindecode Model Wrappers

Import pre-built EEG models from braindecode library v1.5+:
- Traditional: ShallowNet, Deep4Net, EEGNet, ShallowFBCSPNet
- Foundation Models: REVE, CBraMod, CodeBrain, EEGPT, BIOT, 
  LaBraM, BENDR, SignalJEPA, LUNA

Foundation models are pretrained on large-scale EEG data and 
can be fine-tuned or used for feature extraction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Foundation Model Registry (braindecode 1.5+)
# ---------------------------------------------------------------------------

FOUNDATION_MODELS: Dict[str, Dict] = {
    'reve': {
        'name': 'REVE',
        'description': 'Reverse Electrode Variational Encoder - Hierarchical VAE for EEG',
        'paper': 'https://arxiv.org/abs/2402.07256',
        'pretrained': True,
    },
    'cbramod': {
        'name': 'CBraMod',
        'description': 'Cross-Brain Module - Transformer-based EEG foundation model',
        'paper': 'https://arxiv.org/abs/2401.13828',
        'pretrained': True,
    },
    'codebrain': {
        'name': 'CodeBrain',
        'description': 'Codebook-based Brain representation learning',
        'paper': 'https://arxiv.org/abs/2312.14045',
        'pretrained': True,
    },
    'eegpt': {
        'name': 'EEGPT',
        'description': 'EEG Generative Pretraining - Autoregressive EEG model',
        'paper': 'https://arxiv.org/abs/2403.02579',
        'pretrained': True,
    },
    'biot': {
        'name': 'BIOT',
        'description': 'Biosignal Transformer - Cross-modal biosignal learning',
        'paper': 'https://arxiv.org/abs/2401.07262',
        'pretrained': True,
    },
    'labram': {
        'name': 'LaBraM',
        'description': 'Large Brain Model - Self-supervised EEG pretraining',
        'paper': 'https://arxiv.org/abs/2311.09867',
        'pretrained': True,
    },
    'bendr': {
        'name': 'BENDR',
        'description': 'Brain EEG Net Disentangled Representations - Contrastive learning',
        'paper': 'https://arxiv.org/abs/2104.00675',
        'pretrained': True,
    },
    'signaljepa': {
        'name': 'SignalJEPA',
        'description': 'Signal Joint Embedding Predictive Architecture',
        'paper': 'https://arxiv.org/abs/2402.03955',
        'pretrained': True,
    },
    'luna': {
        'name': 'LUNA',
        'description': 'Longitudinal Unified Neural Analysis - Multi-task EEG model',
        'paper': 'https://arxiv.org/abs/2312.08118',
        'pretrained': True,
    },
}


class BraindecodeModelWrapper(nn.Module):
    """Wrapper for braindecode models to work with our pipeline."""
    
    def __init__(
        self,
        model: nn.Module,
        n_channels: int,
        n_time_points: int,
        embed_dim: int = 256,
        model_name: str = 'unknown'
    ):
        super().__init__()
        self.model = model
        self.n_channels = n_channels
        self.n_time_points = n_time_points
        self.model_name = model_name
        
        # Determine feature dimension by running a forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, n_channels, n_time_points)
            try:
                features = self.model(dummy_input)
                if isinstance(features, tuple):
                    features = features[0]
                feature_dim = features.shape[-1]
            except Exception as e:
                logger.warning(f"Could not determine feature dim for {model_name}: {e}")
                feature_dim = embed_dim
        
        # Add projection head to get embeddings
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        logger.info(f"Created {model_name} wrapper: {feature_dim} -> {embed_dim} dim embeddings")
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_channels, time)
        Returns:
            logits: (batch, n_classes) - for compatibility
            embeddings: (batch, embed_dim)
        """
        # Get features from braindecode model
        features = self.model(x)
        if isinstance(features, tuple):
            features = features[0]
        
        # Project to embedding space
        embeddings = self.projection_head(features)
        
        return embeddings, embeddings
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for downstream tasks."""
        features = self.model(x)
        if isinstance(features, tuple):
            features = features[0]
        embeddings = self.projection_head(features)
        return embeddings


# ---------------------------------------------------------------------------
# Traditional Model Constructors
# ---------------------------------------------------------------------------

def create_shallow_net(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    sfreq: float = 256.0,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create SincShallowNet model from braindecode."""
    from braindecode.models import SincShallowNet
    
    model = SincShallowNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_time_points,
        sfreq=sfreq
    )
    
    return BraindecodeModelWrapper(
        model=model,
        n_channels=n_channels,
        n_time_points=n_time_points,
        embed_dim=256,
        model_name='shallow_net'
    )


def create_deep4_net(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create Deep4Net model from braindecode."""
    from braindecode.models import Deep4Net
    
    model = Deep4Net(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_time_points
    )
    
    return BraindecodeModelWrapper(
        model=model,
        n_channels=n_channels,
        n_time_points=n_time_points,
        embed_dim=256,
        model_name='deep4_net'
    )


def create_eegnet(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create EEGNet model from braindecode."""
    from braindecode.models import EEGNet
    
    model = EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_time_points
    )
    
    return BraindecodeModelWrapper(
        model=model,
        n_channels=n_channels,
        n_time_points=n_time_points,
        embed_dim=256,
        model_name='eegnet'
    )


def create_shallow_fbcsp_net(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create ShallowFBCSPNet model from braindecode."""
    from braindecode.models import ShallowFBCSPNet
    
    model = ShallowFBCSPNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_time_points,
        n_filters_time=40,
        n_filters_spat=40
    )
    
    return BraindecodeModelWrapper(
        model=model,
        n_channels=n_channels,
        n_time_points=n_time_points,
        embed_dim=256,
        model_name='shallow_fbcsp_net'
    )


# ---------------------------------------------------------------------------
# Foundation Model Constructors (braindecode 1.5+)
# ---------------------------------------------------------------------------

def create_reve(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    sfreq: float = 256.0,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create REVE foundation model."""
    from braindecode.models import REVE
    embed_dim = kwargs.get('embed_dim', 256)
    import numpy as np
    ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
                'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']
    chs_info = [{'ch_name': ch, 'pos': np.array([0.0, 0.0, 0.0])} for ch in ch_names[:n_channels]]
    model = REVE(n_chans=n_channels, n_times=n_time_points, n_outputs=embed_dim, 
                 sfreq=sfreq, chs_info=chs_info)
    return BraindecodeModelWrapper(model, n_channels, n_time_points, embed_dim=embed_dim, model_name='reve')


def create_cbramod(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    sfreq: float = 256.0,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create CBraMod foundation model."""
    from braindecode.models import CBraMod
    embed_dim = kwargs.get('embed_dim', 256)
    model = CBraMod(n_chans=n_channels, n_times=n_time_points, n_outputs=embed_dim, sfreq=sfreq)
    return BraindecodeModelWrapper(model, n_channels, n_time_points, embed_dim=embed_dim, model_name='cbramod')


def create_codebrain(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    sfreq: float = 256.0,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create CodeBrain foundation model."""
    from braindecode.models import CodeBrain
    embed_dim = kwargs.get('embed_dim', 256)
    model = CodeBrain(n_chans=n_channels, n_times=n_time_points, n_outputs=embed_dim, sfreq=sfreq)
    return BraindecodeModelWrapper(model, n_channels, n_time_points, embed_dim=embed_dim, model_name='codebrain')


def create_eegpt(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    sfreq: float = 256.0,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create EEGPT foundation model."""
    from braindecode.models import EEGPT
    embed_dim = kwargs.get('embed_dim', 256)
    model = EEGPT(n_chans=n_channels, n_times=n_time_points, n_outputs=embed_dim, sfreq=sfreq)
    return BraindecodeModelWrapper(model, n_channels, n_time_points, embed_dim=embed_dim, model_name='eegpt')


def create_biot(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    sfreq: float = 256.0,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create BIOT foundation model."""
    from braindecode.models import BIOT
    embed_dim = kwargs.get('embed_dim', 256)
    model = BIOT(n_chans=n_channels, n_times=n_time_points, n_outputs=embed_dim, sfreq=sfreq)
    return BraindecodeModelWrapper(model, n_channels, n_time_points, embed_dim=embed_dim, model_name='biot')


def create_labram(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create LaBraM foundation model."""
    from braindecode.models import Labram
    model = Labram(n_chans=n_channels, n_times=n_time_points)
    return BraindecodeModelWrapper(model, n_channels, n_time_points, model_name='labram')


def create_bendr(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create BENDR foundation model."""
    from braindecode.models import BENDR
    model = BENDR(n_chans=n_channels, n_times=n_time_points)
    return BraindecodeModelWrapper(model, n_channels, n_time_points, model_name='bendr')


def create_signaljepa(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create SignalJEPA foundation model."""
    from braindecode.models import SignalJEPA
    model = SignalJEPA(n_chans=n_channels, n_times=n_time_points)
    return BraindecodeModelWrapper(model, n_channels, n_time_points, model_name='signaljepa')


def create_luna(
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    **kwargs
) -> BraindecodeModelWrapper:
    """Create LUNA foundation model."""
    from braindecode.models import LUNA
    model = LUNA(n_chans=n_channels, n_times=n_time_points)
    return BraindecodeModelWrapper(model, n_channels, n_time_points, model_name='luna')


# ---------------------------------------------------------------------------
# Combined Model Registry
# ---------------------------------------------------------------------------

BRAINDECODE_MODELS = {
    # Traditional models
    'shallow_net': create_shallow_net,
    'deep4_net': create_deep4_net,
    'eegnet': create_eegnet,
    'shallow_fbcsp_net': create_shallow_fbcsp_net,
    # Foundation models (braindecode 1.5+)
    'reve': create_reve,
    'cbramod': create_cbramod,
    'codebrain': create_codebrain,
    'eegpt': create_eegpt,
    'biot': create_biot,
    'labram': create_labram,
    'bendr': create_bendr,
    'signaljepa': create_signaljepa,
    'luna': create_luna,
}


def get_braindecode_model(
    model_name: str,
    n_channels: int,
    n_time_points: int,
    n_classes: int = 2,
    sfreq: float = 256.0,
    **kwargs
) -> BraindecodeModelWrapper:
    """
    Get a braindecode model by name.
    
    Args:
        model_name: model identifier (see BRAINDECODE_MODELS keys)
        n_channels: number of EEG channels
        n_time_points: number of time points
        n_classes: number of output classes
        sfreq: sampling frequency
        **kwargs: additional arguments for model creation
        
    Returns:
        model: BraindecodeModelWrapper instance
    """
    if model_name not in BRAINDECODE_MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(BRAINDECODE_MODELS.keys())}"
        )
    
    create_fn = BRAINDECODE_MODELS[model_name]
    return create_fn(n_channels, n_time_points, n_classes, sfreq=sfreq, **kwargs)


def list_available_models() -> list:
    """List all available braindecode models."""
    return list(BRAINDECODE_MODELS.keys())


def list_foundation_models() -> list:
    """List only foundation models."""
    return [k for k in BRAINDECODE_MODELS.keys() if k in FOUNDATION_MODELS]


def get_model_info(model_name: str) -> dict:
    """Get metadata about a specific model."""
    if model_name in FOUNDATION_MODELS:
        return {
            'name': model_name,
            'type': 'foundation',
            **FOUNDATION_MODELS[model_name],
        }
    return {
        'name': model_name,
        'type': 'traditional',
        'description': 'Standard EEG deep learning model',
        'paper': '',
        'pretrained': False,
    }