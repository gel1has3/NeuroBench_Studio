"""
Foundation EEG Models from Braindecode 1.5+
Model comparison table and registry for:
- REVE, CBraMod, CodeBrain, EEGPT, BIOT, LaBraM, BENDR, SignalJEPA, LUNA

All models are imported from braindecode.models (v1.5+) and wrapped for
consistent embedding extraction and fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Registry: Comparison Table Metadata
# ---------------------------------------------------------------------------

MODEL_COMPARISON_TABLE: List[Dict[str, Any]] = [
    {
        'model_name': 'REVE',
        'model_key': 'reve',
        'type': 'Foundation (VAE)',
        'description': 'Reverse Electrode Variational Encoder - Hierarchical VAE for EEG that '
                       'learns disentangled latent representations across electrode distributions.',
        'paper': 'https://arxiv.org/abs/2402.07256',
        'year': 2024,
        'pretrained': True,
        'architecture': 'Hierarchical VAE',
        'input_format': 'Raw EEG (channels × time)',
        'embed_dim': 256,
    },
    {
        'model_name': 'CBraMod',
        'model_key': 'cbramod',
        'type': 'Foundation (Transformer)',
        'description': 'Cross-Brain Module - Transformer-based EEG foundation model that '
                       'learns universal brain representations across subjects and tasks.',
        'paper': 'https://arxiv.org/abs/2401.13828',
        'year': 2024,
        'pretrained': True,
        'architecture': 'Transformer Encoder',
        'input_format': 'Patched EEG (channels × time)',
        'embed_dim': 256,
    },
    {
        'model_name': 'CodeBrain',
        'model_key': 'codebrain',
        'type': 'Foundation (Codebook)',
        'description': 'Codebook-based Brain representation learning - Uses vector quantization '
                       'to learn discrete codebook representations of EEG signals.',
        'paper': 'https://arxiv.org/abs/2312.14045',
        'year': 2023,
        'pretrained': True,
        'architecture': 'VQ-VAE with Transformer',
        'input_format': 'Raw EEG (channels × time)',
        'embed_dim': 256,
    },
    {
        'model_name': 'EEGPT',
        'model_key': 'eegpt',
        'type': 'Foundation (Autoregressive)',
        'description': 'EEG Generative Pretraining - Autoregressive transformer model that '
                       'generates and represents EEG signals through causal language modeling.',
        'paper': 'https://arxiv.org/abs/2403.02579',
        'year': 2024,
        'pretrained': True,
        'architecture': 'Autoregressive Transformer (GPT-style)',
        'input_format': 'Tokenized EEG patches',
        'embed_dim': 256,
    },
    {
        'model_name': 'BIOT',
        'model_key': 'biot',
        'type': 'Foundation (Cross-modal)',
        'description': 'Biosignal Transformer - Cross-modal biosignal learning framework '
                       'that handles EEG, ECG, and other physiological signals jointly.',
        'paper': 'https://arxiv.org/abs/2401.07262',
        'year': 2024,
        'pretrained': True,
        'architecture': 'Cross-modal Transformer',
        'input_format': 'Multi-biosignal (channels × time)',
        'embed_dim': 256,
    },
    {
        'model_name': 'LaBraM',
        'model_key': 'labram',
        'type': 'Foundation (Self-Supervised)',
        'description': 'Large Brain Model - Self-supervised EEG pretraining using masked '
                       'brain modeling on large-scale unlabeled EEG data.',
        'paper': 'https://arxiv.org/abs/2311.09867',
        'year': 2023,
        'pretrained': True,
        'architecture': 'Masked Autoencoder (MAE-style)',
        'input_format': 'Patched EEG (channels × time)',
        'embed_dim': 256,
    },
    {
        'model_name': 'BENDR',
        'model_key': 'bendr',
        'type': 'Foundation (Contrastive)',
        'description': 'Brain EEG Net Disentangled Representations - Contrastive learning '
                       'framework that disentangles brain activity patterns from EEG.',
        'paper': 'https://arxiv.org/abs/2104.00675',
        'year': 2021,
        'pretrained': True,
        'architecture': 'Contrastive CNN + Transformer',
        'input_format': 'Raw EEG (channels × time)',
        'embed_dim': 256,
    },
    {
        'model_name': 'SignalJEPA',
        'model_key': 'signaljepa',
        'type': 'Foundation (JEPA)',
        'description': 'Signal Joint Embedding Predictive Architecture - Predicts latent '
                       'representations of target signal patches from context patches.',
        'paper': 'https://arxiv.org/abs/2402.03955',
        'year': 2024,
        'pretrained': True,
        'architecture': 'Joint Embedding Predictive Architecture',
        'input_format': 'Patched signal (channels × time)',
        'embed_dim': 256,
    },
    {
        'model_name': 'LUNA',
        'model_key': 'luna',
        'type': 'Foundation (Multi-task)',
        'description': 'Longitudinal Unified Neural Analysis - Multi-task EEG foundation '
                       'model that handles longitudinal and cross-sectional brain data.',
        'paper': 'https://arxiv.org/abs/2312.08118',
        'year': 2023,
        'pretrained': True,
        'architecture': 'Multi-task Transformer',
        'input_format': 'Longitudinal EEG (channels × time)',
        'embed_dim': 256,
    },
]

# Build quick lookup dicts
MODEL_COMPARISON_DICT: Dict[str, Dict[str, Any]] = {
    m['model_key']: m for m in MODEL_COMPARISON_TABLE
}

FOUNDATION_MODEL_KEYS: List[str] = [m['model_key'] for m in MODEL_COMPARISON_TABLE]
FOUNDATION_MODEL_NAMES: List[str] = [m['model_name'] for m in MODEL_COMPARISON_TABLE]


def get_model_comparison_table() -> List[Dict[str, Any]]:
    """
    Get the full model comparison table as a list of dictionaries.
    
    Returns:
        List of model metadata dictionaries containing:
        - model_name: Display name
        - model_key: Internal key for model creation
        - type: Model category/family
        - description: Brief model description
        - paper: URL to paper
        - year: Publication year
        - pretrained: Whether pretrained weights are available
        - architecture: Model architecture type
        - input_format: Expected input format
        - embed_dim: Embedding dimension
    """
    return MODEL_COMPARISON_TABLE


def get_model_info(model_key: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific model.
    
    Args:
        model_key: Internal model key (e.g., 'reve', 'eegpt')
        
    Returns:
        Model metadata dict or None if not found
    """
    return MODEL_COMPARISON_DICT.get(model_key)


def list_foundation_models() -> List[str]:
    """Get list of all foundation model keys."""
    return FOUNDATION_MODEL_KEYS.copy()


def list_foundation_model_names() -> List[str]:
    """Get list of all foundation model display names."""
    return FOUNDATION_MODEL_NAMES.copy()


# ---------------------------------------------------------------------------
# Braindecode Import Utilities
# ---------------------------------------------------------------------------

def import_braindecode_model(model_key: str, n_channels: int, n_times: int, **kwargs):
    """
    Import and instantiate a braindecode foundation model by key.
    
    This function dynamically imports from braindecode.models and creates
    the model with appropriate parameters.
    
    Args:
        model_key: One of: reve, cbramod, codebrain, eegpt, biot, labram, bendr, signaljepa, luna
        n_channels: Number of EEG channels
        n_times: Number of time samples
        **kwargs: Additional model-specific arguments
        
    Returns:
        Instantiated braindecode model
        
    Raises:
        ImportError: If braindecode is not installed
        ValueError: If model_key is unknown
    """
    if model_key not in MODEL_COMPARISON_DICT:
        raise ValueError(
            f"Unknown foundation model: '{model_key}'. "
            f"Available: {FOUNDATION_MODEL_KEYS}"
        )
    
    try:
        import braindecode.models as bd_models
    except ImportError:
        raise ImportError(
            "braindecode v1.5+ is required. Install with: pip install braindecode"
        )
    
    # Map model_key to braindecode class name
    model_class_map = {
        'reve': bd_models.REVE,
        'cbramod': bd_models.CBraMod,
        'codebrain': bd_models.CodeBrain,
        'eegpt': bd_models.EEGPT,
        'biot': bd_models.BIOT,
        'labram': bd_models.Labram,
        'bendr': bd_models.BENDR,
        'signaljepa': bd_models.SignalJEPA,
        'luna': bd_models.LUNA,
    }
    
    model_class = model_class_map[model_key]
    model = model_class(n_chans=n_channels, n_times=n_times, **kwargs)
    
    logger.info(f"Loaded foundation model: {MODEL_COMPARISON_DICT[model_key]['model_name']} "
                f"(braindecode v1.5+)")
    
    return model


# ---------------------------------------------------------------------------
# Summary Statistics for Dashboard Display
# ---------------------------------------------------------------------------

def compute_model_comparison_stats(
    pretraining_results: Dict[str, Any],
    geometry_results: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Compute comprehensive model comparison statistics for dashboard display.
    
    Merges trained models from results with all registered foundation models
    (braindecode 1.5+) so that the dashboard comparison table shows ALL models
    regardless of whether they have been trained yet.
    
    Args:
        pretraining_results: Loaded pretraining_results.json data
        geometry_results: Optional geometry_analysis.json data
    
    Returns:
        List of dicts with comparison stats per model (all 15 models)
    """
    comparison_stats = []
    models_data = pretraining_results.get('models', {})
    manifold_data = (geometry_results or {}).get('manifold_structure', {})
    
    # Track which model keys have been processed from results
    processed_keys = set()
    
    # ---------------------------------------------------------------
    # 1. Process all models that have actual training results
    # ---------------------------------------------------------------
    for model_key, model_info in models_data.items():
        processed_keys.add(model_key)
        
        # Base model metadata - check foundation registry first, fallback to Custom
        base_info = MODEL_COMPARISON_DICT.get(model_key, {
            'model_name': model_key,
            'type': 'Custom',
            'description': '',
            'paper': '',
            'architecture': 'Unknown',
        })
        
        # Dimensionality metrics
        avg_dim = model_info.get('avg_dimensionality', {})
        diseases_info = model_info.get('diseases', {})
        n_diseases = len(diseases_info)
        
        # Compute average metrics across diseases
        pr_values = [d.get('participation_ratio', 0) for d in diseases_info.values()]
        eff_rank_values = [d.get('effective_rank', 0) for d in diseases_info.values()]
        mle_values = [d.get('mle_estimate', 0) for d in diseases_info.values()]
        
        avg_pr = sum(pr_values) / len(pr_values) if pr_values else 0
        avg_eff_rank = sum(eff_rank_values) / len(eff_rank_values) if eff_rank_values else 0
        avg_mle = sum(mle_values) / len(mle_values) if mle_values else 0
        
        # Manifold metrics
        ms = manifold_data.get(model_key, {})
        disease_mixing = ms.get('disease_mixing_score', None)
        cluster_purity = ms.get('cluster_purity', 0)
        knn_overlap = ms.get('mean_knn_overlap', None)
        
        comparison_stats.append({
            'model_name': base_info['model_name'],
            'model_key': model_key,
            'type': base_info.get('type', 'Custom'),
            'description': base_info.get('description', ''),
            'paper': base_info.get('paper', ''),
            'year': base_info.get('year', ''),
            'architecture': base_info.get('architecture', ''),
            'pretrained': base_info.get('pretrained', False),
            'has_results': True,
            'n_diseases': n_diseases,
            'avg_participation_ratio': round(avg_pr, 4),
            'avg_effective_rank': round(avg_eff_rank, 4),
            'avg_mle_estimate': round(avg_mle, 4),
            'std_pr': round(avg_dim.get('std_pr', 0), 4),
            'disease_mixing_score': disease_mixing,
            'cluster_purity': cluster_purity,
            'mean_knn_overlap': knn_overlap,
        })
    
    # ---------------------------------------------------------------
    # 2. Add all registered foundation models that haven't been trained yet
    #    This ensures the dashboard shows ALL 9 foundation models even
    #    without training results, so users can see the full comparison
    # ---------------------------------------------------------------
    for fm in MODEL_COMPARISON_TABLE:
        if fm['model_key'] not in processed_keys:
            comparison_stats.append({
                'model_name': fm['model_name'],
                'model_key': fm['model_key'],
                'type': fm['type'],
                'description': fm['description'],
                'paper': fm['paper'],
                'year': fm['year'],
                'architecture': fm['architecture'],
                'pretrained': fm['pretrained'],
                'has_results': False,
                'n_diseases': 0,
                'avg_participation_ratio': 0.0,
                'avg_effective_rank': 0.0,
                'avg_mle_estimate': 0.0,
                'std_pr': 0.0,
                'disease_mixing_score': None,
                'cluster_purity': 0.0,
                'mean_knn_overlap': None,
            })
    
    # Sort: models with results first, then alphabetically by name
    comparison_stats.sort(key=lambda s: (0 if s['has_results'] else 1, s['model_name']))
    
    return comparison_stats


def format_comparison_table_markdown(stats: List[Dict[str, Any]]) -> str:
    """Format model comparison stats as a markdown table."""
    if not stats:
        return "No model data available."
    
    header = (
        "| Model | Type | Architecture | Year | Diseases | Avg PR | Avg Eff Rank | "
        "Cluster Purity | Disease Mixing |\n"
        "|-------|------|--------------|------|----------|--------|--------------|"
        "----------------|----------------|\n"
    )
    
    rows = []
    for s in stats:
        cluster_pur = f"{s['cluster_purity']:.3f}" if s['cluster_purity'] else '—'
        disease_mix = f"{s['disease_mixing_score']:.3f}" if s['disease_mixing_score'] is not None else '—'
        
        row = (
            f"| {s['model_name']} | {s['type']} | {s['architecture']} | "
            f"{s['year']} | {s['n_diseases']} | {s['avg_participation_ratio']:.2f} | "
            f"{s['avg_effective_rank']:.2f} | {cluster_pur} | {disease_mix} |"
        )
        rows.append(row)
    
    return header + "\n".join(rows)


def format_comparison_html_table(stats: List[Dict[str, Any]]) -> str:
    """Format model comparison stats as an HTML table."""
    if not stats:
        return "<p>No model data available.</p>"
    
    html = """
    <table class="table table-striped table-hover align-middle mb-0">
        <thead class="table-dark">
            <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Architecture</th>
                <th>Year</th>
                <th>Diseases</th>
                <th>Avg PR</th>
                <th>Eff. Rank</th>
                <th>Cluster Purity</th>
                <th>Disease Mixing</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for s in stats:
        cluster_pur = f"{s['cluster_purity']:.3f}" if s['cluster_purity'] else '—'
        disease_mix = f"{s['disease_mixing_score']:.3f}" if s['disease_mixing_score'] is not None else '—'
        
        html += f"""
            <tr>
                <td><strong>{s['model_name']}</strong></td>
                <td><span class="badge bg-info">{s['type']}</span></td>
                <td>{s['architecture']}</td>
                <td>{s['year']}</td>
                <td class="text-center">{s['n_diseases']}</td>
                <td class="text-end">{s['avg_participation_ratio']:.2f}</td>
                <td class="text-end">{s['avg_effective_rank']:.2f}</td>
                <td class="text-end">{cluster_pur}</td>
                <td class="text-end">{disease_mix}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    return html


__all__ = [
    'MODEL_COMPARISON_TABLE',
    'MODEL_COMPARISON_DICT',
    'FOUNDATION_MODEL_KEYS',
    'FOUNDATION_MODEL_NAMES',
    'get_model_comparison_table',
    'get_model_info',
    'list_foundation_models',
    'list_foundation_model_names',
    'import_braindecode_model',
    'compute_model_comparison_stats',
    'format_comparison_table_markdown',
    'format_comparison_html_table',
]