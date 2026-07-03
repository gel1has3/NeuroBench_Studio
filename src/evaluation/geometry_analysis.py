"""
Representation Geometry Analysis

Tools for analyzing EEG foundation model embeddings:
- CKA (Centered Kernel Alignment)
- Intrinsic dimensionality
- Manifold structure
- Feature probing
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Centered Kernel Alignment between representation matrices.
    
    Args:
        X: (n_samples, n_features) - embeddings from model 1
        Y: (n_samples, n_features) - embeddings from model 2
        
    Returns:
        cka: float in [0, 1], 1 = identical representations
    """
    # Center the matrices
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # Compute Gram matrices
    K = X_centered @ X_centered.T
    L = Y_centered @ Y_centered.T
    
    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_xy = np.sum(K * L) / (X.shape[0] - 1)**2
    hsic_xx = np.sum(K * K) / (X.shape[0] - 1)**2
    hsic_yy = np.sum(L * L) / (Y.shape[0] - 1)**2
    
    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
    
    return float(cka)


def compute_cka_matrix(
    embeddings_per_model: Dict[str, np.ndarray],
    diseases: Optional[List[str]] = None
) -> np.ndarray:
    """
    Compute pairwise CKA between model embeddings across diseases.
    
    Args:
        embeddings_per_model: {model_name: (n_diseases, n_samples, n_features)}
        diseases: list of disease names
        
    Returns:
        cka_matrix: (n_models, n_models) average CKA across diseases
    """
    models = list(embeddings_per_model.keys())
    n_models = len(models)
    cka_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i, n_models):
            cka_values = []
            
            # Compute CKA for each disease
            for d_idx in range(embeddings_per_model[models[i]].shape[0]):
                X = embeddings_per_model[models[i]][d_idx]
                Y = embeddings_per_model[models[j]][d_idx]
                
                if X.shape[0] > 1:  # Need at least 2 samples
                    cka_values.append(linear_cka(X, Y))
            
            avg_cka = np.mean(cka_values) if cka_values else 0.0
            cka_matrix[i, j] = avg_cka
            cka_matrix[j, i] = avg_cka
    
    return cka_matrix


def compute_intrinsic_dimensionality(
    embeddings: np.ndarray,
    method: str = ' participation_ratio'
) -> float:
    """
    Compute intrinsic dimensionality of embedding space.
    
    Args:
        embeddings: (n_samples, n_features)
        method: 'participation_ratio', 'eff_rank', or 'mle'
        
    Returns:
        dimensionality: estimated intrinsic dimension
    """
    if method == 'participation_ratio':
        # Participation ratio: (sum eigenvalues)^2 / (sum eigenvalues^2)
        cov = np.cov(embeddings.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        if len(eigenvalues) == 0:
            return 0.0
        
        pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
        return float(pr)
    
    elif method == 'eff_rank':
        # Effective rank based on entropy of eigenvalues
        cov = np.cov(embeddings.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        if len(eigenvalues) == 0:
            return 0.0
        
        # Normalize eigenvalues to probabilities
        p = eigenvalues / np.sum(eigenvalues)
        
        # Entropy
        entropy = -np.sum(p * np.log(p + 1e-10))
        
        # Effective rank
        eff_rank = np.exp(entropy)
        
        return float(eff_rank)
    
    elif method == 'mle':
        # Maximum likelihood estimator (Levina-Bickel)
        n_samples, n_features = embeddings.shape
        
        if n_samples < 5:
            return 1.0
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(embeddings))
        
        # Sort distances for each point
        sorted_distances = np.sort(distances, axis=1)
        
        # Exclude self-distances (first column)
        sorted_distances = sorted_distances[:, 1:]
        
        # MLE estimator
        k = min(10, n_samples - 1)  # Use k nearest neighbors
        distances_k = sorted_distances[:, k-1]
        
        # Log distances
        log_distances = np.log(distances_k + 1e-10)
        
        # MLE estimate
        mean_log = np.mean(log_distances)
        log_dist0 = np.log(sorted_distances[:, 0] + 1e-10)
        d_mle = 1.0 / (mean_log - log_dist0)
        
        # Take mean if array, then clip
        d_mle = float(np.mean(d_mle))
        d_mle = np.clip(d_mle, 1, n_features)
        
        return float(d_mle)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_manifold_structure(
    embeddings_per_disease: Dict[str, np.ndarray],
    k_neighbors: int = 10
) -> Dict:
    """
    Analyze manifold structure across diseases.
    
    Args:
        embeddings_per_disease: {disease: (n_samples, n_features)}
        k_neighbors: number of neighbors for k-NN analysis
        
    Returns:
        metrics: dictionary with manifold structure metrics
    """
    diseases = list(embeddings_per_disease.keys())
    n_diseases = len(diseases)
    
    # Compute k-NN overlap between diseases
    knn_overlap = np.zeros((n_diseases, n_diseases))
    
    for i in range(n_diseases):
        for j in range(n_diseases):
            if i == j:
                knn_overlap[i, j] = 1.0
            else:
                overlap = _compute_knn_overlap(
                    embeddings_per_disease[diseases[i]],
                    embeddings_per_disease[diseases[j]],
                    k=k_neighbors
                )
                knn_overlap[i, j] = overlap
    
    # Compute cluster purity (how well diseases cluster)
    cluster_purity = _compute_cluster_purity(embeddings_per_disease)
    
    # Compute disease mixing score
    mixing_score = 1 - np.mean(knn_overlap[np.triu_indices(n_diseases, k=1)])
    
    metrics = {
        'knn_overlap_matrix': knn_overlap,
        'diseases': diseases,
        'cluster_purity': cluster_purity,
        'disease_mixing_score': mixing_score,
        'mean_knn_overlap': np.mean(knn_overlap[np.triu_indices(n_diseases, k=1)]),
    }
    
    return metrics


def _compute_knn_overlap(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 10
) -> float:
    """Compute k-NN overlap between two point sets."""
    n_x = min(X.shape[0], k + 1)
    n_y = min(Y.shape[0], k + 1)
    
    if n_x < 2 or n_y < 2:
        return 0.0
    
    # Fit nearest neighbors
    nn_x = NearestNeighbors(n_neighbors=min(k, X.shape[0]-1))
    nn_x.fit(X)
    
    nn_y = NearestNeighbors(n_neighbors=min(k, Y.shape[0]-1))
    nn_y.fit(Y)
    
    # Get neighbors
    _, indices_x = nn_x.kneighbors(X)
    _, indices_y = nn_y.kneighbors(Y)
    
    # Compute overlap (Jaccard similarity)
    overlaps = []
    for i in range(min(len(indices_x), len(indices_y))):
        set_x = set(indices_x[i])
        set_y = set(indices_y[i])
        
        if len(set_x) > 0 and len(set_y) > 0:
            jaccard = len(set_x & set_y) / len(set_x | set_y)
            overlaps.append(jaccard)
    
    return float(np.mean(overlaps)) if overlaps else 0.0


def _compute_cluster_purity(
    embeddings_per_disease: Dict[str, np.ndarray],
    n_clusters: Optional[int] = None
) -> float:
    """Compute how well diseases cluster in embedding space."""
    from sklearn.cluster import KMeans
    
    # Combine all embeddings
    all_embeddings = []
    all_labels = []
    
    for disease, embeddings in embeddings_per_disease.items():
        all_embeddings.append(embeddings)
        all_labels.extend([disease] * len(embeddings))
    
    X = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    
    if n_clusters is None:
        n_clusters = len(embeddings_per_disease)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Compute purity
    purity = 0.0
    for cluster in range(n_clusters):
        mask = cluster_labels == cluster
        if mask.sum() > 0:
            most_common_label = np.bincount(
                [list(embeddings_per_disease.keys()).index(l) for l in labels[mask]]
            ).argmax()
            purity += (labels[mask] == list(embeddings_per_disease.keys())[most_common_label]).sum()
    
    purity /= len(labels)
    
    return float(purity)


def train_feature_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probe_type: str = 'linear',
    cv: int = 5
) -> Dict:
    """
    Train a probe to extract information from embeddings.
    
    Args:
        embeddings: (n_samples, n_features)
        labels: (n_samples,) - target labels
        probe_type: 'linear' or 'logistic'
        cv: cross-validation folds
        
    Returns:
        results: dict with probe performance
    """
    from sklearn.model_selection import cross_val_score
    
    if probe_type == 'linear':
        from sklearn.linear_model import RidgeClassifier
        probe = RidgeClassifier()
    else:
        probe = LogisticRegression(max_iter=1000, multi_class='auto')
    
    # Cross-validated accuracy
    scores = cross_val_score(probe, embeddings, labels, cv=cv, scoring='accuracy')
    
    # Fit on all data for final probe
    probe.fit(embeddings, labels)
    
    results = {
        'probe_type': probe_type,
        'cv_accuracy_mean': float(np.mean(scores)),
        'cv_accuracy_std': float(np.std(scores)),
        'n_samples': len(embeddings),
        'n_features': embeddings.shape[1],
        'n_classes': len(np.unique(labels)),
    }
    
    return results


def compute_feature_importance(
    embeddings: np.ndarray,
    labels: np.ndarray,
    task_name: str
) -> Dict:
    """
    Compute which embedding dimensions are important for a task.
    
    Args:
        embeddings: (n_samples, n_features)
        labels: (n_samples,)
        task_name: name of the probing task
        
    Returns:
        importance: dict with feature importance metrics
    """
    from sklearn.linear_model import LogisticRegression
    
    # Train logistic regression
    lr = LogisticRegression(max_iter=1000, multi_class='auto')
    lr.fit(embeddings, labels)
    
    # Feature importance (absolute weights)
    if len(np.unique(labels)) == 2:
        importance = np.abs(lr.coef_[0])
    else:
        importance = np.abs(lr.coef_).mean(axis=0)
    
    # Top features
    top_k = min(20, len(importance))
    top_indices = np.argsort(importance)[-top_k:][::-1]
    
    results = {
        'task': task_name,
        'mean_importance': float(np.mean(importance)),
        'std_importance': float(np.std(importance)),
        'top_features': {
            f'dim_{idx}': float(importance[idx]) 
            for idx in top_indices
        },
        'sparsity': float(np.mean(importance < 0.01)),
    }
    
    return results


def analyze_representation_geometry(
    embeddings_per_model_disease: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path
) -> Dict:
    """
    Complete geometry analysis pipeline.
    
    Args:
        embeddings_per_model_disease: {
            model_name: {
                disease: (n_samples, n_features)
            }
        }
        output_dir: directory to save results
        
    Returns:
        analysis_results: complete geometry analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'models': {},
        'cross_model_cka': {},
        'intrinsic_dimensionality': {},
        'manifold_structure': {},
    }
    
    # Analyze each model
    for model_name, disease_embeddings in embeddings_per_model_disease.items():
        model_results = {
            'diseases': {},
            'avg_dimensionality': {},
        }
        
        # Intrinsic dimensionality per disease
        dims = []
        for disease, embeddings in disease_embeddings.items():
            dim_pr = compute_intrinsic_dimensionality(embeddings, 'participation_ratio')
            dim_eff = compute_intrinsic_dimensionality(embeddings, 'eff_rank')
            dim_mle = compute_intrinsic_dimensionality(embeddings, 'mle')
            
            model_results['diseases'][disease] = {
                'participation_ratio': dim_pr,
                'effective_rank': dim_eff,
                'mle_estimate': dim_mle,
                'n_samples': embeddings.shape[0],
                'n_features': embeddings.shape[1],
            }
            
            dims.append(dim_pr)
        
        model_results['avg_dimensionality'] = {
            'mean_pr': float(np.mean(dims)),
            'std_pr': float(np.std(dims)),
        }
        
        results['models'][model_name] = model_results
        
        # Manifold structure per model
        manifold = compute_manifold_structure(disease_embeddings)
        results['manifold_structure'][model_name] = manifold
    
    # Cross-model CKA
    models = list(embeddings_per_model_disease.keys())
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            # Stack embeddings across diseases
            X_list = []
            Y_list = []
            
            for disease in embeddings_per_model_disease[models[i]].keys():
                if disease in embeddings_per_model_disease[models[j]]:
                    X_list.append(embeddings_per_model_disease[models[i]][disease])
                    Y_list.append(embeddings_per_model_disease[models[j]][disease])
            
            if X_list and Y_list:
                X = np.vstack(X_list)
                Y = np.vstack(Y_list)
                
                cka = linear_cka(X, Y)
                
                pair_key = f"{models[i]}_vs_{models[j]}"
                results['cross_model_cka'][pair_key] = {
                    'cka': float(cka),
                    'n_samples': X.shape[0],
                }
    
    # Save results (convert numpy arrays to lists)
    results_file = output_dir / 'geometry_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(_convert_to_serializable(results), f, indent=2)
    
    return results


def _convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-compatible types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def compare_geometry_across_objectives(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    diseases: List[str]
) -> pd.DataFrame:
    """
    Compare geometry metrics across self-supervised objectives (MAE, Contrastive, JEPA).
    
    Args:
        embeddings: {objective: {disease: (n_samples, n_features)}}
        diseases: list of disease names
        
    Returns:
        comparison: DataFrame with geometry metrics per objective
    """
    import pandas as pd
    
    rows = []
    
    for objective, disease_embeddings in embeddings.items():
        # Intrinsic dimensionality
        dims = []
        for disease in diseases:
            if disease in disease_embeddings:
                dim = compute_intrinsic_dimensionality(
                    disease_embeddings[disease], 
                    'participation_ratio'
                )
                dims.append(dim)
        
        # Manifold structure
        manifold = compute_manifold_structure(disease_embeddings)
        
        # Cross-disease CKA
        cka_values = []
        for i in range(len(diseases)):
            for j in range(i+1, len(diseases)):
                d1, d2 = diseases[i], diseases[j]
                if d1 in disease_embeddings and d2 in disease_embeddings:
                    cka = linear_cka(
                        disease_embeddings[d1],
                        disease_embeddings[d2]
                    )
                    cka_values.append(cka)
        
        rows.append({
            'objective': objective,
            'mean_dimensionality': float(np.mean(dims)),
            'std_dimensionality': float(np.std(dims)),
            'disease_mixing_score': manifold['disease_mixing_score'],
            'mean_knn_overlap': manifold['mean_knn_overlap'],
            'cluster_purity': manifold['cluster_purity'],
            'mean_cross_disease_cka': float(np.mean(cka_values)) if cka_values else 0.0,
            'std_cross_disease_cka': float(np.std(cka_values)) if cka_values else 0.0,
        })
    
    return pd.DataFrame(rows)