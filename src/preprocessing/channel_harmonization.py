"""
Channel Harmonization for Cross-Dataset EEG Analysis

Implements dual-track harmonization:
- Track A: 19-channel common space (10-20 system)
- Track B: Native montage preservation
"""

import numpy as np
import mne
from mne.channels import make_standard_montage
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json


# Standard 19-channel 10-20 montage
COMMON_19_CHANNELS = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# Alternative names mapping
CHANNEL_ALIASES = {
    'T7': 'T3',
    'T8': 'T4',
    'P7': 'T5',
    'P8': 'T6',
}


class ChannelHarmonizer:
    """Harmonize EEG channels across datasets."""
    
    def __init__(
        self,
        target_montage: str = 'standard_1020',
        common_channels: Optional[List[str]] = None
    ):
        self.target_montage = target_montage
        self.common_channels = common_channels or COMMON_19_CHANNELS
        self.montage = make_standard_montage(target_montage)
        
    def harmonize_to_19ch(
        self,
        raw: mne.io.BaseRaw,
        method: str = 'interpolation',
        montage: Any = None
    ) -> mne.io.BaseRaw:
        """
        Project raw EEG to 19-channel common space.
        
        Args:
            raw: MNE Raw object
            method: 'interpolation' or 'csd' (current source density)
            montage: Optional montage to use (defaults to self.montage)
            
        Returns:
            raw_19ch: Raw object with exactly 19 channels
        """
        # Use provided montage or default
        active_montage = montage if montage is not None else self.montage
        
        # Normalize channel names
        raw = self._normalize_channel_names(raw)
        
        # Get available channels
        available = [ch for ch in self.common_channels if ch in raw.ch_names]
        missing = [ch for ch in self.common_channels if ch not in raw.ch_names]
        
        if not available:
            raise ValueError(f"No common channels found. Available: {raw.ch_names}")
        
        # Pick available channels
        raw.pick_channels(available)
        
        # Interpolate missing channels
        if missing:
            print(f"Interpolating {len(missing)} missing channels: {missing}")
            
            if method == 'csd':
                # Current Source Density (better for spatial interpolation)
                raw = self._interpolate_csd(raw, missing)
            else:
                # Standard spherical spline interpolation
                raw = self._interpolate_spherical(raw, missing)
        
        # Reorder to standard order
        raw.reorder_channels(self.common_channels[:len(raw.ch_names)])
        
        # Set montage
        raw.set_montage(active_montage, on_missing='ignore')
        
        return raw
    
    def _normalize_channel_names(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Standardize channel names using aliases."""
        name_mapping = {}
        for ch in raw.ch_names:
            normalized = ch.upper().strip()
            normalized = CHANNEL_ALIASES.get(normalized, normalized)
            if normalized != ch:
                name_mapping[ch] = normalized
        
        if name_mapping:
            raw.rename_channels(name_mapping)
        
        return raw
    
    def _interpolate_spherical(
        self,
        raw: mne.io.BaseRaw,
        missing_channels: List[str]
    ) -> mne.io.BaseRaw:
        """Interpolate missing channels using spherical spline."""
        # Add reference channels (zeros)
        raw = mne.add_reference_channels(raw, missing_channels)
        
        # Set montage (on_missing='ignore' for the new channels)
        raw.set_montage(self.montage, on_missing='ignore')
        
        # Mark the added channels as bad so they get interpolated
        for ch in missing_channels:
            if ch in raw.ch_names:
                raw.info['bads'].append(ch)
        
        # Interpolate bads
        raw.interpolate_bads(reset_bads=False, method='spline')
        
        return raw
    
    def _interpolate_csd(
        self,
        raw: mne.io.BaseRaw,
        missing_channels: List[str]
    ) -> mne.io.BaseRaw:
        """Interpolate using Current Source Density."""
        from mne.channels import compute_current_source_density
        
        # Add reference channels
        raw = mne.add_reference_channels(raw, missing_channels)
        raw.set_montage(self.montage, on_missing='ignore')
        
        # Compute CSD
        raw_csd = compute_current_source_density(raw)
        
        return raw_csd
    
    def get_native_channel_count(self, raw: mne.io.BaseRaw) -> int:
        """Get number of native channels."""
        return len(raw.ch_names)
    
    def get_channel_positions(
        self,
        raw: mne.io.BaseRaw
    ) -> Dict[str, Tuple[float, float, float]]:
        """Get 3D positions of channels."""
        if not raw.info['dig']:
            raw.set_montage(self.montage, on_missing='ignore')
        
        positions = {}
        for ch in raw.ch_names:
            if ch in raw.info['chs']:
                loc = raw.info['chs'][raw.ch_names.index(ch)]['loc'][:3]
                positions[ch] = tuple(loc)
        
        return positions
    
    def harmonize_to_n_channels(
        self,
        raw: mne.io.BaseRaw,
        target_channels: List[str],
        method: str = 'spline',
        montage: Any = None
    ) -> mne.io.BaseRaw:
        """
        Project EEG to arbitrary channel layout using spherical spline interpolation.
        
        This enables training models that require specific channel montages
        (e.g., BENDR with 20ch, LaBraM with 128ch) from 19-channel data.
        
        Args:
            raw: MNE Raw object with 19 channels
            target_channels: List of target channel names
            method: 'spline' or 'csd'
            montage: Optional montage to use (defaults to self.montage)
            
        Returns:
            raw_nch: Raw object with target channel layout
        """
        # Use provided montage or default
        active_montage = montage if montage is not None else self.montage
        
        # First harmonize to 19ch
        raw_19ch = self.harmonize_to_19ch(raw, method=method)
        
        # Get current channels that exist in target montage
        current_channels = [ch for ch in raw_19ch.ch_names if ch in active_montage.ch_names]
        
        # Determine which channels need to be added
        channels_to_add = [ch for ch in target_channels if ch not in current_channels]
        
        if not channels_to_add:
            # Just reorder if all channels present
            available_target = [ch for ch in target_channels if ch in raw_19ch.ch_names]
            raw_19ch.reorder_channels(available_target)
            return raw_19ch
        
        print(f"Projecting {len(current_channels)}ch to {len(target_channels)}ch layout...")
        print(f"  Adding {len(channels_to_add)} channels via spherical spline: {channels_to_add[:5]}...")
        
        # Add reference channels for interpolation
        raw_19ch = mne.add_reference_channels(raw_19ch, channels_to_add)
        
        # Set montage (new channels will have positions from montage)
        raw_19ch.set_montage(active_montage, on_missing='ignore')
        
        # Interpolate the added channels
        raw_19ch.interpolate_bads(reset_bads=False, method='spline')
        
        # Reorder to target layout
        available_target = [ch for ch in target_channels if ch in raw_19ch.ch_names]
        raw_19ch.reorder_channels(available_target)
        
        return raw_19ch
    
    def harmonize_to_20ch_for_bendr(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Project to 20-channel layout for BENDR model.
        BENDR requires: standard 10-20 + SCALE channel.
        """
        bendr_channels = COMMON_19_CHANNELS + ['SCALE']
        return self.harmonize_to_n_channels(raw, bendr_channels)
    
    def harmonize_to_128ch_for_labram(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Project to 128-channel layout for LaBraM model.
        Uses GSN-HydroCel-128 montage which has exactly 128 channels.
        """
        # Use a 128-channel montage
        montage_128 = make_standard_montage('GSN-HydroCel-128')
        all_128 = montage_128.ch_names[:128]
        
        # First harmonize to 19ch using the 128ch montage for proper positioning
        raw_19ch = self.harmonize_to_19ch(raw, method='interpolation', montage=montage_128)
        
        # Then project to 128ch
        result = self.harmonize_to_n_channels(raw, all_128, montage=montage_128)
        
        return result


class NativeMontagePreserver:
    """Preserve native channel configurations for ablation studies."""
    
    def __init__(self):
        self.channel_stats = {}
        
    def analyze_montage(
        self,
        raw: mne.io.BaseRaw
    ) -> Dict:
        """
        Analyze native montage characteristics.
        
        Returns:
            dict with channel count, positions, density metrics
        """
        n_channels = len(raw.ch_names)
        
        # Get positions if available
        positions = None
        if raw.info['dig']:
            positions = {}
            for i, ch in enumerate(raw.ch_names):
                if i < len(raw.info['chs']):
                    loc = raw.info['chs'][i]['loc'][:3]
                    if np.any(loc):
                        positions[ch] = loc
        
        analysis = {
            'n_channels': n_channels,
            'channel_names': raw.ch_names,
            'positions': positions,
            'montage_type': self._infer_montage_type(n_channels),
        }
        
        return analysis
    
    def _infer_montage_type(self, n_channels: int) -> str:
        """Infer montage type from channel count."""
        if n_channels <= 19:
            return 'clinical_19ch'
        elif n_channels <= 32:
            return 'medium_density'
        elif n_channels <= 64:
            return 'high_density_64ch'
        elif n_channels <= 128:
            return 'high_density_128ch'
        elif n_channels <= 256:
            return 'ultra_high_256ch'
        else:
            return 'unknown'
    
    def compute_channel_density(
        self,
        positions: Dict[str, Tuple[float, float, float]]
    ) -> float:
        """Compute spatial density of channels (channels per unit area)."""
        if len(positions) < 3:
            return 0.0
        
        # Compute convex hull area (2D projection)
        coords = np.array([pos[:2] for pos in positions.values()])
        
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords)
        area = hull.volume  # In 2D, volume is area
        
        density = len(positions) / area if area > 0 else 0
        
        return density


def create_harmonization_report(
    dataset_name: str,
    native_analysis: Dict,
    harmonized_analysis: Dict
) -> Dict:
    """
    Create report comparing native vs harmonized montages.
    
    Args:
        dataset_name: Name of the dataset
        native_analysis: Analysis of native montage
        harmonized_analysis: Analysis of harmonized montage
        
    Returns:
        report: Dictionary with comparison metrics
    """
    report = {
        'dataset': dataset_name,
        'native': {
            'n_channels': native_analysis['n_channels'],
            'montage_type': native_analysis['montage_type'],
        },
        'harmonized': {
            'n_channels': harmonized_analysis['n_channels'],
            'montage_type': 'standard_19ch',
        },
        'interpolation': {
            'channels_added': harmonized_analysis['n_channels'] - len([
                ch for ch in COMMON_19_CHANNELS 
                if ch in native_analysis.get('channel_names', [])
            ]),
            'channels_removed': native_analysis['n_channels'] - harmonized_analysis['n_channels'],
        }
    }
    
    return report


def save_harmonization_config(
    config: Dict,
    output_path: Path
) -> None:
    """Save harmonization configuration to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_harmonization_config(input_path: Path) -> Dict:
    """Load harmonization configuration from JSON."""
    with open(input_path, 'r') as f:
        return json.load(f)