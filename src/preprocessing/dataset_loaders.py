"""
Dataset Loaders for Cross-Disease EEG Analysis

Loaders for:
- ADHD (High-density EEG)
- Seizure (TUH EEG)
- Migraine (OpenNeuro)
- Depression (MODMA)
- Parkinson (Open EEG)

Each loader returns MNE Raw objects with standardized metadata.
"""

import mne
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
from datetime import datetime


class BaseDatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(
        self,
        data_dir: Path,
        disease_name: str,
        target_sfreq: int = 256,
        n_channels: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.disease_name = disease_name
        self.target_sfreq = target_sfreq
        self.n_channels = n_channels
        
    def load_subject(
        self,
        subject_id: str
    ) -> Tuple[mne.io.BaseRaw, Dict]:
        """
        Load a single subject.
        
        Returns:
            raw: MNE Raw object
            metadata: dict with subject info
        """
        raise NotImplementedError
    
    def load_all_subjects(self) -> List[Tuple[mne.io.BaseRaw, Dict]]:
        """Load all available subjects."""
        raise NotImplementedError
    
    def get_available_subjects(self) -> List[str]:
        """Get list of available subject IDs."""
        raise NotImplementedError
    
    def get_channel_names(self) -> List[str]:
        """Get native channel names for this dataset."""
        raise NotImplementedError
    
    def get_metadata_schema(self) -> Dict:
        """Return expected metadata fields."""
        return {
            'subject_id': str,
            'disease': str,
            'age': float,
            'sex': str,
            'site': str,
            'montage': list,
            'n_channels': int,
            'sfreq': float,
            'duration_sec': float,
        }


class ADHDDatasetLoader(BaseDatasetLoader):
    """
    ADHD High-Density EEG Dataset Loader.
    
    Expected structure:
    data/raw/adhd200/
    ├── subject_001/
    │   ├── session_01/
    │   │   └── recording.edf
    │   └── metadata.json
    └── ...
    """
    
    def __init__(self, data_dir: Path, **kwargs):
        super().__init__(
            data_dir=data_dir,
            disease_name='adhd',
            target_sfreq=256,
            n_channels=256  # High-density
        )
        
    def get_available_subjects(self) -> List[str]:
        """Find all subject directories."""
        subjects = []
        for subj_dir in sorted(self.data_dir.iterdir()):
            if subj_dir.is_dir() and subj_dir.name.startswith('sub-'):
                subjects.append(subj_dir.name)
        return subjects
    
    def load_subject(self, subject_id: str) -> Tuple[mne.io.BaseRaw, Dict]:
        """Load ADHD subject data."""
        subj_path = self.data_dir / subject_id
        
        # Find EDF files
        edf_files = list(subj_path.rglob('*.edf'))
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found for {subject_id}")
        
        # Load first recording (or concatenate if multiple)
        raw = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
        
        # Load metadata if available
        metadata = self._load_metadata(subj_path)
        metadata['subject_id'] = subject_id
        metadata['disease'] = self.disease_name
        
        return raw, metadata
    
    def _load_metadata(self, subj_path: Path) -> Dict:
        """Load subject metadata from JSON."""
        meta_file = subj_path / 'metadata.json'
        if meta_file.exists():
            with open(meta_file) as f:
                return json.load(f)
        
        # Default metadata
        return {
            'age': None,
            'sex': 'unknown',
            'site': 'adhd200',
            'montage': self.get_channel_names(),
        }
    
    def get_channel_names(self) -> List[str]:
        """Return typical 256-channel names (10-10 system)."""
        # Standard 256-channel 10-10 system
        return [
            'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
            'O1', 'OZ', 'O2',
            # Additional channels for 256
            'IZ', 'AFp1', 'AFp2', 'HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'HP6',
            'HP7', 'HP8', 'HP9', 'HP10', 'HP11', 'HP12', 'HP13', 'HP14',
            'HP15', 'HP16', 'HP17', 'HP18', 'HP19', 'HP20', 'HP21', 'HP22',
            'HP23', 'HP24', 'HP25', 'HP26', 'HP27', 'HP28', 'HP29', 'HP30',
            'HP31', 'HP32', 'HP33', 'HP34', 'HP35', 'HP36', 'HP37', 'HP38',
            'HP39', 'HP40', 'HP41', 'HP42', 'HP43', 'HP44', 'HP45', 'HP46',
            'HP47', 'HP48', 'HP49', 'HP50', 'HP51', 'HP52', 'HP53', 'HP54',
            'HP55', 'HP56', 'HP57', 'HP58', 'HP59', 'HP60', 'HP61', 'HP62',
            'HP63', 'HP64', 'HP65', 'HP66', 'HP67', 'HP68', 'HP69', 'HP70',
            'HP71', 'HP72', 'HP73', 'HP74', 'HP75', 'HP76', 'HP77', 'HP78',
            'HP79', 'HP80', 'HP81', 'HP82', 'HP83', 'HP84', 'HP85', 'HP86',
            'HP87', 'HP88', 'HP89', 'HP90', 'HP91', 'HP92', 'HP93', 'HP94',
            'HP95', 'HP96', 'HP97', 'HP98', 'HP99', 'HP100', 'HP101', 'HP102',
            'HP103', 'HP104', 'HP105', 'HP106', 'HP107', 'HP108', 'HP109', 'HP110',
            'HP111', 'HP112', 'HP113', 'HP114', 'HP115', 'HP116', 'HP117', 'HP118',
            'HP119', 'HP120', 'HP121', 'HP122', 'HP123', 'HP124', 'HP125', 'HP126',
            'HP127', 'HP128', 'HP129', 'HP130', 'HP131', 'HP132', 'HP133', 'HP134',
            'HP135', 'HP136', 'HP137', 'HP138', 'HP139', 'HP140', 'HP141', 'HP142',
            'HP143', 'HP144', 'HP145', 'HP146', 'HP147', 'HP148', 'HP149', 'HP150',
            'HP151', 'HP152', 'HP153', 'HP154', 'HP155', 'HP156', 'HP157', 'HP158',
            'HP159', 'HP160', 'HP161', 'HP162', 'HP163', 'HP164', 'HP165', 'HP166',
            'HP167', 'HP168', 'HP169', 'HP170', 'HP171', 'HP172', 'HP173', 'HP174',
            'HP175', 'HP176', 'HP177', 'HP178', 'HP179', 'HP180', 'HP181', 'HP182',
            'HP183', 'HP184', 'HP185', 'HP186', 'HP187', 'HP188', 'HP189', 'HP190',
            'HP191', 'HP192', 'HP193', 'HP194', 'HP195', 'HP196', 'HP197', 'HP198',
            'HP199', 'HP200', 'HP201', 'HP202', 'HP203', 'HP204', 'HP205', 'HP206',
            'HP207', 'HP208', 'HP209', 'HP210', 'HP211', 'HP212', 'HP213', 'HP214',
            'HP215', 'HP216', 'HP217', 'HP218', 'HP219', 'HP220', 'HP221', 'HP222',
            'HP223', 'HP224', 'HP225', 'HP226', 'HP227', 'HP228', 'HP229', 'HP230',
            'HP231', 'HP232', 'HP233', 'HP234', 'HP235', 'HP236', 'HP237', 'HP238',
            'HP239', 'HP240', 'HP241', 'HP242', 'HP243', 'HP244', 'HP245', 'HP246',
            'HP247', 'HP248', 'HP249', 'HP250', 'HP251', 'HP252', 'HP253', 'HP254',
            'HP255', 'HP256'
        ][:256]


class SeizureDatasetLoader(BaseDatasetLoader):
    """
    TUH EEG Seizure (TUSZ) Dataset Loader.
    
    Expected structure:
    data/raw/tuh_eeg/
    ├── edf/
    │   ├── train/
    │   │   ├── sub-001/
    │   │   │   └── session-01/
    │   │   │       └── recording.edf
    │   │   └── ...
    │   └── dev/
    └── tsv/
        └── _events.tsv
    """
    
    def __init__(self, data_dir: Path, **kwargs):
        super().__init__(
            data_dir=data_dir,
            disease_name='seizure',
            target_sfreq=256,
            n_channels=19  # Clinical standard
        )
        
    def get_available_subjects(self) -> List[str]:
        """Find all subject directories."""
        subjects = []
        
        # Try TUH structure first (edf/train/sub-xxx)
        edf_dir = self.data_dir / 'edf'
        if edf_dir.exists():
            for split_dir in ['train', 'dev', 'eval']:
                split_path = edf_dir / split_dir
                if split_path.exists():
                    for subj_dir in sorted(split_path.iterdir()):
                        if subj_dir.is_dir():
                            subjects.append(f"{split_dir}/{subj_dir.name}")
        
        # If no subjects found, try simple structure (sub-xxx/sub-xxx.edf)
        if not subjects:
            for subj_dir in sorted(self.data_dir.iterdir()):
                if subj_dir.is_dir() and subj_dir.name.startswith('sub-'):
                    # Check if EDF files exist
                    edf_files = list(subj_dir.rglob('*.edf'))
                    if edf_files:
                        subjects.append(subj_dir.name)
        
        return subjects
    
    def load_subject(self, subject_id: str) -> Tuple[mne.io.BaseRaw, Dict]:
        """Load seizure subject data."""
        # Parse subject ID (may include split)
        if '/' in subject_id:
            split, subj_name = subject_id.split('/')
            subj_path = self.data_dir / 'edf' / split / subj_name
        else:
            subj_path = self.data_dir / subject_id
        
        # Find all EDF files for this subject
        edf_files = list(subj_path.rglob('*.edf'))
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found for {subject_id}")
        
        # Load and concatenate
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) 
                for f in edf_files[:5]]  # Limit to first 5 sessions
        
        raw = mne.concatenate_raws(raws)
        
        metadata = {
            'subject_id': subject_id,
            'disease': self.disease_name,
            'site': 'tuh_eeg',
            'sex': 'unknown',
            'age': None,
            'montage': raw.ch_names,
            'n_sessions': len(edf_files),
        }
        
        return raw, metadata
    
    def get_channel_names(self) -> List[str]:
        """Return standard 19-channel 10-20 system."""
        return [
            'Fp1', 'Fp2',
            'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2'
        ]


class MigraineDatasetLoader(BaseDatasetLoader):
    """
    Migraine OpenNeuro Dataset Loader.
    
    Expected structure (BIDS format):
    data/raw/migraine/
    ├── sub-001/
    │   ├── ses-01/
    │   │   ├── eeg/
    │   │   │   └── sub-001_ses-01_task-rest_eeg.vhdr
    │   │   └── sub-001_ses-01_scans.tsv
    │   └── sub-001_sessions.tsv
    └── dataset_description.json
    """
    
    def __init__(self, data_dir: Path, **kwargs):
        super().__init__(
            data_dir=data_dir,
            disease_name='migraine',
            target_sfreq=256,
            n_channels=64  # Typical for migraine studies
        )
        
    def get_available_subjects(self) -> List[str]:
        """Find all BIDS subject directories."""
        subjects = []
        for subj_dir in sorted(self.data_dir.iterdir()):
            if subj_dir.is_dir() and subj_dir.name.startswith('sub-'):
                subjects.append(subj_dir.name)
        return subjects
    
    def load_subject(self, subject_id: str) -> Tuple[mne.io.BaseRaw, Dict]:
        """Load migraine subject using MNE-BIDS."""
        from mne_bids import BIDSPath, read_raw_bids
        
        bids_path = BIDSPath(
            subject=subject_id.replace('sub-', ''),
            session='01',
            task='rest',
            datatype='eeg',
            root=self.data_dir
        )
        
        try:
            raw = read_raw_bids(bids_path, verbose=False)
        except Exception as e:
            raise FileNotFoundError(f"Could not load {subject_id}: {e}")
        
        # Extract metadata from BIDS sidecar
        metadata = {
            'subject_id': subject_id,
            'disease': self.disease_name,
            'site': 'openneuro_migraine',
            'sex': 'unknown',
            'age': None,
            'montage': raw.ch_names,
        }
        
        return raw, metadata
    
    def get_channel_names(self) -> List[str]:
        """Return typical 64-channel 10-20 system."""
        # Standard 64-channel subset
        return [
            'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
            'O1', 'OZ', 'O2'
        ][:64]


class DepressionDatasetLoader(BaseDatasetLoader):
    """
    Depression Dataset Loader (MODMA or similar).
    
    Expected structure:
    data/raw/depression/
    ├── sub-001/
    │   ├── resting_state.edf
    │   └── metadata.json
    └── ...
    """
    
    def __init__(self, data_dir: Path, **kwargs):
        super().__init__(
            data_dir=data_dir,
            disease_name='depression',
            target_sfreq=256,
            n_channels=64
        )
        
    def get_available_subjects(self) -> List[str]:
        """Find all subject directories."""
        subjects = []
        for subj_dir in sorted(self.data_dir.iterdir()):
            if subj_dir.is_dir() and subj_dir.name.startswith('sub-'):
                subjects.append(subj_dir.name)
        return subjects
    
    def load_subject(self, subject_id: str) -> Tuple[mne.io.BaseRaw, Dict]:
        """Load depression subject data."""
        subj_path = self.data_dir / subject_id
        
        # Find EDF or EEG files
        edf_files = list(subj_path.glob('*.edf'))
        vhdr_files = list(subj_path.glob('*.vhdr'))
        
        if edf_files:
            raw = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
        elif vhdr_files:
            raw = mne.io.read_raw_brainvision(vhdr_files[0], preload=True, verbose=False)
        else:
            raise FileNotFoundError(f"No EEG files found for {subject_id}")
        
        metadata = self._load_metadata(subj_path)
        metadata['subject_id'] = subject_id
        metadata['disease'] = self.disease_name
        
        return raw, metadata
    
    def _load_metadata(self, subj_path: Path) -> Dict:
        """Load subject metadata."""
        meta_file = subj_path / 'metadata.json'
        if meta_file.exists():
            with open(meta_file) as f:
                return json.load(f)
        
        return {
            'age': None,
            'sex': 'unknown',
            'site': 'modma',
            'montage': self.get_channel_names(),
        }
    
    def get_channel_names(self) -> List[str]:
        """Return typical 64-channel names."""
        return MigraineDatasetLoader(self.data_dir).get_channel_names()


class ParkinsonDatasetLoader(BaseDatasetLoader):
    """
    Parkinson's Disease Dataset Loader.
    
    Expected structure:
    data/raw/parkinson/
    ├── sub-001/
    │   ├── recording.edf
    │   └── metadata.json
    └── ...
    """
    
    def __init__(self, data_dir: Path, **kwargs):
        super().__init__(
            data_dir=data_dir,
            disease_name='parkinson',
            target_sfreq=256,
            n_channels=19  # Clinical standard
        )
        
    def get_available_subjects(self) -> List[str]:
        """Find all subject directories."""
        subjects = []
        for subj_dir in sorted(self.data_dir.iterdir()):
            if subj_dir.is_dir() and subj_dir.name.startswith('sub-'):
                subjects.append(subj_dir.name)
        return subjects
    
    def load_subject(self, subject_id: str) -> Tuple[mne.io.BaseRaw, Dict]:
        """Load Parkinson's subject data."""
        subj_path = self.data_dir / subject_id
        
        edf_files = list(subj_path.glob('*.edf'))
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found for {subject_id}")
        
        raw = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
        
        metadata = self._load_metadata(subj_path)
        metadata['subject_id'] = subject_id
        metadata['disease'] = self.disease_name
        
        return raw, metadata
    
    def _load_metadata(self, subj_path: Path) -> Dict:
        """Load subject metadata."""
        meta_file = subj_path / 'metadata.json'
        if meta_file.exists():
            with open(meta_file) as f:
                return json.load(f)
        
        return {
            'age': None,
            'sex': 'unknown',
            'site': 'open_eeg',
            'montage': self.get_channel_names(),
        }
    
    def get_channel_names(self) -> List[str]:
        """Return standard 19-channel names."""
        return SeizureDatasetLoader(self.data_dir).get_channel_names()


# Dataset registry with directory name mappings
DATASET_LOADERS = {
    'adhd': ADHDDatasetLoader,
    'adhd200': ADHDDatasetLoader,  # Alias for directory name
    'seizure': SeizureDatasetLoader,
    'tuh_eeg': SeizureDatasetLoader,  # Alias for directory name
    'migraine': MigraineDatasetLoader,
    'depression': DepressionDatasetLoader,
    'parkinson': ParkinsonDatasetLoader,
}

# Directory name to disease name mapping
DIRECTORY_TO_DISEASE = {
    'adhd200': 'adhd',
    'tuh_eeg': 'seizure',
    'chb_mit': 'seizure',
    'helsinki_neonatal': 'seizure',
    'sleep_edf': 'seizure',  # Map to seizure loader for testing
}


def get_dataset_loader(
    disease: str,
    data_dir: Path,
    **kwargs
) -> BaseDatasetLoader:
    """
    Get appropriate dataset loader for a disease.
    
    Args:
        disease: disease name or directory name (adhd, adhd200, seizure, tuh_eeg, etc.)
        data_dir: root data directory
        **kwargs: additional arguments for loader
        
    Returns:
        loader: dataset loader instance
    """
    # Check if it's a directory name that needs mapping
    if disease in DIRECTORY_TO_DISEASE:
        disease = DIRECTORY_TO_DISEASE[disease]
    
    if disease not in DATASET_LOADERS:
        raise ValueError(f"Unknown disease: {disease}. "
                        f"Available: {list(DATASET_LOADERS.keys())}")
    
    loader_class = DATASET_LOADERS[disease]
    return loader_class(data_dir, **kwargs)


def create_dataset_catalog(
    data_root: Path,
    output_file: Path
) -> pd.DataFrame:
    """
    Create catalog of all available datasets.
    
    Args:
        data_root: root directory containing all raw data
        output_file: path to save catalog CSV
        
    Returns:
        catalog: DataFrame with dataset information
    """
    records = []
    
    for disease, loader_class in DATASET_LOADERS.items():
        data_dir = data_root / 'raw' / disease
        
        if not data_dir.exists():
            continue
        
        try:
            loader = loader_class(data_dir)
            subjects = loader.get_available_subjects()
            
            for subj in subjects[:5]:  # Sample first 5 subjects
                try:
                    raw, meta = loader.load_subject(subj)
                    records.append({
                        'disease': disease,
                        'subject_id': subj,
                        'n_channels': len(raw.ch_names),
                        'sfreq': raw.info['sfreq'],
                        'duration_sec': raw.times[-1],
                        'n_samples': raw.n_times,
                        'site': meta.get('site', 'unknown'),
                        'age': meta.get('age'),
                        'sex': meta.get('sex', 'unknown'),
                    })
                except Exception as e:
                    print(f"Error loading {subj}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error scanning {disease}: {e}")
            continue
    
    catalog = pd.DataFrame(records)
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(output_file, index=False)
    
    return catalog