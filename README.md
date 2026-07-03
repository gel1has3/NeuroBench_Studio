# NeuroBench Studio — EEG Foundation Model Platform

A comprehensive platform for EEG foundation model research, providing tools for **dataset exploration**, **braindecode model discovery**, and **visual pipeline building** for both coders and non-coders.

## Core Mission

> Democratize EEG deep learning by providing a unified platform that combines EEGDash dataset streaming, Braindecode foundation models, and an intuitive MLOps pipeline builder accessible to neuroscientists, clinicians, and machine learning researchers.

## Platform Features

### 1. EEG Dataset Explorer
- Browse and explore curated local EEG datasets
- Stream remote OpenNeuro datasets via **EEGDash** integration
- View subject metadata, channel configurations, and data quality metrics
- Auto-configure braindecode models based on dataset characteristics

### 2. Braindecode Model Discoverer
- Dynamically discover all models available in `braindecode.models`
- Inspect model parameters, architectures, and default values
- Generate ready-to-use Python code snippets
- Filter models by type (Transformer, CNN, Sleep, etc.)

### 3. Visual MLOps Pipeline Builder
- Drag-and-drop interface for building EEG ML pipelines
- No coding required — accessible to non-programmers
- Configure preprocessing steps, model selection, and validation strategies
- Execute pipelines with real-time progress streaming
- Suitable for both rapid prototyping and production workflows

### 4. Foundation Model Reference
- Comprehensive documentation of all 9+ foundation models available via braindecode v1.5+
- Models include: REVE, CBraMod, CodeBrain, EEGPT, BIOT, LaBraM, BENDR, SignalJEPA, LUNA
- Pretrained on large-scale EEG data for fine-tuning or feature extraction

## Architecture Overview

```
NeuroBench_Studio/
├── src/
│   ├── dashboard/              # Flask web application
│   │   ├── flask_app.py        # Main Flask server
│   │   ├── pipeline_executor.py # Pipeline execution engine
│   │   ├── templates/          # HTML templates
│   │   │   ├── base.html            # Base layout with navigation
│   │   │   ├── datasets_explorer.html  # Dataset browsing
│   │   │   ├── braindecode_explorer.html # Model discovery
│   │   │   ├── pipeline_builder.html  # Visual pipeline builder
│   │   │   ├── about.html             # About page
│   │   │   ├── index.html             # Landing page
│   │   │   ├── experiment.html        # Experiment detail
│   │   │   └── error.html             # Error page
│   │   └── static/             # CSS, JS assets
│   ├── models/                 # Brain decoding models
│   ├── preprocessing/          # EEG preprocessing pipeline
│   ├── evaluation/             # Evaluation metrics & analysis
│   └── reporting/              # Figure & table generation
├── data/
│   ├── raw/                    # Local EEG datasets
│   └── preprocessed/           # Preprocessed MNE epochs
├── results/                    # Training results & checkpoints
├── configs/                    # YAML configuration files
└── docs/                       # Documentation
```

## Quick Start

### 1. Install Dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Platform

```bash
# Start the Flask web application
python src/dashboard/flask_app.py

# Open in your browser
# → http://localhost:5000
#   - /datasets      → Explore EEG datasets
#   - /braindecode    → Discover braindecode models
#   - /pipeline       → Visual pipeline builder
#   - /about          → About NeuroBench Studio
```

### 3. Explore the Menus

| Menu | Path | Description |
|------|------|-------------|
| 🧠 Datasets | `/datasets` | Browse local datasets & stream from EEGDash |
| 🧠 Braindecode Models | `/braindecode` | Discover & configure braindecode models |
| 🧠 Pipeline Builder | `/pipeline` | Drag-and-drop ML pipeline builder |
| 🧠 About | `/about` | Platform information & tech stack |

## Dataset Explorer

The Datasets page provides two data sources:

### Local Datasets
- Scans `data/raw/` directory for locally stored EEG datasets
- Displays subject count, channel configurations, sampling rates
- Supports ADHD-200, TUH EEG, CHB-MIT, Helsinki Neonatal, Sleep EDF

### EEGDash Remote Streaming
- Connect to OpenNeuro datasets via the EEGDash API
- Stream metadata and recordings on-demand
- Auto-configure braindecode models based on dataset parameters
- Cached locally for fast re-access

## Braindecode Model Explorer

The Braindecode Models page automatically discovers all models from the `braindecode.models` package:

- **Automatic Discovery**: Scans installed braindecode version for all available models
- **Parameter Inspection**: View required and optional parameters for each model
- **Code Generation**: Get ready-to-use Python instantiation code
- **Filtering**: Filter by architecture type (Transformer, CNN, Sleep, etc.)

## Pipeline Builder

The visual MLOps pipeline builder allows both coders and non-coders to create EEG ML pipelines:

- **Drag-and-Drop Interface**: Add dataset, preprocessing, model, and evaluation nodes
- **Connection Mapping**: Visually connect nodes to define data flow
- **Preprocessing Pipeline**: Configure filtering, downsampling, ICA, segmentation, and more
- **Model Selection**: Choose from all available braindecode architectures
- **Validation Strategy**: Configure data splitting and evaluation metrics
- **Execution**: Run pipelines with real-time progress tracking

### Pipeline Nodes

| Node Type | Description |
|-----------|-------------|
| Dataset | Select local or remote EEG dataset |
| Preprocessing | Filtering, downsampling, bad channel detection, re-referencing, ICA, segmentation, baseline correction |
| Data Splitting | Train/val/test split or leave-one-subject-out |
| Model | Braindecode model selection with auto-configuration |
| Validation | Evaluation metrics and statistical testing |

## REST API

The platform provides a full REST API for programmatic access:

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/datasets` | List all local datasets |
| `GET /api/datasets/<name>` | Dataset details |
| `GET /api/eegdash/catalog` | EEGDash dataset catalog |
| `POST /api/eegdash/connect` | Stream EEGDash dataset |
| `GET /api/braindecode/models` | List all braindecode models |
| `GET /api/braindecode/models/<name>` | Model parameter details |
| `POST /api/run-pipeline` | Execute a pipeline |
| `GET /api/pipeline/progress/<id>` | Pipeline progress stream |

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | Flask (Python) |
| Frontend | Bootstrap 5.3, Plotly.js |
| EEG Processing | MNE-Python, Braindecode |
| Deep Learning | PyTorch |
| Dataset Streaming | EEGDash, OpenNeuro |

## Dependencies

See `requirements.txt` for the full list of dependencies.

## License

MIT License

## Citation

```bibtex
@software{NeuroBenchStudio,
  title={NeuroBench Studio: Vibe Coding for EEG — A No-Code Platform for EEG Deep Learning, Foundation Models, and Visual Pipeline Orchestration},
  author={Geletaw Sahle Tegenaw and Tomas Ward},
  year={2026},
  url={https://github.com/your-repo/NeuroBenchStudio}
}