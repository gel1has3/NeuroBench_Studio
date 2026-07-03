# NeuroBench Studio Platform Guide

## Quick Start

### Start the Platform

Open a terminal and run:

```bash
cd NeuroBench_Studio
source venv/bin/activate
python src/dashboard/flask_app.py
```

The platform will be available at **http://localhost:5000**

### Main Pages

| Page | URL | Description |
|------|-----|-------------|
| 🏠 Home | `/` | Control center dashboard |
| 📊 Datasets | `/datasets` | Explore local EEG datasets & stream from EEGDash |
| 🧠 Models | `/braindecode` | Discover and inspect braindecode models |
| 🔧 Pipeline Builder | `/pipeline` | Visual drag-and-drop pipeline builder |
| 📊 Results | `/results` | Run history, metrics, and benchmark hub |
| 🤖 EEG Assistant | `/assistant` | AI assistant for EEG analysis |
| ℹ️ About | `/about` | Platform information |

The platform **auto-refreshes every 30 seconds** (configurable via `REFRESH_INTERVAL` env var).

---

## Home Dashboard

The Home page serves as the control center:

- **Quick Actions**: New Pipeline, Import Dataset, Browse Models, View Results
- **Recent Projects**: Table showing last 5 experiments with status
- **System Status**: API server, GPU availability, storage info
- **Results Snapshot**: Best model, accuracy, completed runs
- **Latest Checkpoints**: Recent model checkpoints

---

## Datasets Explorer

### Local Datasets
- Scans `data/raw/` for locally stored EEG datasets
- Displays dataset statistics: subjects, channels, sampling rates
- Supports: adhd200, tuh_eeg, chb_mit, helsinki_neonatal, sleep_edf
- Click any dataset to view subject details and channel configurations

### EEGDash Remote Streaming
- Browse a curated catalog of OpenNeuro datasets
- Select a dataset from the dropdown menu
- Click **Stream** to connect and fetch metadata via the EEGDash API
- View auto-configured braindecode model parameters
- Results are cached locally for fast re-access

---

## Braindecode Model Explorer

Automatically discovers all models from `braindecode.models`:

- **Model List**: Sorted sidebar with parameter counts
- **Search**: Filter models by name
- **Type Filters**: All, Transformer, CNN, Sleep, EEG*, Signal
- **Detail Panel**: Shows parameters, types, default values
- **Code Generation**: Ready-to-use Python snippets for instantiation

---

## Pipeline Builder

Visual MLOps pipeline builder with drag-and-drop interface:

### Nodes
| Node Type | Color | Function |
|-----------|-------|----------|
| Dataset | Blue | Select local/remote EEG data source |
| Preprocessing | Orange | 7-step preprocessing pipeline |
| Data Splitting | Pink | Train/val/test split configuration |
| Validation | Blue | Evaluation metrics & statistical tests |
| Model | Purple | Braindecode architecture selection |

### Features
- **Drag & Drop**: Add nodes from palette to canvas
- **Connect**: Link nodes by clicking output/input ports
- **Configure**: Edit node properties via modal dialog
- **Run**: Execute pipeline with real-time progress
- **Results**: View metrics and summary after completion
- **Default Pipeline**: Auto-generate a complete pipeline

### Preprocessing Steps
1. Temporal Filtering (high-pass, low-pass, notch)
2. Downsampling
3. Bad Channel Detection
4. Re-Referencing (CAR, REST, Average)
5. ICA Artifact Removal
6. Segmentation (windowing)
7. Baseline Correction

---

## Results & Benchmark Hub

### Run History
- Table of all experiments with dataset, model, validation strategy
- Status badges (Success, Running, Pending)
- Quick links to detailed results

### Metrics Dashboard
- Model performance summaries
- Average dimensionality metrics
- Disease-level breakdowns

### Benchmark Table
- Global comparison leaderboard
- Filter by dataset, model, validation method
- Rank pipelines by performance metrics

### Reports
- Export paper-ready reports (coming soon)
- Methods section auto-generation
- Reproducibility checklist

---

## EEG Assistant

Context-aware AI assistant for EEG research:

### Features
- **Chat Interface**: Ask questions in natural language
- **EEG Upload**: Upload .edf, .csv, .mat, .fif files for analysis
- **Analysis Types**: Full analysis, waveform, spectrogram, quality check, AI interpretation
- **Quick Actions**: Preprocessing help, model recommendations, results explanation, debug training
- **Context-Aware Suggestions**: Pipeline help, results explanation, EEG interpretation, research mode

### Example Questions
- "What preprocessing steps should I use for motor imagery?"
- "Which model works best for sleep stage classification?"
- "Why is my F1 score lower than accuracy?"
- "My loss is NaN after epoch 5, what's wrong?"

---

## REST API

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/refresh` | Clear server cache |
| `GET /api/datasets` | List all local datasets |
| `GET /api/datasets/<name>` | Dataset details |
| `GET /api/eegdash/catalog` | EEGDash catalog |
| `POST /api/eegdash/connect` | Stream EEGDash dataset |
| `GET /api/braindecode/models` | List braindecode models |
| `GET /api/braindecode/models/<name>` | Model parameters |
| `GET /api/experiments` | List experiments |
| `GET /api/experiments/<name>/summary` | Experiment summary |
| `POST /api/run-pipeline` | Execute pipeline |
| `GET /api/pipeline/progress/<id>` | Pipeline progress |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_HOST` | `0.0.0.0` | Bind address |
| `FLASK_PORT` | `5000` | Port number |
| `FLASK_DEBUG` | `false` | Debug mode |
| `FLASK_SECRET_KEY` | `dev-key-...` | Secret key |
| `RESULTS_DIR` | `results/` | Results directory |
| `REFRESH_INTERVAL` | `30` | Auto-refresh seconds |

### Custom settings:
```bash
FLASK_PORT=8080 REFRESH_INTERVAL=15 python src/dashboard/flask_app.py
```

---

## Troubleshooting

### Platform won't start
```bash
# Make sure Flask is installed
pip install flask

# Check for port conflicts
lsof -i :5000
```

### No datasets showing
- Check that `data/raw/` contains EEG dataset directories
- For EEGDash streaming, ensure `eegdash` is installed: `pip install eegdash`
- Verify internet connection for remote streaming

### Port already in use
```bash
# Use a different port
FLASK_PORT=8080 python src/dashboard/flask_app.py
```

### Charts not rendering
- Ensure internet connection (Plotly.js loads from CDN)
- Check browser console for errors

---

## Tips

1. **Keep platform running**: Start it once and leave it open
2. **Use the Pipeline Builder**: No coding needed for ML pipelines
3. **Stream from EEGDash**: Access thousands of OpenNeuro datasets
4. **Explore models**: Filter braindecode models by type
5. **Copy code snippets**: Get instant Python code for any model
6. **Use EEG Assistant**: Upload files for quick analysis and interpretation

---

## Next Steps

1. **Home Dashboard** → Overview of projects and system status
2. **Explore Datasets** → Browse local and remote EEG data
3. **Discover Models** → Inspect braindecode architectures
4. **Build Pipelines** → Create and run ML pipelines visually
5. **View Results** → Benchmark and compare experiments
6. **EEG Assistant** → Get AI-powered help and analysis

Happy exploring! 🧠