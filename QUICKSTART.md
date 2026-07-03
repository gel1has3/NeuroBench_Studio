# NeuroBench Studio Quick Start Guide

## Setup Complete! ✓

Your NeuroBench Studio platform is now fully configured with:
- ✅ Flask web application (primary interface)
- ✅ Dataset explorer with local & EEGDash streaming
- ✅ Braindecode model discovery & parameter inspection
- ✅ Visual MLOps pipeline builder (no coding required)
- ✅ All modules verified and importable

## Launch the Platform

```bash
# Start the Flask web application
python src/dashboard/flask_app.py

# Open in your browser
# → http://localhost:5000
```

The platform provides four main menus:

| Menu | Path | Description |
|------|------|-------------|
| 📊 Datasets | `/datasets` | Browse local datasets & stream from EEGDash |
| 🧠 Braindecode Models | `/braindecode` | Discover & configure braindecode models |
| 🔧 Pipeline Builder | `/pipeline` | Drag-and-drop ML pipeline builder |
| ℹ️ About | `/about` | Platform information & tech stack |

## Dataset Explorer

### Local Datasets
1. Navigate to **Datasets** (`/datasets`)
2. Click on a dataset in the left sidebar
3. View subject metadata, channel configurations, and data quality

### EEGDash Remote Streaming
1. Switch to **EEG Dash Datasets** tab
2. Select a dataset from the curated dropdown
3. Click **Stream** to connect and fetch metadata
4. View auto-configured braindecode model parameters

## Braindecode Model Explorer

1. Navigate to **Braindecode Models** (`/braindecode`)
2. Browse the model list (filterable by type)
3. Click any model to inspect its parameters
4. Copy ready-to-use Python code snippets

## Pipeline Builder

1. Navigate to **Pipeline Builder** (`/pipeline`)
2. Drag nodes onto the canvas:
   - **Dataset**: Select your EEG data source
   - **Preprocessing**: Configure filtering, downsampling, ICA, etc.
   - **Data Splitting**: Choose train/val/test strategy
   - **Model**: Select braindecode architecture
   - **Validation**: Configure evaluation metrics
3. Connect nodes to define the data flow
4. Click **Run Experiment** to execute

## Example: Train with Pretraining Script

```bash
# Run pretraining (use in separate terminal)
python run_pretraining.py \
  --config configs/pretraining_config.yaml \
  --output-dir results/pretraining \
  --diseases adhd200 tuh_eeg \
  --model mae \
  --n-epochs 10 \
  --batch-size 2
```

## Example: Channel Harmonization

```python
import mne
from src.preprocessing.channel_harmonization import ChannelHarmonizer

# Load raw EEG
raw = mne.io.read_raw_edf('data/raw/adhd200/sub-001/recording.edf', preload=True)

# Harmonize to 19-channel common space
harmonizer = ChannelHarmonizer()
raw_19ch = harmonizer.harmonize_to_19ch(raw, method='interpolation')

print(f"Original channels: {len(raw.ch_names)}")
print(f"Harmonized channels: {len(raw_19ch.ch_names)}")
```

## Troubleshooting

### Dashboard won't start
```bash
# Make sure Flask is installed
pip install flask

# Check for port conflicts
lsof -i :5000
```

### EEGDash connection fails
- Ensure `eegdash` package is installed: `pip install eegdash`
- Check internet connection for remote streaming

### Pipeline execution errors
- Ensure dataset node is connected to model node
- Check dataset availability in `data/raw/`

## Key Files

- `src/dashboard/flask_app.py` - Main Flask application
- `src/dashboard/pipeline_executor.py` - Pipeline execution engine
- `src/dashboard/templates/` - HTML templates for UI
- `configs/pretraining_config.yaml` - Training configuration
- `run_pretraining.py` - Main pretraining script

## Next Steps

1. **Explore Datasets** → Browse available EEG data
2. **Discover Models** → Inspect braindecode architectures
3. **Build Pipelines** → Create and run ML pipelines
4. **Train Models** → Run pretraining for custom experiments

Happy exploring! 🧠