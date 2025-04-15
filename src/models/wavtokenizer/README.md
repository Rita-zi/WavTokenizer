# WavTokenizer Model Structure

## Directory Organization

```
src/models/wavtokenizer/
├── tsne_model.py      # Main model implementation (moved from tsne.py)
├── dataset.py         # Dataset handling logic (from try3.py)
└── utils/
    └── visualization.py  # Training visualization (from add.py)

data/
├── raw/               # Raw audio files
│   ├── box/          # Box material recordings
│   ├── plastic/      # Plastic material recordings
│   └── papercup/     # Paper cup material recordings
├── clean/            # Clean target audio (from box2/)
└── processed/        # Processed data files

outputs/
├── enhanced/         # Enhanced audio outputs
├── tsne/            # TSNE analysis results
└── visualization/    # Training visualizations
```

## Key Components

1. **Core Model Files**
   - `tsne_model.py`: Main model implementation
   - `dataset.py`: Audio dataset handling
   - `visualization.py`: Training monitoring tools

2. **Data Organization**
   - Raw audio files organized by material type
   - Clean target audio files in separate directory
   - Clear separation of input and target data

3. **Output Organization**
   - Enhanced audio outputs
   - TSNE analysis results
   - Training visualizations and metrics

## Usage

1. **Training**
```bash
python src/models/wavtokenizer/train.py --config config/model_config.yaml
```

2. **Testing**
```bash
python src/models/wavtokenizer/test.py --model_path checkpoints/best_model.pth
```

3. **Visualization**
```bash
python src/models/wavtokenizer/utils/visualization.py --log_dir outputs/training_logs
```