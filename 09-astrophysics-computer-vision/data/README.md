# Data Directory

This directory contains astronomical data for galaxy and transient classification.

## Datasets Used

### 1. Galaxy Zoo 2 (Galaxy Morphology Classification)
- **Source**: [Kaggle Galaxy Zoo](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
- **Description**: 61,578 galaxy images from SDSS with morphological labels
- **Image Size**: 424×424 pixels (RGB)
- **Labels**: Vote fractions for 37 morphological questions

### 2. PLAsTiCC (Transient Classification)
- **Source**: [Kaggle PLAsTiCC](https://www.kaggle.com/c/PLAsTiCC-2018)
- **Description**: Simulated light curves from LSST
- **Features**: Time-series flux measurements in 6 passbands (ugrizy)
- **Classes**: 14 transient/variable types

## Directory Structure

```
data/
├── raw/              # Original downloaded data
│   ├── galaxy_zoo/   # Galaxy images
│   └── plasticc/     # Light curve data
├── processed/        # Preprocessed data ready for training
│   ├── galaxy/       # Processed galaxy images
│   └── transients/   # Processed light curves
└── sample/           # Small sample for demos
```

## Download Instructions

1. Set up Kaggle API credentials:
   ```bash
   # Create ~/.kaggle/kaggle.json with your API token
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. Run the download script:
   ```bash
   python scripts/download_data.py --dataset all
   ```

## Data Sizes

| Dataset | Raw Size | Processed Size |
|---------|----------|----------------|
| Galaxy Zoo 2 | ~4 GB | ~1 GB |
| PLAsTiCC | ~500 MB | ~200 MB |
