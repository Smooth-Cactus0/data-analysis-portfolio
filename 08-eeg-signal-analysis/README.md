# Project 8: EEG Signal Analysis & Motor Imagery Classification

<div align="center">

**Brain-Computer Interface (BCI) Analysis using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![MNE](https://img.shields.io/badge/MNE-1.5+-green.svg)](https://mne.tools)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

This project demonstrates a complete **EEG (Electroencephalography) signal analysis pipeline** for Brain-Computer Interface (BCI) applications. We analyze brain signals recorded during motor imagery tasks (imagining left/right hand movements) and build classifiers to decode user intent.

### Key Objectives

1. **Preprocess** raw EEG signals (filtering, artifact removal, epoching)
2. **Visualize** brain activity patterns (topomaps, ERPs, time-frequency)
3. **Extract features** from multiple domains (time, frequency, spatial)
4. **Classify** motor imagery using classical ML and deep learning
5. **Predict** neural states using sequence models

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset**

| Property | Value |
|----------|-------|
| **Source** | [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) |
| **Subjects** | 109 volunteers |
| **Channels** | 64 EEG electrodes (10-20 system) |
| **Sampling Rate** | 160 Hz |
| **Tasks** | Motor imagery (left/right hand), Real movement, Rest |
| **Trials** | ~45 per task per subject |

### Tasks Analyzed

- **T1:** Left hand motor imagery
- **T2:** Right hand motor imagery
- **T0:** Rest baseline

---

## Project Structure

```
08-eeg-signal-analysis/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── app.py                             # Streamlit demo application
├── data/
│   ├── README.md                      # Dataset documentation
│   ├── raw/                           # Downloaded EEG files (.edf)
│   ├── processed/                     # Preprocessed epochs
│   └── sample/                        # Demo subset
├── notebooks/
│   └── eeg_motor_imagery_analysis.ipynb  # Main analysis notebook
├── scripts/
│   └── download_data.py               # Data download script
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── preprocessing.py               # EEG preprocessing functions
│   ├── features.py                    # Feature extraction
│   ├── models.py                      # ML and DL models
│   └── visualization.py               # Plotting utilities
├── models/                            # Saved model files
└── images/                            # Generated visualizations
```

---

## Methodology

### 1. Preprocessing Pipeline

```
Raw EEG → Bandpass Filter (1-40 Hz) → Notch Filter → Re-reference → ICA → Epochs
```

- **Bandpass filtering:** Remove drift (<1 Hz) and high-frequency noise (>40 Hz)
- **Notch filter:** Remove power line interference (50/60 Hz)
- **Re-referencing:** Common average reference
- **ICA:** Remove eye blink and muscle artifacts
- **Epoching:** Extract time-locked segments around events

### 2. Feature Extraction

| Domain | Features | Description |
|--------|----------|-------------|
| **Time** | Mean, Std, Variance, RMS, Skewness, Kurtosis | Statistical measures per channel |
| **Frequency** | Band power (delta, theta, alpha, beta, gamma) | Power in frequency bands |
| **Spatial** | CSP patterns | Common Spatial Patterns for class discrimination |
| **Spectral** | Peak frequency, spectral entropy, edge frequency | PSD-derived features |

### 3. Classification Models

#### Classical Machine Learning
- **LDA:** Linear Discriminant Analysis
- **SVM:** Support Vector Machine (RBF & Linear kernels)
- **Random Forest:** Ensemble decision trees
- **XGBoost:** Gradient boosting

#### Deep Learning
- **EEGNet:** Compact CNN designed for EEG (Lawhern et al., 2018)
- **ShallowConvNet:** Shallow CNN for oscillatory patterns
- **DeepConvNet:** Deep CNN for hierarchical features

#### Neural State Prediction
- **LSTM:** Long Short-Term Memory networks
- **Transformer:** Self-attention based sequence model

---

## Results

### Classification Performance (Subject 1)

| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| LDA | 0.72 | 0.68 |
| SVM-RBF | 0.74 | 0.71 |
| Random Forest | 0.71 | 0.69 |
| **CSP + LDA** | **0.78** | **0.75** |

### Multi-Subject Analysis

![Subject Variability](images/subject_variability.png)

*Classification accuracy varies significantly across subjects, highlighting the need for subject-specific calibration.*

---

## Visualizations

### Power Spectral Density
![PSD Comparison](images/psd_comparison.png)
*Alpha (8-13 Hz) and beta (13-30 Hz) band differences between left and right hand motor imagery.*

### Event-Related Potentials
![ERP](images/erp_comparison.png)
*Time-locked brain responses at motor cortex channels (C3, Cz, C4).*

### Topographic Maps
![Topomaps](images/topomaps.png)
*Spatial distribution of activity during motor imagery tasks.*

### Common Spatial Patterns
![CSP](images/csp_patterns.png)
*Learned spatial filters that maximize class separability.*

---

## Installation

### Prerequisites

- Python 3.8+
- 4+ GB RAM recommended
- GPU optional (for deep learning)

### Setup

```bash
# Clone repository
cd 08-eeg-signal-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Download EEG data for 10 subjects
python scripts/download_data.py

# Download for specific number of subjects
python scripts/download_data.py --subjects 5
```

---

## Usage

### Run the Analysis Notebook

```bash
jupyter notebook notebooks/eeg_motor_imagery_analysis.ipynb
```

### Launch Streamlit Demo

```bash
streamlit run app.py
```

The demo allows you to:
- Visualize raw EEG signals
- Explore frequency content (PSD)
- View topographic maps
- Train and compare classifiers
- Interact with real-time predictions

### Quick Start (Python)

```python
from src.preprocessing import preprocess_pipeline
from src.features import extract_features_from_epochs
from src.models import train_classical_models

# Load and preprocess data
epochs, info = preprocess_pipeline(
    subject=1,
    runs=[4, 8, 12],  # Motor imagery
    l_freq=1.0,
    h_freq=40.0
)

# Extract features
X, y, feature_names = extract_features_from_epochs(epochs)

# Train models
results = train_classical_models(X, y)
print(results)
```

---

## Key Insights

### Neuroscience Background

**Motor imagery** engages similar brain regions as actual movement, producing characteristic patterns:

1. **Event-Related Desynchronization (ERD):** Decrease in alpha/beta power over motor cortex during imagery
2. **Contralateral organization:** Left hand imagery activates right motor cortex (C4), and vice versa
3. **Mu rhythm:** 8-13 Hz oscillations over sensorimotor areas, suppressed during motor tasks

### Technical Insights

1. **CSP is highly effective** for motor imagery, achieving ~75-80% accuracy
2. **Frequency features** (especially alpha/beta band power) are most discriminative
3. **Subject variability** is significant - personalized models outperform generic ones
4. **Deep learning** requires more data but can capture complex temporal patterns

---

## References

1. **Dataset:** Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004). BCI2000: A General-Purpose Brain-Computer Interface System. *IEEE TBME*.

2. **EEGNet:** Lawhern, V.J., et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. *Journal of Neural Engineering*.

3. **CSP:** Blankertz, B., et al. (2008). Optimizing Spatial Filters for Robust EEG Single-Trial Analysis. *IEEE Signal Processing Magazine*.

4. **MNE-Python:** Gramfort, A., et al. (2013). MEG and EEG Data Analysis with MNE-Python. *Frontiers in Neuroscience*.

---

## Author

**Alexy Louis**

*Data Analyst & Machine Learning Engineer*

---

## License

This project is licensed under the MIT License. The PhysioNet dataset is available under the PhysioNet Open Data License.
