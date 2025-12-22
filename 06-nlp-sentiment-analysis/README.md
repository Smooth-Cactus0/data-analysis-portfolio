# NLP Sentiment Analysis & Text Classification

A comprehensive NLP project demonstrating sentiment analysis from classical ML through deep learning to transformers.

**Author:** Alexy Louis  
**Email:** alexy.louis.scholar@gmail.com  
**LinkedIn:** [Alexy Louis](https://www.linkedin.com/in/alexy-louis-19a5a9262/)

## Project Overview

This project implements sentiment classification on movie reviews, progressively building from simple models to state-of-the-art transformers:

| Phase | Models | Status |
|-------|--------|--------|
| Classical ML | Logistic Regression, Naive Bayes, SVM, Random Forest | ✅ Complete |
| Deep Learning | LSTM, BiLSTM, CNN, CNN+LSTM | ✅ Complete |
| Transformers | DistilBERT, BERT | 📋 Planned |
| Demo App | Gradio/Streamlit | 📋 Planned |

## Results Summary

### Classical ML (Baseline)
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Linear SVM | 86.3% | 0.863 |
| Logistic Regression | 86.2% | 0.862 |
| Complement NB | 86.1% | 0.861 |

*Note: ~86% accuracy is realistic for IMDB-like sentiment classification with classical approaches.*

### Deep Learning
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| CNN | 87.7% | 0.877 |
| BiLSTM | 87.7% | 0.877 |

*Deep learning models outperform classical ML by ~1.5%. CNN and BiLSTM achieve comparable results with the CNN being faster to train.*

## Project Structure

```
nlp-sentiment-analysis/
├── data/
│   └── sample/              # Small sample for demos
├── images/                  # Visualizations (7 plots)
├── models/                  # Trained models (git-ignored)
├── notebooks/               # Jupyter notebooks
├── scripts/
│   ├── synthetic_data.py        # Data generator
│   ├── train_classical_ml.py    # Classical ML training
│   └── train_deep_learning.py   # Deep learning training
├── src/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/Smooth-Cactus0/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Train classical ML models
python scripts/train_classical_ml.py

# Train deep learning models
python scripts/train_deep_learning.py
```

## Dataset

Uses synthetic movie reviews designed to mimic IMDB characteristics:
- 25,000 training / 25,000 test samples
- Binary sentiment (positive/negative)
- Realistic difficulty through ambiguous vocabulary and negation patterns

## Visualizations

### Classical ML
| Model Comparison | Confusion Matrix | ROC Curves |
|------------------|------------------|------------|
| ![Model Comparison](images/01_model_comparison.png) | ![Confusion Matrix](images/02_confusion_matrix.png) | ![ROC Curves](images/03_roc_curves.png) |

### Deep Learning
| Training History | Model Comparison |
|------------------|------------------|
| ![Training History](images/04_dl_training_history.png) | ![DL Comparison](images/05_dl_model_comparison.png) |

| Confusion Matrix | ROC Curves |
|------------------|------------|
| ![DL Confusion Matrix](images/06_dl_confusion_matrix.png) | ![DL ROC Curves](images/07_dl_roc_curves.png) |

## License

MIT License
