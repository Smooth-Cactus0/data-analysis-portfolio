# NLP Sentiment Analysis & Text Classification

A comprehensive NLP project demonstrating sentiment analysis progression from classical machine learning through deep learning to transformer-based models.

**Author:** Alexy Louis
**Email:** alexy.louis.scholar@gmail.com
**LinkedIn:** [Alexy Louis](https://www.linkedin.com/in/alexy-louis-19a5a9262/)

---

## The Problem: Understanding Sentiment in Text

Sentiment analysis is the task of automatically determining whether a piece of text expresses a positive, negative, or neutral opinion. It's one of the most common NLP applications with real-world uses in:

- **Customer feedback analysis** - Automatically categorizing product reviews
- **Social media monitoring** - Tracking brand perception at scale
- **Market research** - Gauging public opinion on products or topics
- **Content moderation** - Flagging potentially negative or harmful content

The challenge lies in the complexity of human language: sarcasm, negation ("not bad" = positive), context-dependent meaning, and varying writing styles all make this task non-trivial for machines.

---

## Objective: Comparing Three Eras of NLP

This project implements sentiment classification using three fundamentally different approaches, answering the question: **How much do modern deep learning techniques improve over classical methods for text classification?**

| Era | Approach | How It Works |
|-----|----------|--------------|
| **Classical ML** (2000s) | TF-IDF + Linear Models | Convert text to word frequency vectors, classify with traditional algorithms |
| **Deep Learning** (2010s) | Word Embeddings + Neural Networks | Learn word representations, capture sequential patterns with LSTM/CNN |
| **Transformers** (2020s) | Pre-trained Language Models | Fine-tune models that already understand language structure |

By implementing all three on the same dataset, we can directly compare their accuracy, training time, and complexity tradeoffs.

---

## Results: The Surprising Truth About Model Complexity

### Summary

| Approach | Best Model | Accuracy | Training Time | Complexity |
|----------|------------|----------|---------------|------------|
| Classical ML | Linear SVM | 86.3% | ~2 min | Low |
| Deep Learning | BiLSTM / CNN | 87.7% | ~10 min (GPU) | Medium |
| Transformers | DistilBERT | 87.6% | ~15 min (GPU) | High |

**Key Finding**: Deep learning provides only a **1.4% accuracy improvement** over classical ML, while requiring significantly more computational resources. For many practical applications, classical ML may be the better choice.

---

## Detailed Analysis

### Classical ML: The Strong Baseline

Six models were trained using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which converts text into numerical features based on word importance.

![Model Comparison](images/01_model_comparison.png)

*Figure 1: Classical ML model comparison. Linear models (SVM, Logistic Regression) outperform tree-based methods on high-dimensional sparse text data.*

| Model | Accuracy | Why This Performance? |
|-------|----------|----------------------|
| **Linear SVM** | **86.3%** | Excels at high-dimensional data; finds optimal separating hyperplane |
| Logistic Regression | 86.2% | Similar to SVM but probabilistic; highly interpretable |
| Complement NB | 86.1% | Naive Bayes variant designed for imbalanced text data |
| Multinomial NB | 85.8% | Classic text classifier; assumes word independence |
| Random Forest | 84.5% | Tree ensembles struggle with sparse high-dimensional features |
| Gradient Boosting | 82.1% | Sequential boosting less effective for text than parallel methods |

**Interpretation**: Linear models dominate because TF-IDF creates high-dimensional sparse vectors (thousands of features). Linear SVM and Logistic Regression handle this naturally, while tree-based methods struggle to find meaningful splits in sparse space.

#### Confusion Matrix Analysis

![Confusion Matrix](images/02_confusion_matrix.png)

*Figure 2: Confusion matrix for Linear SVM. The model makes roughly equal errors on both classes (~1,700 false positives and ~1,700 false negatives), indicating no systematic bias toward either sentiment.*

The balanced error distribution shows the model isn't simply predicting the majority class - it genuinely learned to distinguish positive from negative sentiment.

#### ROC Curves: Measuring Discrimination Ability

![ROC Curves](images/03_roc_curves.png)

*Figure 3: ROC curves for all classical models. The curve plots True Positive Rate vs False Positive Rate at various classification thresholds. Curves closer to the top-left corner indicate better discrimination.*

| Model | AUC Score | Interpretation |
|-------|-----------|----------------|
| Linear SVM | 0.93 | Excellent discrimination |
| Logistic Regression | 0.93 | Equally strong |
| Naive Bayes variants | 0.91-0.92 | Very good |
| Tree models | 0.89-0.91 | Good but lower |

An AUC of 0.93 means: if we randomly pick one positive and one negative review, the model correctly ranks them 93% of the time.

---

### Deep Learning: Neural Networks for Text

Four architectures were trained using PyTorch with learned word embeddings:

- **LSTM** (Long Short-Term Memory): Processes text sequentially, maintaining a "memory" of previous words
- **BiLSTM**: Reads text both forward and backward, capturing context from both directions
- **CNN** (Convolutional Neural Network): Applies filters to detect local patterns (n-grams)
- **CNN+LSTM**: Hybrid combining CNN feature extraction with LSTM sequence modeling

#### Training Dynamics

![Training History](images/04_dl_training_history.png)

*Figure 4: Training and validation metrics over 10 epochs. Top row shows loss (lower is better), bottom row shows accuracy (higher is better). Solid lines = training, dashed lines = validation.*

**What the curves reveal:**

- **BiLSTM & CNN** (left two columns): Smooth convergence with training and validation curves tracking closely - healthy learning without overfitting
- **LSTM & CNN+LSTM** (right two columns): Flat lines at 50% accuracy - complete failure to learn

#### Why Did LSTM and CNN+LSTM Fail?

The vanilla LSTM and hybrid model failed to converge, stuck at random-chance accuracy (50%). This is likely due to:

1. **Vanishing gradients**: Long sequences cause gradients to diminish during backpropagation
2. **Synthetic data patterns**: The generated reviews may have patterns that BiLSTM captures but vanilla LSTM misses
3. **Architecture sensitivity**: These models require more careful hyperparameter tuning

The BiLSTM's bidirectional processing and CNN's local pattern detection proved more robust to these issues.

#### Deep Learning Comparison

![DL Model Comparison](images/05_dl_model_comparison.png)

*Figure 5: Final test accuracy for deep learning models. BiLSTM and CNN both achieve 87.7%, a 1.4% improvement over classical ML. The failed models serve as a reminder that neural networks aren't magic - architecture matters.*

#### BiLSTM Confusion Matrix

![DL Confusion Matrix](images/06_dl_confusion_matrix.png)

*Figure 6: BiLSTM confusion matrix on 25,000 test samples. True Positives: ~11,000, True Negatives: ~11,000, False Positives: ~1,500, False Negatives: ~1,500. Similar error pattern to classical ML but with fewer total errors.*

#### Deep Learning ROC Curves

![DL ROC Curves](images/07_dl_roc_curves.png)

*Figure 7: ROC curves for deep learning models. BiLSTM and CNN achieve AUC = 0.878, slightly lower than classical ML's 0.93. The failed models show AUC = 0.50 (diagonal line = no discrimination, equivalent to random guessing).*

**Interesting observation**: Classical ML achieves higher AUC (0.93) than deep learning (0.878) despite lower accuracy. This suggests classical models produce better-calibrated probability estimates, while deep learning models are more "confident" but less nuanced in their predictions.

---

### Transformers: Pre-trained Language Understanding

DistilBERT is a compressed version of BERT (Bidirectional Encoder Representations from Transformers), retaining 97% of BERT's performance with 40% fewer parameters.

| Model | Accuracy | Parameters | Why DistilBERT? |
|-------|----------|------------|-----------------|
| DistilBERT | 87.6% | 66M | Fits in 3GB GPU memory |
| BERT-base | N/A | 110M | Requires >6GB GPU memory |

**Key insight**: DistilBERT matches our custom deep learning models (87.6% vs 87.7%) despite being trained on general text, not movie reviews. This demonstrates the power of transfer learning - pre-trained knowledge about language structure transfers well to specific tasks.

---

## Key Learnings

### 1. Diminishing Returns on Complexity

```
Classical ML (86.3%) → Deep Learning (87.7%) → Transformers (87.6%)
         +1.4%                    -0.1%
```

The jump from classical to deep learning yields only 1.4% improvement. Transformers don't improve further on this task. For production systems where simplicity and speed matter, classical ML may be the pragmatic choice.

### 2. Why Classical ML Holds Up

TF-IDF + Linear SVM works well because:
- Sentiment often depends on **specific words** ("excellent", "terrible") that TF-IDF captures directly
- The model doesn't need to understand grammar or context for most reviews
- Linear decision boundaries are sufficient for this feature space

### 3. When Deep Learning Helps

BiLSTM and CNN provide benefits when:
- **Word order matters**: "not good" vs "good not" (negation handling)
- **Context is important**: "the acting was bad but the story was amazing"
- **Subtle patterns exist**: Sarcasm, implied sentiment

### 4. Architecture Matters More Than Depth

The failure of vanilla LSTM while BiLSTM succeeded shows that neural network design choices (bidirectional processing, skip connections, attention) often matter more than simply adding layers.

### 5. Transfer Learning is Powerful

DistilBERT, trained on Wikipedia and books, achieved competitive performance on movie reviews with minimal fine-tuning. Pre-trained models encode general language understanding that transfers across domains.

---

## Project Structure

```
06-nlp-sentiment-analysis/
├── data/
│   └── sample/                     # Small sample for demos (500 reviews)
├── images/                         # 7 visualizations
├── models/                         # Trained models (not in git)
├── scripts/
│   ├── synthetic_data.py           # Generates 50K movie reviews
│   ├── train_classical_ml.py       # Trains 6 classical models
│   ├── train_deep_learning.py      # Trains LSTM/CNN variants
│   └── train_transformers.py       # Fine-tunes DistilBERT
├── src/
│   └── preprocessing.py            # Text cleaning utilities
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Navigate to project
cd 06-nlp-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Train classical ML models (~2 minutes, CPU)
python scripts/train_classical_ml.py

# Train deep learning models (~10 minutes on GPU, longer on CPU)
python scripts/train_deep_learning.py

# Fine-tune transformer (~15 minutes on GPU)
python scripts/train_transformers.py
```

### GPU Setup

For deep learning and transformer training, CUDA acceleration is recommended:

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Dataset

The project uses synthetic movie reviews designed to mimic IMDB characteristics:

| Property | Value |
|----------|-------|
| Total reviews | 50,000 |
| Train/Test split | 25,000 / 25,000 |
| Classes | Positive / Negative (balanced) |
| Avg. review length | ~150 words |

The synthetic data includes:
- Sentiment-bearing vocabulary ("excellent", "terrible", "boring")
- Negation patterns ("not good", "wasn't bad")
- Neutral filler text for realism
- Varying sentence structures

Using synthetic data ensures reproducibility and eliminates external dependencies while maintaining realistic NLP challenges.

---

## Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| Classical ML | scikit-learn | TF-IDF, SVM, Naive Bayes, ensembles |
| Deep Learning | PyTorch | Custom LSTM, BiLSTM, CNN architectures |
| Transformers | HuggingFace | DistilBERT fine-tuning |
| Visualization | Matplotlib, Seaborn | Training curves, confusion matrices, ROC |
| Text Processing | NLTK | Tokenization, preprocessing |

---

## License

MIT License
