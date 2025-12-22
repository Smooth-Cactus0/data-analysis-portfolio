"""
Transformer Training Script - DistilBERT and BERT fine-tuning.
Author: Alexy Louis
"""
import sys
sys.path.insert(0, '.')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from scripts.synthetic_data import create_imdb_like_dataset
import warnings
warnings.filterwarnings('ignore')

# Paths
MODELS_DIR = Path('models')
IMAGES_DIR = Path('images')
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
MAX_LEN = 64  # Shorter sequences for faster training
BATCH_SIZE = 16 if torch.cuda.is_available() else 4
EPOCHS = 2  # Fewer epochs for CPU
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# =============================================================================
# Dataset
# =============================================================================

class SentimentDataset(Dataset):
    """PyTorch dataset for transformer models."""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1}


# =============================================================================
# Training Function
# =============================================================================

def train_transformer(model_name, train_dataset, val_dataset, test_dataset,
                      output_dir, epochs=3, batch_size=8):
    """Fine-tune a transformer model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print('='*60)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # Create datasets
    train_ds = SentimentDataset(
        train_dataset['text'], train_dataset['label'], tokenizer, MAX_LEN
    )
    val_ds = SentimentDataset(
        val_dataset['text'], val_dataset['label'], tokenizer, MAX_LEN
    )
    test_ds = SentimentDataset(
        test_dataset['text'], test_dataset['label'], tokenizer, MAX_LEN
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        save_total_limit=1,
        report_to='none',
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    print("Training...")
    train_result = trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds)

    # Get predictions for visualizations
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
    labels = predictions.label_ids

    # Extract training history
    history = {
        'train_loss': [],
        'eval_loss': [],
        'eval_accuracy': []
    }
    for log in trainer.state.log_history:
        if 'loss' in log and 'eval_loss' not in log:
            history['train_loss'].append(log['loss'])
        if 'eval_loss' in log:
            history['eval_loss'].append(log['eval_loss'])
        if 'eval_accuracy' in log:
            history['eval_accuracy'].append(log['eval_accuracy'])

    return {
        'model_name': model_name,
        'test_accuracy': test_results['eval_accuracy'],
        'test_f1': test_results['eval_f1'],
        'predictions': preds,
        'probabilities': probs,
        'labels': labels,
        'history': history,
        'trainer': trainer
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TRANSFORMER TRAINING PIPELINE")
    print("="*60)

    # 1. Generate data (smaller subset for transformers due to compute)
    print("\n[1/4] Generating data...")
    # Use smaller dataset for faster training on CPU
    n_train = 2000 if not torch.cuda.is_available() else 20000
    n_test = 1000 if not torch.cuda.is_available() else 5000

    dataset = create_imdb_like_dataset(n_train=n_train, n_test=n_test, seed=42)

    # Split train into train/val
    val_size = int(len(dataset['train']['text']) * 0.1)

    train_data = {
        'text': dataset['train']['text'][val_size:],
        'label': dataset['train']['label'][val_size:]
    }
    val_data = {
        'text': dataset['train']['text'][:val_size],
        'label': dataset['train']['label'][:val_size]
    }
    test_data = {
        'text': dataset['test']['text'],
        'label': dataset['test']['label']
    }

    print(f"  Train: {len(train_data['text']):,} | Val: {len(val_data['text']):,} | Test: {len(test_data['text']):,}")

    # 2. Train models
    print("\n[2/4] Training transformer models...")

    models_to_train = [
        ('distilbert-base-uncased', 'DistilBERT'),
        ('bert-base-uncased', 'BERT'),
    ]

    results = []
    all_results = {}

    for model_id, model_name in models_to_train:
        output_dir = MODELS_DIR / model_name.lower()
        result = train_transformer(
            model_id,
            train_data,
            val_data,
            test_data,
            output_dir,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        results.append({
            'Model': model_name,
            'Accuracy': result['test_accuracy'],
            'F1': result['test_f1']
        })
        all_results[model_name] = result
        print(f"\n{model_name} - Test Accuracy: {result['test_accuracy']:.4f} | F1: {result['test_f1']:.4f}")

    # Results dataframe
    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))

    # 3. Visualizations
    print("\n[3/4] Generating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'DistilBERT': '#1f77b4', 'BERT': '#ff7f0e'}

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = [colors[m] for m in results_df['Model']]
    bars = ax.barh(results_df['Model'], results_df['Accuracy'], color=colors_bar)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Transformer Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(0.85, 0.95)
    for bar, acc in zip(bars, results_df['Accuracy']):
        ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2, f'{acc:.2%}', va='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '08_transformer_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Best model confusion matrix
    best_model_name = results_df.iloc[0]['Model']
    best_result = all_results[best_model_name]
    cm = confusion_matrix(best_result['labels'], best_result['predictions'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '09_transformer_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, result in all_results.items():
        fpr, tpr, _ = roc_curve(result['labels'], result['probabilities'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors[name], label=f'{name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Transformer Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '10_transformer_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # All models comparison (Classical + DL + Transformers)
    print("\n[4/4] Creating final comparison...")

    # Load previous results if available
    all_models_results = []

    # Classical ML results (from previous training)
    classical_results = [
        {'Model': 'Linear SVM', 'Accuracy': 0.863, 'F1': 0.863, 'Type': 'Classical ML'},
        {'Model': 'Logistic Reg', 'Accuracy': 0.862, 'F1': 0.862, 'Type': 'Classical ML'},
    ]

    # Deep Learning results
    dl_results = [
        {'Model': 'CNN', 'Accuracy': 0.877, 'F1': 0.877, 'Type': 'Deep Learning'},
        {'Model': 'BiLSTM', 'Accuracy': 0.877, 'F1': 0.877, 'Type': 'Deep Learning'},
    ]

    # Transformer results
    transformer_results = [
        {'Model': name, 'Accuracy': r['Accuracy'], 'F1': r['F1'], 'Type': 'Transformer'}
        for name, r in zip(results_df['Model'], results_df[['Accuracy', 'F1']].to_dict('records'))
    ]

    all_models_df = pd.DataFrame(classical_results + dl_results + transformer_results)
    all_models_df = all_models_df.sort_values('Accuracy', ascending=True)

    # Final comparison plot
    fig, ax = plt.subplots(figsize=(12, 7))
    type_colors = {'Classical ML': '#2ecc71', 'Deep Learning': '#3498db', 'Transformer': '#9b59b6'}
    bar_colors = [type_colors[t] for t in all_models_df['Type']]

    bars = ax.barh(all_models_df['Model'], all_models_df['Accuracy'], color=bar_colors)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Complete Model Comparison: Classical ML vs Deep Learning vs Transformers',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0.82, 0.96)

    for bar, acc in zip(bars, all_models_df['Accuracy']):
        ax.text(acc + 0.003, bar.get_y() + bar.get_height()/2, f'{acc:.1%}', va='center', fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '11_complete_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_df.to_csv(MODELS_DIR / 'transformer_results.csv', index=False)
    all_models_df.to_csv(MODELS_DIR / 'all_models_comparison.csv', index=False)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Transformer: {best_model_name}")
    print(f"Test Accuracy: {results_df.iloc[0]['Accuracy']:.2%}")
    print(f"Test F1: {results_df.iloc[0]['F1']:.3f}")
    print(f"\nFiles saved:")
    print(f"  - models/distilbert/, models/bert/ (model weights)")
    print(f"  - models/transformer_results.csv")
    print(f"  - models/all_models_comparison.csv")
    print(f"  - images/08_transformer_comparison.png")
    print(f"  - images/09_transformer_confusion_matrix.png")
    print(f"  - images/10_transformer_roc_curves.png")
    print(f"  - images/11_complete_model_comparison.png")
