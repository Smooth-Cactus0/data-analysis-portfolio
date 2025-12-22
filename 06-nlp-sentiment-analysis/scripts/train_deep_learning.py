"""
Deep Learning Training Script - LSTM, BiLSTM, CNN, CNN+LSTM models.
Author: Alexy Louis
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from scripts.synthetic_data import create_imdb_like_dataset
from src.preprocessing import preprocess_texts
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
MAX_VOCAB = 15000
MAX_LEN = 100
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 3

# =============================================================================
# Data Preparation
# =============================================================================

class Vocabulary:
    """Simple vocabulary for tokenization."""
    def __init__(self, max_size=15000):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.max_size = max_size

    def build(self, texts):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        for word, _ in word_counts.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        return self

    def encode(self, text, max_len=100):
        tokens = text.split()[:max_len]
        indices = [self.word2idx.get(w, 1) for w in tokens]
        # Pad
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)


class SentimentDataset(Dataset):
    """PyTorch dataset for sentiment classification."""
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.vocab.encode(self.texts[idx], self.max_len)
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


# =============================================================================
# Model Architectures
# =============================================================================

class LSTMClassifier(nn.Module):
    """Simple LSTM for text classification."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq, embed)
        _, (hidden, _) = self.lstm(embedded)  # hidden: (1, batch, hidden)
        hidden = hidden.squeeze(0)  # (batch, hidden)
        out = self.dropout(hidden)
        return self.fc(out).squeeze(-1)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for text classification."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)  # hidden: (2, batch, hidden)
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)  # (batch, hidden*2)
        out = self.dropout(hidden)
        return self.fc(out).squeeze(-1)


class CNNClassifier(nn.Module):
    """1D CNN for text classification."""
    def __init__(self, vocab_size, embed_dim, num_filters=128, filter_sizes=[3, 4, 5], dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq, embed)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed, seq) for Conv1d

        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # (batch, filters, seq-fs+1)
            pooled = torch.max(conv_out, dim=2)[0]  # (batch, filters)
            conv_outputs.append(pooled)

        concat = torch.cat(conv_outputs, dim=1)  # (batch, filters * num_convs)
        out = self.dropout(concat)
        return self.fc(out).squeeze(-1)


class CNNLSTMClassifier(nn.Module):
    """CNN + LSTM hybrid for text classification."""
    def __init__(self, vocab_size, embed_dim, num_filters=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(num_filters, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq, embed)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed, seq)
        conv_out = torch.relu(self.conv(embedded))  # (batch, filters, seq)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, seq, filters)
        _, (hidden, _) = self.lstm(conv_out)  # hidden: (1, batch, hidden)
        hidden = hidden.squeeze(0)
        out = self.dropout(hidden)
        return self.fc(out).squeeze(-1)


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for texts, labels in dataloader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, accuracy, f1, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def train_model(model, train_loader, val_loader, epochs=15, patience=3, model_name='model'):
    """Train model with early stopping."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _, _, _ = evaluate(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f'{model_name}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / f'{model_name}.pt', weights_only=True))
    return history


# =============================================================================
# Main Training Pipeline
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("DEEP LEARNING TRAINING PIPELINE")
    print("="*60)

    # 1. Generate data
    print("\n[1/5] Generating and preprocessing data...")
    dataset = create_imdb_like_dataset(n_train=25000, n_test=25000, seed=42)
    train_texts = preprocess_texts(dataset['train']['text'])
    test_texts = preprocess_texts(dataset['test']['text'])
    y_train = dataset['train']['label']
    y_test = dataset['test']['label']

    # Split train into train/val (80/20)
    val_size = int(len(train_texts) * 0.2)
    val_texts, val_labels = train_texts[:val_size], y_train[:val_size]
    train_texts, train_labels = train_texts[val_size:], y_train[val_size:]

    print(f"  Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    # 2. Build vocabulary
    print("\n[2/5] Building vocabulary...")
    vocab = Vocabulary(max_size=MAX_VOCAB)
    vocab.build(train_texts)
    print(f"  Vocabulary size: {len(vocab):,}")

    # 3. Create datasets and dataloaders
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, MAX_LEN)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab, MAX_LEN)
    test_dataset = SentimentDataset(test_texts, y_test, vocab, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 4. Train models
    print("\n[3/5] Training deep learning models...")

    models_config = {
        'LSTM': LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM),
        'BiLSTM': BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM),
        'CNN': CNNClassifier(len(vocab), EMBED_DIM),
        'CNN_LSTM': CNNLSTMClassifier(len(vocab), EMBED_DIM),
    }

    all_histories = {}
    results = []
    test_predictions = {}

    criterion = nn.BCEWithLogitsLoss()

    for name, model in models_config.items():
        print(f"\n--- {name} ---")
        model = model.to(DEVICE)
        history = train_model(model, train_loader, val_loader, EPOCHS, PATIENCE, name.lower())
        all_histories[name] = history

        # Evaluate on test set
        test_loss, test_acc, test_f1, y_true, y_pred, y_probs = evaluate(model, test_loader, criterion)
        results.append({'Model': name, 'Accuracy': test_acc, 'F1': test_f1})
        test_predictions[name] = {'y_true': y_true, 'y_pred': y_pred, 'y_probs': y_probs}
        print(f"  Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")

    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))

    # 5. Visualizations
    print("\n[4/5] Generating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'LSTM': '#1f77b4', 'BiLSTM': '#ff7f0e', 'CNN': '#2ca02c', 'CNN_LSTM': '#d62728'}

    for name, history in all_histories.items():
        epochs_range = range(1, len(history['train_loss']) + 1)
        axes[0, 0].plot(epochs_range, history['train_loss'], color=colors[name], label=name)
        axes[0, 1].plot(epochs_range, history['val_loss'], color=colors[name], label=name)
        axes[1, 0].plot(epochs_range, history['train_acc'], color=colors[name], label=name)
        axes[1, 1].plot(epochs_range, history['val_acc'], color=colors[name], label=name)

    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    axes[1, 0].set_title('Training Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()

    axes[1, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()

    plt.suptitle('Deep Learning Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '04_dl_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = sns.color_palette('viridis', len(results_df))
    bars = ax.barh(results_df['Model'], results_df['Accuracy'], color=colors_bar)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Deep Learning Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(0.80, 0.92)
    for bar, acc in zip(bars, results_df['Accuracy']):
        ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2, f'{acc:.2%}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '05_dl_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Best model confusion matrix
    best_model_name = results_df.iloc[0]['Model']
    best_preds = test_predictions[best_model_name]
    cm = confusion_matrix(best_preds['y_true'], best_preds['y_pred'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '06_dl_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, preds in test_predictions.items():
        fpr, tpr, _ = roc_curve(preds['y_true'], preds['y_probs'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors[name], label=f'{name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Deep Learning Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '07_dl_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Save results
    print("\n[5/5] Saving results...")
    results_df.to_csv(MODELS_DIR / 'dl_results.csv', index=False)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {results_df.iloc[0]['Accuracy']:.2%}")
    print(f"Test F1: {results_df.iloc[0]['F1']:.3f}")
    print(f"\nFiles saved:")
    print(f"  - models/*.pt (4 model weights)")
    print(f"  - models/dl_results.csv")
    print(f"  - images/04_dl_training_history.png")
    print(f"  - images/05_dl_model_comparison.png")
    print(f"  - images/06_dl_confusion_matrix.png")
    print(f"  - images/07_dl_roc_curves.png")
