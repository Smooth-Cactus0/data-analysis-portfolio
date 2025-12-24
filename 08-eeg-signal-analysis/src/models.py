"""
EEG Classification and Prediction Models

Includes:
- Classical ML models (LDA, SVM, Random Forest, XGBoost)
- Deep Learning models (EEGNet, ShallowConvNet, DeepConvNet)
- Neural state prediction models (LSTM, Transformer)
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =============================================================================
# Classical ML Models
# =============================================================================

def get_classical_models() -> Dict[str, Any]:
    """
    Get dictionary of classical ML models.

    Returns
    -------
    models : dict
        Dictionary of model name to model instance
    """
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "SVM_linear": SVC(kernel="linear", probability=True, random_state=42),
        "SVM_rbf": SVC(kernel="rbf", probability=True, random_state=42),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Logistic_Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    return models


def train_classical_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Train a classical ML model and evaluate.

    Parameters
    ----------
    model : sklearn estimator
        Model to train
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray, optional
        Test features
    y_test : np.ndarray, optional
        Test labels

    Returns
    -------
    results : dict
        Training results including accuracy, predictions, etc.
    """
    # Train
    model.fit(X_train, y_train)

    results = {
        "model": model,
        "train_accuracy": model.score(X_train, y_train)
    }

    # Evaluate on test set
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        results["test_accuracy"] = accuracy_score(y_test, y_pred)
        results["predictions"] = y_pred
        results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        results["classification_report"] = classification_report(y_test, y_pred)

        # Probabilities if available
        if hasattr(model, "predict_proba"):
            results["probabilities"] = model.predict_proba(X_test)

    return results


def cross_validate_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Cross-validate multiple models.

    Parameters
    ----------
    models : dict
        Dictionary of models
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : int
        Number of folds

    Returns
    -------
    results : dict
        Cross-validation results for each model
    """
    results = {}

    for name, model in models.items():
        print(f"Cross-validating {name}...")
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        results[name] = {
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
            "scores": scores
        }
        print(f"  Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return results


# =============================================================================
# Deep Learning Models
# =============================================================================

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_samples, n_channels, n_times)
        y : np.ndarray
            Labels
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EEGNet(nn.Module):
    """
    EEGNet: A Compact CNN for EEG-based BCIs.

    Reference: Lawhern et al. (2018)
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 640,
        n_classes: int = 2,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times

        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise spatial convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable convolution
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flattened size
        self._calculate_flatten_size()

        # Classifier
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def _calculate_flatten_size(self):
        """Calculate the size of flattened features."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self._forward_features(x)
            self.flatten_size = x.numel()

    def _forward_features(self, x):
        """Forward pass through feature extraction layers."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n_channels, n_times)

        Returns
        -------
        out : torch.Tensor
            Class logits
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, channels, times)

        x = self._forward_features(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


class ShallowConvNet(nn.Module):
    """
    Shallow ConvNet for EEG classification.

    Reference: Schirrmeister et al. (2017)
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 640,
        n_classes: int = 2,
        n_filters: int = 40,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times

        # Temporal convolution
        self.conv_time = nn.Conv2d(1, n_filters, (1, 25), bias=False)

        # Spatial convolution
        self.conv_spatial = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)

        # Pooling
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate flatten size
        self._calculate_flatten_size()

        # Classifier
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def _calculate_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self._forward_features(x)
            self.flatten_size = x.numel()

    def _forward_features(self, x):
        x = self.conv_time(x)
        x = self.conv_spatial(x)
        x = self.bn(x)
        x = x ** 2  # Square activation
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # Log activation
        x = self.dropout(x)
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self._forward_features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class DeepConvNet(nn.Module):
    """
    Deep ConvNet for EEG classification.

    Reference: Schirrmeister et al. (2017)
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 640,
        n_classes: int = 2,
        n_filters: int = 25,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times

        # First conv block
        self.conv1 = nn.Conv2d(1, n_filters, (1, 10), bias=False)
        self.conv2 = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.pool1 = nn.MaxPool2d((1, 3))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second conv block
        self.conv3 = nn.Conv2d(n_filters, n_filters * 2, (1, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters * 2)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Third conv block
        self.conv4 = nn.Conv2d(n_filters * 2, n_filters * 4, (1, 10), bias=False)
        self.bn3 = nn.BatchNorm2d(n_filters * 4)
        self.pool3 = nn.MaxPool2d((1, 3))
        self.dropout3 = nn.Dropout(dropout_rate)

        # Calculate flatten size
        self._calculate_flatten_size()

        # Classifier
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def _calculate_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self._forward_features(x)
            self.flatten_size = x.numel()

    def _forward_features(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self._forward_features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# =============================================================================
# Neural State Prediction Models
# =============================================================================

class EEGPredictor(nn.Module):
    """
    LSTM-based model for predicting future EEG states.
    """

    def __init__(
        self,
        n_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        prediction_steps: int = 1
    ):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.prediction_steps = prediction_steps

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_size, n_channels * prediction_steps)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n_times, n_channels)

        Returns
        -------
        out : torch.Tensor
            Predicted future states (batch, prediction_steps, n_channels)
        """
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])
        out = out.view(-1, self.prediction_steps, self.n_channels)
        return out


class EEGTransformerPredictor(nn.Module):
    """
    Transformer-based model for predicting future EEG states.
    """

    def __init__(
        self,
        n_channels: int = 64,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        prediction_steps: int = 1
    ):
        super().__init__()

        self.n_channels = n_channels
        self.d_model = d_model
        self.prediction_steps = prediction_steps

        # Input embedding
        self.input_embedding = nn.Linear(n_channels, d_model)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.fc = nn.Linear(d_model, n_channels * prediction_steps)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n_times, n_channels)

        Returns
        -------
        out : torch.Tensor
            Predicted future states (batch, prediction_steps, n_channels)
        """
        seq_len = x.size(1)

        # Embed input
        x = self.input_embedding(x)

        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)

        # Use last position for prediction
        out = self.fc(x[:, -1, :])
        out = out.view(-1, self.prediction_steps, self.n_channels)

        return out


# =============================================================================
# Training Utilities
# =============================================================================

def train_deep_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cpu",
    patience: int = 10
) -> Dict[str, Any]:
    """
    Train a deep learning model.

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    epochs : int
        Number of epochs
    lr : float
        Learning rate
    device : str
        Device to train on
    patience : int
        Early stopping patience

    Returns
    -------
    results : dict
        Training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += y_batch.size(0)
                    val_correct += predicted.eq(y_batch).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def evaluate_deep_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluate a deep learning model.

    Parameters
    ----------
    model : nn.Module
        Trained model
    test_loader : DataLoader
        Test data loader
    device : str
        Device

    Returns
    -------
    results : dict
        Evaluation results
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "classification_report": classification_report(all_labels, all_preds)
    }
