"""
Deep Learning Models for Astrophysics Computer Vision.

This module implements specialized neural network architectures for:
1. Galaxy Morphology Classification (2D image input)
2. Transient/Variable Star Classification (1D time-series input)
3. Anomaly Detection (Autoencoder for both domains)

================================================================================
MODEL ARCHITECTURE RATIONALE
================================================================================

WHY THESE ARCHITECTURES FOR EACH TASK?

GALAXY CLASSIFICATION (2D Images):
----------------------------------
We use 2D CNNs because:
- Galaxy morphology is fundamentally a spatial pattern recognition task
- Convolutional filters detect local features (spiral arms, bars, bulges)
- Translation equivariance handles galaxy position in frame
- Hierarchical feature learning: edges → textures → structures → morphology

Why Transfer Learning (ResNet, EfficientNet)?
- Galaxy images share low-level features with natural images (edges, gradients)
- Pre-trained models provide robust feature extractors
- Significantly reduces training data requirements
- Fine-tuning top layers adapts to astronomical specifics

Why NOT U-Net or Segmentation Models?
- U-Net is for pixel-wise segmentation (e.g., galaxy/background separation)
- Our task is classification, not segmentation
- We don't need pixel-level output, just class probabilities
- U-Net would be appropriate for: finding galaxy boundaries, masking, deblending


TRANSIENT CLASSIFICATION (Light Curves):
----------------------------------------
We use 1D-CNN and LSTM because:
- Light curves are 1D time series with temporal structure
- 1D convolutions detect local temporal patterns (rise, peak, decline shapes)
- LSTMs capture long-range dependencies (e.g., late-time behavior)
- Recurrent connections naturally handle variable-length sequences

Why 1D-CNN over 2D-CNN?
- Light curves are NOT images! They are 1D signals across time
- 2D-CNN would require artificially creating an image (spectrogram-like)
- This loses temporal ordering and requires arbitrary binning
- 1D-CNN directly processes the natural structure of the data

Why LSTM over Transformer?
- Transformers excel with very long sequences (1000s of tokens)
- Our light curves have ~100-200 observations
- LSTM is more parameter-efficient for this scale
- Bidirectional LSTM captures past and future context effectively
- For very large datasets, Transformers would be worth exploring

Why NOT standard RNNs?
- Vanilla RNNs suffer from vanishing gradients
- LSTMs have gating mechanisms to preserve long-term information
- Supernovae have characteristic timescales of weeks to months
- Need to remember early behavior to classify correctly


ANOMALY DETECTION (Autoencoders):
---------------------------------
We use autoencoders because:
- Unsupervised: can detect novel objects without labeled examples
- Learn compressed representation of "normal" astronomical objects
- High reconstruction error signals unusual/interesting objects
- Bottleneck forces learning meaningful latent features

Why autoencoder over other methods?
- Isolation Forest: works well but misses subtle anomalies in high-D data
- One-Class SVM: struggles with complex, multimodal distributions
- Autoencoders learn the data manifold, generalizing better to new data
- Can visualize latent space for scientific interpretation

================================================================================
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import warnings

# Deep Learning Frameworks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Some models unavailable.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not installed. Some models unavailable.")

# Classical ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =============================================================================
# GALAXY CLASSIFICATION MODELS (2D CNN)
# =============================================================================

class GalaxyCNN(nn.Module):
    """
    Custom CNN for Galaxy Morphology Classification.

    Architecture designed for galaxy images:
    - Small initial filters (3x3) to capture fine structure (spiral arms)
    - Progressive downsampling to capture global morphology
    - Batch normalization for training stability
    - Dropout for regularization (prevents overfitting to specific galaxies)

    Input: (batch, 3, H, W) RGB galaxy images
    Output: (batch, n_classes) class probabilities
    """

    def __init__(
        self,
        n_classes: int = 5,
        input_size: Tuple[int, int] = (128, 128),
        dropout: float = 0.5
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: Low-level features (edges, gradients)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: Mid-level features (textures, small structures)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: High-level features (spiral arms, bars, bulges)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 4: Global morphology
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_transfer_model(
    n_classes: int = 5,
    backbone: str = 'resnet18',
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Create a transfer learning model for galaxy classification.

    Uses ImageNet-pretrained backbone and replaces final layer.
    Low-level features (edges, textures) transfer well from natural images.

    Parameters
    ----------
    n_classes : int
        Number of galaxy morphology classes
    backbone : str
        Which pretrained model to use ('resnet18', 'resnet50', 'efficientnet_b0')
    pretrained : bool
        Whether to use ImageNet pretrained weights
    freeze_backbone : bool
        Whether to freeze backbone weights (only train classifier)

    Returns
    -------
    nn.Module
        Transfer learning model
    """
    import torchvision.models as models

    if backbone == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, n_classes)

    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, n_classes)

    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        n_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(n_features, n_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        if backbone.startswith('resnet'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif backbone.startswith('efficientnet'):
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model


# =============================================================================
# TRANSIENT CLASSIFICATION MODELS (1D CNN / LSTM)
# =============================================================================

class TransientCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for Light Curve Classification.

    Why 1D convolutions for light curves?
    - Light curves are temporal sequences, not 2D images
    - 1D conv filters detect local temporal patterns:
      * Rising phase → declining phase (supernova-like)
      * Periodic oscillations (variable stars)
      * Rapid transients (kilonovae)
    - Captures shape features regardless of exact timing

    Architecture:
    - Multiple 1D conv blocks with increasing receptive field
    - Each passband processed independently, then combined
    - Global pooling aggregates over time dimension

    Input: (batch, n_bands, n_time) multi-band light curves
    Output: (batch, n_classes) class probabilities
    """

    def __init__(
        self,
        n_classes: int = 14,
        n_bands: int = 6,
        n_time: int = 100,
        dropout: float = 0.5
    ):
        super().__init__()

        self.n_bands = n_bands

        # Per-band 1D convolutions
        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ) for _ in range(n_bands)
        ])

        # Combined processing
        self.combined_conv = nn.Sequential(
            nn.Conv1d(64 * n_bands, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bands, n_time)

        # Process each band separately
        band_features = []
        for i in range(self.n_bands):
            band_input = x[:, i:i+1, :]  # (batch, 1, n_time)
            band_out = self.band_convs[i](band_input)
            band_features.append(band_out)

        # Concatenate band features
        combined = torch.cat(band_features, dim=1)  # (batch, 64*n_bands, n_time/2)

        # Combined processing
        features = self.combined_conv(combined)

        # Classify
        output = self.classifier(features)

        return output


class TransientLSTM(nn.Module):
    """
    Bidirectional LSTM for Light Curve Classification.

    Why LSTM for transients?
    - Light curves have sequential nature with long-range dependencies
    - Early rise behavior predicts late-time classification
    - LSTM gates prevent vanishing gradients over long sequences
    - Bidirectional processing captures context from both directions

    This is particularly important for:
    - Supernovae: early spectra-like features predict type
    - Variable stars: period and amplitude relationships
    - AGN: irregular but correlated variability patterns

    Input: (batch, n_time, n_bands) multi-band light curves
    Output: (batch, n_classes) class probabilities
    """

    def __init__(
        self,
        n_classes: int = 14,
        n_bands: int = 6,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_bands,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bands, n_time) -> transpose to (batch, n_time, n_bands)
        x = x.transpose(1, 2)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state (concatenated for bidirectional)
        if self.lstm.bidirectional:
            # Concatenate forward and backward final hidden states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_final = h_n[-1]

        # Classify
        output = self.classifier(h_final)

        return output


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM for Light Curve Classification.

    Combines the best of both approaches:
    - CNN extracts local temporal features (rise/decline shapes)
    - LSTM captures long-range dependencies between features

    This architecture is effective because:
    1. CNNs are good at detecting local patterns invariant to position
    2. LSTMs are good at modeling sequential relationships
    3. Combined: detects local features AND their temporal relationships

    Input: (batch, n_bands, n_time) multi-band light curves
    Output: (batch, n_classes) class probabilities
    """

    def __init__(
        self,
        n_classes: int = 14,
        n_bands: int = 6,
        dropout: float = 0.5
    ):
        super().__init__()

        # 1D CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(n_bands, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # LSTM on CNN features
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 128 * 2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bands, n_time)

        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, 64, n_time/4)

        # Reshape for LSTM: (batch, n_time/4, 64)
        lstm_in = cnn_out.transpose(1, 2)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)

        # Use final hidden state
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)

        # Classify
        output = self.classifier(h_final)

        return output


# =============================================================================
# ANOMALY DETECTION (Autoencoders)
# =============================================================================

class GalaxyAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for Galaxy Anomaly Detection.

    The autoencoder learns to compress and reconstruct normal galaxy images.
    Anomalous objects (unusual morphologies, artifacts, rare types) will
    have high reconstruction error, flagging them for review.

    Applications:
    - Finding rare galaxy types (ring galaxies, mergers, gravitational lenses)
    - Detecting image artifacts (cosmic rays, satellite trails)
    - Discovering previously unknown object classes

    Architecture:
    - Encoder: Progressive downsampling to compressed latent space
    - Decoder: Progressive upsampling to reconstruct original
    - Bottleneck: Forces learning meaningful compressed representation
    """

    def __init__(
        self,
        latent_dim: int = 64,
        input_size: Tuple[int, int] = (128, 128)
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # 128x128
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score (reconstruction error).

        Higher score = more anomalous
        """
        x_recon, _ = self.forward(x)
        mse = torch.mean((x - x_recon) ** 2, dim=(1, 2, 3))
        return mse


class LightCurveAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for Light Curve Anomaly Detection.

    Learns to reconstruct typical light curve patterns.
    Unusual transients (novel physics, instrument artifacts) will
    have high reconstruction error.

    This can discover:
    - New transient types not in training data
    - Peculiar supernovae with unusual properties
    - Instrument malfunctions or data quality issues
    """

    def __init__(
        self,
        n_bands: int = 6,
        n_time: int = 100,
        latent_dim: int = 32
    ):
        super().__init__()

        self.n_bands = n_bands
        self.n_time = n_time
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_bands, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Calculate flattened size
        self.flat_size = 128 * (n_time // 8)

        self.fc_encode = nn.Linear(self.flat_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, n_time // 8)),

            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, n_bands, 7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        z = self.fc_encode(features)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.fc_decode(z)
        x_recon = self.decoder(features)
        # Crop/pad to exact original size
        x_recon = x_recon[:, :, :self.n_time]
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        x_recon, _ = self.forward(x)
        # Handle size mismatch
        min_len = min(x.shape[2], x_recon.shape[2])
        mse = torch.mean((x[:, :, :min_len] - x_recon[:, :, :min_len]) ** 2, dim=(1, 2))
        return mse


# =============================================================================
# CLASSICAL MACHINE LEARNING MODELS
# =============================================================================

def create_classical_pipeline(
    model_type: str = 'random_forest',
    n_classes: int = 14
) -> Pipeline:
    """
    Create a classical ML pipeline for light curve classification.

    These models work with extracted features (not raw time series).
    Often competitive with deep learning on smaller datasets.

    Parameters
    ----------
    model_type : str
        One of 'random_forest', 'gradient_boosting', 'svm', 'logistic'

    Returns
    -------
    Pipeline
        sklearn Pipeline with scaler and classifier
    """
    if model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )

    elif model_type == 'gradient_boosting':
        classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

    elif model_type == 'svm':
        classifier = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )

    elif model_type == 'logistic':
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            multi_class='multinomial',
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda'
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop
