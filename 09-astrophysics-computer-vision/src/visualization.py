"""
Visualization module for astrophysics computer vision.

Provides plotting functions for:
1. Galaxy morphology images and classifications
2. Light curve time series
3. Model performance metrics
4. Anomaly detection results

All visualizations follow astronomy conventions:
- Logarithmic scaling for brightness
- North-up, East-left for images (when applicable)
- Standard filter colors (u=purple, g=blue, r=green, i=orange, z=red, y=darkred)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Ellipse
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple, Union
import warnings

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')

# Standard passband colors (approximate filter wavelengths)
PASSBAND_COLORS = {
    0: '#8b5cf6',  # u - ultraviolet (purple)
    1: '#3b82f6',  # g - green (blue in plot)
    2: '#22c55e',  # r - red (green in plot for contrast)
    3: '#f97316',  # i - infrared (orange)
    4: '#ef4444',  # z - near-IR (red)
    5: '#7f1d1d',  # y - near-IR (dark red)
}

PASSBAND_NAMES = {
    0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'
}

# Galaxy morphology colors
MORPHOLOGY_COLORS = {
    'Elliptical': '#e74c3c',
    'Spiral': '#3498db',
    'Edge-on': '#2ecc71',
    'Irregular': '#9b59b6',
    'Merger': '#f39c12',
}

# Transient class colors (grouped by type)
TRANSIENT_COLORS = {
    # Type Ia SNe (cosmological)
    90: '#e74c3c',   # SNIa
    67: '#c0392b',   # SNIa-91bg
    52: '#a93226',   # SNIax

    # Core-collapse SNe
    42: '#3498db',   # SNII
    62: '#2980b9',   # SNIbc

    # Exotic transients
    95: '#9b59b6',   # SLSN-I
    15: '#8e44ad',   # TDE
    64: '#6c3483',   # Kilonova

    # Variable stars
    92: '#27ae60',   # RR Lyrae
    65: '#229954',   # M-dwarf
    16: '#1e8449',   # Eclipsing Binary
    53: '#196f3d',   # Mira

    # AGN
    88: '#f39c12',   # AGN

    # Other
    6: '#7f8c8d',    # Lens-Single
}


# =============================================================================
# GALAXY VISUALIZATION
# =============================================================================

def plot_galaxy_grid(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    n_rows: int = 3,
    n_cols: int = 4,
    figsize: Tuple[int, int] = (12, 9),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Display a grid of galaxy images with labels/predictions.

    Parameters
    ----------
    images : np.ndarray
        Array of images (N, H, W, C)
    labels : np.ndarray, optional
        True class labels
    predictions : np.ndarray, optional
        Predicted class labels
    class_names : list, optional
        Names of classes
    n_rows, n_cols : int
        Grid dimensions
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    n_images = min(len(images), n_rows * n_cols)

    for i in range(n_images):
        ax = axes[i]
        img = images[i]

        # Handle different image formats
        if img.shape[-1] == 1:
            ax.imshow(img[:, :, 0], cmap='gray')
        else:
            # Clip and normalize for display
            img_display = np.clip(img, 0, 1)
            ax.imshow(img_display)

        ax.axis('off')

        # Add label/prediction
        title_parts = []
        if labels is not None:
            true_label = labels[i]
            if class_names:
                true_label = class_names[true_label]
            title_parts.append(f'True: {true_label}')

        if predictions is not None:
            pred_label = predictions[i]
            if class_names:
                pred_label = class_names[pred_label]
            title_parts.append(f'Pred: {pred_label}')

            # Color by correctness
            if labels is not None:
                color = 'green' if labels[i] == predictions[i] else 'red'
                ax.set_title('\n'.join(title_parts), fontsize=9, color=color)
            else:
                ax.set_title('\n'.join(title_parts), fontsize=9)
        elif title_parts:
            ax.set_title('\n'.join(title_parts), fontsize=9)

    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_galaxy_morphology_examples(
    images_by_class: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot example galaxies for each morphological class with descriptions.

    Parameters
    ----------
    images_by_class : dict
        Dictionary mapping class name to array of example images

    Returns
    -------
    matplotlib.Figure
    """
    # Morphology descriptions
    descriptions = {
        'Elliptical': 'Smooth, featureless, round to elliptical shape.\n'
                      'Old stellar populations, little gas/dust.',
        'Spiral': 'Central bulge with spiral arms.\n'
                  'Active star formation in arms.',
        'Edge-on': 'Disk galaxy viewed from the side.\n'
                   'Often shows dust lane.',
        'Irregular': 'No regular structure.\n'
                     'Often result of gravitational interaction.',
        'Merger': 'Two or more galaxies interacting.\n'
                  'Tidal tails, distorted morphology.',
    }

    n_classes = len(images_by_class)
    n_examples = 4

    fig, axes = plt.subplots(n_classes, n_examples + 1, figsize=figsize)

    for i, (class_name, images) in enumerate(images_by_class.items()):
        # Text description
        ax_text = axes[i, 0]
        ax_text.text(0.5, 0.5, f'{class_name}\n\n{descriptions.get(class_name, "")}',
                     ha='center', va='center', fontsize=10, wrap=True,
                     transform=ax_text.transAxes)
        ax_text.axis('off')
        ax_text.set_facecolor(MORPHOLOGY_COLORS.get(class_name, '#cccccc') + '30')

        # Example images
        for j in range(n_examples):
            ax = axes[i, j + 1]
            if j < len(images):
                img = np.clip(images[j], 0, 1)
                ax.imshow(img)
            ax.axis('off')

    plt.suptitle('Galaxy Morphology Classification', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# LIGHT CURVE VISUALIZATION
# =============================================================================

def plot_light_curve(
    lc_df: pd.DataFrame,
    object_id: Optional[int] = None,
    title: Optional[str] = None,
    show_detections_only: bool = False,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a multi-band light curve.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve data with columns: mjd, passband, flux, flux_err
    object_id : int, optional
        Filter to specific object
    title : str, optional
        Plot title
    show_detections_only : bool
        Only show points with detected=1
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    if object_id is not None:
        lc_df = lc_df[lc_df['object_id'] == object_id].copy()

    if show_detections_only and 'detected' in lc_df.columns:
        lc_df = lc_df[lc_df['detected'] == 1]

    fig, ax = plt.subplots(figsize=figsize)

    for pb in sorted(lc_df['passband'].unique()):
        pb_data = lc_df[lc_df['passband'] == pb].sort_values('mjd')

        color = PASSBAND_COLORS.get(pb, '#333333')
        label = PASSBAND_NAMES.get(pb, str(pb))

        ax.errorbar(
            pb_data['mjd'],
            pb_data['flux'],
            yerr=pb_data['flux_err'] if 'flux_err' in pb_data.columns else None,
            fmt='o',
            color=color,
            label=label,
            markersize=4,
            alpha=0.7,
            capsize=2
        )

    ax.set_xlabel('MJD (Modified Julian Date)', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    ax.legend(title='Passband', loc='best')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_light_curve_grid(
    lc_df: pd.DataFrame,
    object_ids: List[int],
    labels: Optional[Dict[int, str]] = None,
    n_cols: int = 3,
    figsize_per_plot: Tuple[float, float] = (4, 3),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple light curves in a grid.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve data for all objects
    object_ids : list
        List of object IDs to plot
    labels : dict, optional
        Dictionary mapping object_id to class label
    n_cols : int
        Number of columns in grid
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    n_objects = len(object_ids)
    n_rows = (n_objects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    )
    axes = np.atleast_2d(axes)

    for idx, obj_id in enumerate(object_ids):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        obj_lc = lc_df[lc_df['object_id'] == obj_id]

        for pb in sorted(obj_lc['passband'].unique()):
            pb_data = obj_lc[obj_lc['passband'] == pb].sort_values('mjd')
            color = PASSBAND_COLORS.get(pb, '#333333')

            ax.scatter(
                pb_data['mjd'],
                pb_data['flux'],
                c=color,
                s=10,
                alpha=0.7
            )

        ax.set_xlabel('MJD', fontsize=8)
        ax.set_ylabel('Flux', fontsize=8)

        if labels and obj_id in labels:
            ax.set_title(f'{labels[obj_id]}', fontsize=10)
        else:
            ax.set_title(f'Object {obj_id}', fontsize=10)

    # Hide empty subplots
    for idx in range(n_objects, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_transient_class_examples(
    lc_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    samples_per_class: int = 3,
    figsize: Tuple[int, int] = (15, 20),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot example light curves for each transient class.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve data
    metadata_df : pd.DataFrame
        Metadata with object_id, target, class_name
    samples_per_class : int
        Number of examples per class

    Returns
    -------
    matplotlib.Figure
    """
    # Class descriptions
    class_descriptions = {
        90: 'SNIa: Standard candle for cosmology.\nPeaks in ~20 days, fades over months.',
        67: 'SNIa-91bg: Subluminous Type Ia.\nFaster decline, redder colors.',
        52: 'SNIax: Type Iax, "failed" SNIa.\nLower luminosity, unusual spectra.',
        42: 'SNII: Core-collapse with hydrogen.\nPlateau phase, then decline.',
        62: 'SNIbc: Core-collapse, stripped envelope.\nNo hydrogen, helium (Ib) or neither (Ic).',
        95: 'SLSN-I: Superluminous supernova.\n10-100x brighter than normal SN.',
        15: 'TDE: Tidal Disruption Event.\nStar destroyed by black hole.',
        64: 'Kilonova: Neutron star merger.\nRapid rise and fall, red color.',
        88: 'AGN: Active Galactic Nucleus.\nStochastic variability, long timescales.',
        92: 'RR Lyrae: Pulsating variable.\nPeriod < 1 day, amplitude ~1 mag.',
        65: 'M-dwarf flare: Stellar flare.\nRapid brightening, exponential decay.',
        16: 'EB: Eclipsing Binary.\nPeriodic dips from stellar eclipses.',
        53: 'Mira: Long-period variable.\nPeriod 80-1000 days, large amplitude.',
        6: 'Microlensing: Gravitational lens.\nSymmetric rise and fall.'
    }

    classes = sorted(metadata_df['target'].unique())
    n_classes = len(classes)

    fig, axes = plt.subplots(
        n_classes, samples_per_class + 1,
        figsize=figsize
    )

    for i, class_id in enumerate(classes):
        # Get class info
        class_mask = metadata_df['target'] == class_id
        class_objects = metadata_df[class_mask]['object_id'].values
        class_name = metadata_df[class_mask]['class_name'].values[0] if 'class_name' in metadata_df.columns else str(class_id)

        # Description column
        ax_desc = axes[i, 0]
        desc = class_descriptions.get(class_id, class_name)
        ax_desc.text(0.5, 0.5, desc, ha='center', va='center',
                     fontsize=9, wrap=True, transform=ax_desc.transAxes)
        ax_desc.axis('off')

        color = TRANSIENT_COLORS.get(class_id, '#333333')
        ax_desc.set_facecolor(color + '20')

        # Example light curves
        sample_ids = np.random.choice(
            class_objects,
            min(samples_per_class, len(class_objects)),
            replace=False
        )

        for j, obj_id in enumerate(sample_ids):
            ax = axes[i, j + 1]
            obj_lc = lc_df[lc_df['object_id'] == obj_id]

            for pb in sorted(obj_lc['passband'].unique()):
                pb_data = obj_lc[obj_lc['passband'] == pb].sort_values('mjd')
                ax.scatter(
                    pb_data['mjd'],
                    pb_data['flux'],
                    c=PASSBAND_COLORS.get(pb, '#333333'),
                    s=8,
                    alpha=0.7
                )

            ax.set_xlabel('MJD', fontsize=8)
            if j == 0:
                ax.set_ylabel('Flux', fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused columns
        for j in range(len(sample_ids) + 1, samples_per_class + 1):
            axes[i, j].axis('off')

    plt.suptitle('Transient Classification: Example Light Curves', fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# MODEL PERFORMANCE VISUALIZATION
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix with proper formatting.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Class names for axis labels
    normalize : bool
        Whether to normalize by row (true class)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cm[np.isnan(cm)] = 0
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, cmap=cmap)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(class_names, fontsize=9)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            if normalize:
                text = f'{val:.2f}'
            else:
                text = f'{val:d}'
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if val > thresh else 'black', fontsize=8)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation metrics over epochs.

    Parameters
    ----------
    history : dict
        Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(list(history.values())[0]) + 1)

    # Loss plot
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()

    # Accuracy plot
    ax = axes[1]
    if 'train_acc' in history:
        ax.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    if 'val_acc' in history:
        ax.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare performance across different models.

    Parameters
    ----------
    results : dict
        Dictionary mapping model name to metrics dict
    metric : str
        Which metric to compare
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = list(results.keys())
    scores = [results[m][metric] for m in models]

    # Sort by performance
    sorted_idx = np.argsort(scores)[::-1]
    models = [models[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.barh(models, scores, color=colors)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)

    ax.set_xlabel(metric.capitalize())
    ax.set_title(f'Model Comparison: {metric.capitalize()}')
    ax.set_xlim(0, max(scores) * 1.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# ANOMALY DETECTION VISUALIZATION
# =============================================================================

def plot_anomaly_scores(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of anomaly scores.

    Parameters
    ----------
    scores : np.ndarray
        Reconstruction error / anomaly scores
    labels : np.ndarray, optional
        True labels (for coloring)
    threshold : float, optional
        Decision threshold for anomaly detection
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.hist(scores[mask], bins=50, alpha=0.5, label=f'Class {label}')
        ax.legend()
    else:
        ax.hist(scores, bins=50, alpha=0.7, color='steelblue')

    if threshold is not None:
        ax.axvline(threshold, color='red', linestyle='--',
                   label=f'Threshold = {threshold:.4f}')
        ax.legend()

    ax.set_xlabel('Anomaly Score (Reconstruction Error)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Anomaly Scores')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_reconstruction_examples(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_examples: int = 5,
    data_type: str = 'galaxy',
    figsize: Tuple[int, int] = (15, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot original vs reconstructed examples from autoencoder.

    Parameters
    ----------
    original : np.ndarray
        Original data
    reconstructed : np.ndarray
        Reconstructed data
    n_examples : int
        Number of examples to show
    data_type : str
        'galaxy' for images, 'lightcurve' for time series
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(2, n_examples, figsize=figsize)

    for i in range(n_examples):
        if data_type == 'galaxy':
            # Image data
            axes[0, i].imshow(np.clip(original[i], 0, 1))
            axes[1, i].imshow(np.clip(reconstructed[i], 0, 1))
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        else:
            # Light curve data
            for pb in range(original.shape[1]):
                axes[0, i].plot(original[i, pb], color=PASSBAND_COLORS.get(pb, 'gray'), alpha=0.7)
                axes[1, i].plot(reconstructed[i, pb], color=PASSBAND_COLORS.get(pb, 'gray'), alpha=0.7)

    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)

    plt.suptitle('Autoencoder Reconstruction Examples', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_latent_space(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    method: str = 'tsne',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize latent space using dimensionality reduction.

    Parameters
    ----------
    latent_vectors : np.ndarray
        Latent representations (N, latent_dim)
    labels : np.ndarray
        Class labels
    class_names : list, optional
        Names for classes
    method : str
        Reduction method: 'tsne', 'pca', 'umap'
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    from sklearn.decomposition import PCA

    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    coords = reducer.fit_transform(latent_vectors)

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else str(label)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[i]], label=name, alpha=0.7, s=30)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Latent Space Visualization ({method.upper()})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
