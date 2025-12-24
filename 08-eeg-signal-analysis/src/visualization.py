"""
EEG Visualization Functions

Comprehensive plotting utilities for EEG signal analysis:
- Raw signal plots
- Topographic maps
- Time-frequency analysis
- ERD/ERS patterns
- Model performance
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import mne
from mne.time_frequency import tfr_morlet
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'left': '#2ecc71', 'right': '#e74c3c', 'rest': '#3498db'}


def plot_raw_signals(
    raw: mne.io.Raw,
    duration: float = 10.0,
    start: float = 0.0,
    n_channels: int = 10,
    title: str = "Raw EEG Signals",
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot raw EEG signals.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    duration : float
        Duration to plot (seconds)
    start : float
        Start time (seconds)
    n_channels : int
        Number of channels to display
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)

    sfreq = raw.info['sfreq']
    start_idx = int(start * sfreq)
    end_idx = int((start + duration) * sfreq)
    times = np.arange(start_idx, end_idx) / sfreq

    data = raw.get_data()[:n_channels, start_idx:end_idx]
    ch_names = raw.ch_names[:n_channels]

    for i, (ax, ch_data, ch_name) in enumerate(zip(axes, data, ch_names)):
        ax.plot(times, ch_data * 1e6, 'b-', linewidth=0.5)
        ax.set_ylabel(ch_name, fontsize=8, rotation=0, ha='right')
        ax.set_ylim([-100, 100])
        ax.tick_params(axis='y', labelsize=6)

        if i < n_channels - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(title)

    plt.tight_layout()
    return fig


def plot_psd_comparison(
    epochs: mne.Epochs,
    event_id: Dict[str, int],
    fmin: float = 1.0,
    fmax: float = 40.0,
    picks: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot PSD comparison between conditions.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    event_id : dict
        Event mapping
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    picks : list, optional
        Channels to include
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = list(COLORS.values())[:len(event_id)]

    for idx, (event_name, event_code) in enumerate(event_id.items()):
        epochs_cond = epochs[event_name]

        # Compute PSD using MNE
        spectrum = epochs_cond.compute_psd(
            method='welch',
            fmin=fmin,
            fmax=fmax,
            picks=picks
        )

        psds, freqs = spectrum.get_data(return_freqs=True)
        psds_mean = psds.mean(axis=(0, 1))  # Average over epochs and channels
        psds_std = psds.std(axis=(0, 1))

        # Linear scale
        axes[0].plot(freqs, psds_mean * 1e12, color=colors[idx], label=event_name, linewidth=2)
        axes[0].fill_between(
            freqs,
            (psds_mean - psds_std) * 1e12,
            (psds_mean + psds_std) * 1e12,
            color=colors[idx],
            alpha=0.2
        )

        # Log scale
        axes[1].semilogy(freqs, psds_mean * 1e12, color=colors[idx], label=event_name, linewidth=2)

    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power Spectral Density (μV²/Hz)')
    axes[0].set_title('PSD - Linear Scale')
    axes[0].legend()
    axes[0].set_xlim([fmin, fmax])

    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density (μV²/Hz)')
    axes[1].set_title('PSD - Log Scale')
    axes[1].legend()
    axes[1].set_xlim([fmin, fmax])

    # Add frequency band annotations
    bands = {'δ': (1, 4), 'θ': (4, 8), 'α': (8, 13), 'β': (13, 30), 'γ': (30, 40)}
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        for band_name, (fmin_b, fmax_b) in bands.items():
            if fmax_b <= fmax:
                ax.axvspan(fmin_b, fmax_b, alpha=0.1, color='gray')
                ax.text((fmin_b + fmax_b) / 2, ymax * 0.9, band_name,
                        ha='center', fontsize=10, alpha=0.7)

    plt.tight_layout()
    return fig


def plot_topomap(
    epochs: mne.Epochs,
    times: List[float],
    event_name: str,
    band: Tuple[float, float] = (8, 13),
    figsize: Tuple[int, int] = (14, 4)
) -> plt.Figure:
    """
    Plot topographic maps at different time points.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    times : list
        Time points for topomaps
    event_name : str
        Event condition name
    band : tuple
        Frequency band (fmin, fmax)
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    epochs_cond = epochs[event_name]

    fig, axes = plt.subplots(1, len(times), figsize=figsize)
    if len(times) == 1:
        axes = [axes]

    for ax, t in zip(axes, times):
        # Get data around the time point
        tmin = t - 0.1
        tmax = t + 0.1

        epochs_crop = epochs_cond.copy().crop(tmin=max(tmin, epochs_cond.tmin),
                                               tmax=min(tmax, epochs_cond.tmax))

        # Compute band power
        spectrum = epochs_crop.compute_psd(method='welch', fmin=band[0], fmax=band[1])
        band_power = spectrum.get_data().mean(axis=(0, 2))  # Average over epochs and freqs

        # Plot topomap
        mne.viz.plot_topomap(
            band_power,
            epochs.info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            contours=6
        )
        ax.set_title(f't = {t:.2f}s')

    fig.suptitle(f'{event_name} - {band[0]}-{band[1]} Hz Power', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_erp_comparison(
    epochs: mne.Epochs,
    event_id: Dict[str, int],
    picks: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot Event-Related Potential comparison.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    event_id : dict
        Event mapping
    picks : list, optional
        Channels to plot
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    if picks is None:
        # Default: central channels for motor imagery
        picks = ['C3', 'Cz', 'C4']

    # Filter to available channels
    picks = [ch for ch in picks if ch in epochs.ch_names]

    fig, axes = plt.subplots(1, len(picks), figsize=figsize)
    if len(picks) == 1:
        axes = [axes]

    colors = list(COLORS.values())[:len(event_id)]
    times = epochs.times

    for ax, ch_name in zip(axes, picks):
        ch_idx = epochs.ch_names.index(ch_name)

        for idx, (event_name, event_code) in enumerate(event_id.items()):
            epochs_cond = epochs[event_name]
            data = epochs_cond.get_data()[:, ch_idx, :]

            mean = data.mean(axis=0) * 1e6
            sem = data.std(axis=0) / np.sqrt(len(data)) * 1e6

            ax.plot(times, mean, color=colors[idx], label=event_name, linewidth=2)
            ax.fill_between(times, mean - sem, mean + sem, color=colors[idx], alpha=0.2)

        ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title(f'Channel {ch_name}')
        ax.legend()

    fig.suptitle('Event-Related Potentials', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_time_frequency(
    epochs: mne.Epochs,
    event_name: str,
    picks: str = 'C3',
    freqs: np.ndarray = None,
    n_cycles: Union[int, np.ndarray] = 7,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot time-frequency representation using Morlet wavelets.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    event_name : str
        Event condition name
    picks : str
        Channel to plot
    freqs : np.ndarray, optional
        Frequencies to compute
    n_cycles : int or array
        Number of cycles for wavelet
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    if freqs is None:
        freqs = np.arange(2, 35, 1)

    epochs_cond = epochs[event_name]

    # Compute TFR
    power = tfr_morlet(
        epochs_cond,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        picks=picks,
        average=True
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Get channel index
    if isinstance(picks, str):
        ch_idx = 0
    else:
        ch_idx = 0

    # Plot TFR
    data = power.data[ch_idx]

    # Baseline normalize (dB)
    baseline_mask = power.times < 0
    baseline = data[:, baseline_mask].mean(axis=1, keepdims=True)
    data_db = 10 * np.log10(data / baseline)

    # Create symmetric colormap
    vmax = np.abs(data_db).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(
        power.times,
        freqs,
        data_db,
        cmap='RdBu_r',
        norm=norm,
        shading='auto'
    )

    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Time-Frequency: {event_name} - {picks}')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')

    plt.tight_layout()
    return fig


def plot_erd_ers(
    epochs: mne.Epochs,
    event_id: Dict[str, int],
    band: Tuple[float, float] = (8, 13),
    picks: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot Event-Related Desynchronization/Synchronization.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    event_id : dict
        Event mapping
    band : tuple
        Frequency band (fmin, fmax)
    picks : list, optional
        Channels to plot
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    if picks is None:
        picks = ['C3', 'Cz', 'C4']

    picks = [ch for ch in picks if ch in epochs.ch_names]

    fig, axes = plt.subplots(1, len(picks), figsize=figsize)
    if len(picks) == 1:
        axes = [axes]

    freqs = np.arange(band[0], band[1] + 1, 1)
    colors = list(COLORS.values())[:len(event_id)]

    for ax, ch_name in zip(axes, picks):
        for idx, (event_name, event_code) in enumerate(event_id.items()):
            epochs_cond = epochs[event_name]

            # Compute TFR
            power = tfr_morlet(
                epochs_cond,
                freqs=freqs,
                n_cycles=freqs / 2,
                return_itc=False,
                picks=ch_name,
                average=False
            )

            # Get data and average over frequency
            data = power.data[:, 0, :, :].mean(axis=1)  # (n_epochs, n_times)

            # Compute ERD/ERS (% change from baseline)
            baseline_mask = power.times < 0
            baseline = data[:, baseline_mask].mean(axis=1, keepdims=True)
            erd_ers = (data - baseline) / baseline * 100

            mean = erd_ers.mean(axis=0)
            sem = erd_ers.std(axis=0) / np.sqrt(len(erd_ers))

            ax.plot(power.times, mean, color=colors[idx], label=event_name, linewidth=2)
            ax.fill_between(power.times, mean - sem, mean + sem, color=colors[idx], alpha=0.2)

        ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ERD/ERS (%)')
        ax.set_title(f'{ch_name} ({band[0]}-{band[1]} Hz)')
        ax.legend()

    fig.suptitle('Event-Related Desynchronization/Synchronization', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list
        Class names
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Proportion'}
    )

    # Add raw counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j + 0.5, i + 0.7, f'(n={cm[i, j]})',
                    ha='center', va='center', fontsize=9, color='gray')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: List[str],
    title: str = "ROC Curves",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curves for multiclass classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    labels : list
        Class names
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_classes = len(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, (label, color) in enumerate(zip(labels, colors)):
        # Binary labels for this class
        y_binary = (y_true == i).astype(int)

        fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{label} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1', 'auc'],
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot comparison of multiple models.

    Parameters
    ----------
    results : dict
        Dictionary of model results {model_name: {metric: value}}
    metrics : list
        Metrics to compare
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    models = list(results.keys())
    x = np.arange(len(models))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

    for ax, metric in zip(axes, metrics):
        values = [results[model].get(metric, 0) for model in models]

        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim([0, 1])

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance.

    Parameters
    ----------
    feature_names : list
        Feature names
    importance : np.ndarray
        Feature importance values
    top_n : int
        Number of top features to show
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(indices))
    colors = plt.cm.viridis(importance[indices] / importance[indices].max())

    ax.barh(y_pos, importance[indices], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot training history for deep learning models.

    Parameters
    ----------
    history : dict
        Training history {metric: [values]}
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], 'b-', label='Train', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()

    # Accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], 'b-', label='Train', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_csp_patterns(
    csp,
    epochs: mne.Epochs,
    n_components: int = 4,
    figsize: Tuple[int, int] = (12, 3)
) -> plt.Figure:
    """
    Plot CSP spatial patterns.

    Parameters
    ----------
    csp : CSP
        Fitted CSP object
    epochs : mne.Epochs
        Epoched EEG data
    n_components : int
        Number of components to plot
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, n_components, figsize=figsize)

    patterns = csp.patterns_[:n_components]

    for idx, (ax, pattern) in enumerate(zip(axes, patterns)):
        mne.viz.plot_topomap(
            pattern,
            epochs.info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            contours=6
        )
        ax.set_title(f'CSP {idx + 1}')

    fig.suptitle('CSP Spatial Patterns', fontsize=12, y=1.05)
    plt.tight_layout()
    return fig


def create_summary_figure(
    epochs: mne.Epochs,
    event_id: Dict[str, int],
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a summary figure with multiple EEG visualizations.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    event_id : dict
        Event mapping
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Average ERP
    ax1 = fig.add_subplot(gs[0, :])
    times = epochs.times
    colors = list(COLORS.values())[:len(event_id)]

    for idx, (event_name, event_code) in enumerate(event_id.items()):
        epochs_cond = epochs[event_name]
        data = epochs_cond.get_data().mean(axis=(0, 1)) * 1e6
        ax1.plot(times, data, color=colors[idx], label=event_name, linewidth=2)

    ax1.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (μV)')
    ax1.set_title('Grand Average ERP')
    ax1.legend()

    # 2-4. Topomaps at different times
    topomap_times = [0.5, 1.5, 2.5]
    for i, t in enumerate(topomap_times):
        ax = fig.add_subplot(gs[1, i])

        # Get data at this time point
        t_idx = np.argmin(np.abs(epochs.times - t))
        data = epochs.get_data()[:, :, t_idx].mean(axis=0)

        mne.viz.plot_topomap(
            data,
            epochs.info,
            axes=ax,
            show=False,
            cmap='RdBu_r'
        )
        ax.set_title(f't = {t:.1f}s')

    # 5. PSD comparison
    ax5 = fig.add_subplot(gs[2, 0])
    for idx, (event_name, event_code) in enumerate(event_id.items()):
        epochs_cond = epochs[event_name]
        spectrum = epochs_cond.compute_psd(method='welch', fmin=1, fmax=40)
        psds, freqs = spectrum.get_data(return_freqs=True)
        psds_mean = psds.mean(axis=(0, 1)) * 1e12
        ax5.semilogy(freqs, psds_mean, color=colors[idx], label=event_name, linewidth=2)
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('PSD (μV²/Hz)')
    ax5.set_title('Power Spectrum')
    ax5.legend()

    # 6. Epoch count
    ax6 = fig.add_subplot(gs[2, 1])
    counts = [len(epochs[name]) for name in event_id.keys()]
    ax6.bar(event_id.keys(), counts, color=colors[:len(event_id)], edgecolor='black')
    ax6.set_ylabel('Count')
    ax6.set_title('Epochs per Condition')

    # 7. Data info
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    info_text = f"""Dataset Information

Sampling Rate: {epochs.info['sfreq']:.1f} Hz
Channels: {len(epochs.ch_names)}
Time Window: [{epochs.tmin:.2f}, {epochs.tmax:.2f}]s
Total Epochs: {len(epochs)}
    """
    ax7.text(0.1, 0.5, info_text, transform=ax7.transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace')

    fig.suptitle('EEG Data Summary', fontsize=14, y=1.02)
    return fig
