"""
EEG Feature Extraction

Functions for extracting features from EEG signals:
- Time-domain features
- Frequency-domain features (band power)
- Spatial features (CSP)
- Statistical features
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.signal import welch
import mne
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler


# Frequency bands
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}


def extract_time_features(epoch_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract time-domain features from epoch data.

    Parameters
    ----------
    epoch_data : np.ndarray
        Epoch data of shape (n_channels, n_times)

    Returns
    -------
    features : dict
        Dictionary of feature arrays (n_channels,)
    """
    features = {
        "mean": np.mean(epoch_data, axis=1),
        "std": np.std(epoch_data, axis=1),
        "var": np.var(epoch_data, axis=1),
        "max": np.max(epoch_data, axis=1),
        "min": np.min(epoch_data, axis=1),
        "ptp": np.ptp(epoch_data, axis=1),  # Peak-to-peak
        "rms": np.sqrt(np.mean(epoch_data ** 2, axis=1)),
        "skewness": stats.skew(epoch_data, axis=1),
        "kurtosis": stats.kurtosis(epoch_data, axis=1),
        "median": np.median(epoch_data, axis=1),
        "iqr": stats.iqr(epoch_data, axis=1),
    }

    # Zero crossings
    zero_crossings = np.sum(np.diff(np.sign(epoch_data), axis=1) != 0, axis=1)
    features["zero_crossings"] = zero_crossings

    return features


def compute_psd(
    epoch_data: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_fft: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Parameters
    ----------
    epoch_data : np.ndarray
        Epoch data of shape (n_channels, n_times)
    sfreq : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    n_fft : int
        FFT length

    Returns
    -------
    freqs : np.ndarray
        Frequency bins
    psd : np.ndarray
        Power spectral density (n_channels, n_freqs)
    """
    n_channels = epoch_data.shape[0]
    psds = []

    for ch in range(n_channels):
        freqs, psd = welch(
            epoch_data[ch],
            fs=sfreq,
            nperseg=min(n_fft, epoch_data.shape[1]),
            noverlap=n_fft // 2
        )
        psds.append(psd)

    psds = np.array(psds)

    # Filter to frequency range
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    psds = psds[:, freq_mask]

    return freqs, psds


def extract_band_power(
    epoch_data: np.ndarray,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    relative: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract band power features.

    Parameters
    ----------
    epoch_data : np.ndarray
        Epoch data of shape (n_channels, n_times)
    sfreq : float
        Sampling frequency
    bands : dict, optional
        Frequency bands {name: (fmin, fmax)}
    relative : bool
        If True, compute relative band power

    Returns
    -------
    features : dict
        Band power for each frequency band
    """
    if bands is None:
        bands = FREQ_BANDS

    freqs, psd = compute_psd(epoch_data, sfreq)
    total_power = np.sum(psd, axis=1)

    features = {}
    for band_name, (fmin, fmax) in bands.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.sum(psd[:, band_mask], axis=1)

        if relative and total_power.sum() > 0:
            band_power = band_power / total_power

        features[f"power_{band_name}"] = band_power

    # Add total power
    features["power_total"] = total_power

    return features


def extract_frequency_features(
    epoch_data: np.ndarray,
    sfreq: float
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive frequency-domain features.

    Parameters
    ----------
    epoch_data : np.ndarray
        Epoch data of shape (n_channels, n_times)
    sfreq : float
        Sampling frequency

    Returns
    -------
    features : dict
        Frequency features
    """
    freqs, psd = compute_psd(epoch_data, sfreq)

    # Band power features
    band_features = extract_band_power(epoch_data, sfreq)

    # Spectral features
    features = dict(band_features)

    # Peak frequency (frequency with maximum power)
    peak_freq_idx = np.argmax(psd, axis=1)
    features["peak_frequency"] = freqs[peak_freq_idx]

    # Spectral entropy
    psd_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)
    features["spectral_entropy"] = spectral_entropy

    # Spectral edge frequency (frequency below which X% of power lies)
    cumsum = np.cumsum(psd, axis=1)
    total = cumsum[:, -1:]
    for percentile in [50, 75, 90, 95]:
        threshold = total * (percentile / 100)
        edge_idx = np.argmax(cumsum >= threshold, axis=1)
        features[f"spectral_edge_{percentile}"] = freqs[edge_idx]

    # Band power ratios
    if "power_alpha" in features and "power_theta" in features:
        features["alpha_theta_ratio"] = (
            features["power_alpha"] / (features["power_theta"] + 1e-10)
        )
    if "power_beta" in features and "power_alpha" in features:
        features["beta_alpha_ratio"] = (
            features["power_beta"] / (features["power_alpha"] + 1e-10)
        )

    return features


def fit_csp(
    epochs: mne.Epochs,
    labels: np.ndarray,
    n_components: int = 6,
    reg: str = "ledoit_wolf"
) -> CSP:
    """
    Fit Common Spatial Patterns (CSP) transformer.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    labels : np.ndarray
        Class labels
    n_components : int
        Number of CSP components
    reg : str
        Regularization method

    Returns
    -------
    csp : CSP
        Fitted CSP transformer
    """
    csp = CSP(
        n_components=n_components,
        reg=reg,
        log=True,
        norm_trace=False
    )

    X = epochs.get_data()
    csp.fit(X, labels)

    return csp


def extract_csp_features(
    epoch_data: np.ndarray,
    csp: CSP
) -> np.ndarray:
    """
    Extract CSP features from epoch data.

    Parameters
    ----------
    epoch_data : np.ndarray
        Epoch data of shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
    csp : CSP
        Fitted CSP transformer

    Returns
    -------
    features : np.ndarray
        CSP features
    """
    # Ensure 3D input
    if epoch_data.ndim == 2:
        epoch_data = epoch_data[np.newaxis, :, :]

    return csp.transform(epoch_data)


def extract_all_features(
    epoch_data: np.ndarray,
    sfreq: float,
    csp: Optional[CSP] = None,
    flatten: bool = True
) -> np.ndarray:
    """
    Extract all features from epoch data.

    Parameters
    ----------
    epoch_data : np.ndarray
        Epoch data of shape (n_channels, n_times)
    sfreq : float
        Sampling frequency
    csp : CSP, optional
        Fitted CSP transformer
    flatten : bool
        If True, flatten features to 1D array

    Returns
    -------
    features : np.ndarray
        Feature vector
    """
    all_features = []
    feature_names = []

    # Time features
    time_feats = extract_time_features(epoch_data)
    for name, values in time_feats.items():
        all_features.append(values)
        feature_names.extend([f"{name}_ch{i}" for i in range(len(values))])

    # Frequency features
    freq_feats = extract_frequency_features(epoch_data, sfreq)
    for name, values in freq_feats.items():
        all_features.append(values)
        feature_names.extend([f"{name}_ch{i}" for i in range(len(values))])

    # CSP features
    if csp is not None:
        csp_feats = extract_csp_features(epoch_data, csp)
        all_features.append(csp_feats.flatten())
        feature_names.extend([f"csp_{i}" for i in range(csp_feats.size)])

    if flatten:
        return np.concatenate([f.flatten() for f in all_features])
    else:
        return all_features, feature_names


def extract_features_from_epochs(
    epochs: mne.Epochs,
    csp: Optional[CSP] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from all epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    csp : CSP, optional
        Fitted CSP transformer

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_epochs, n_features)
    y : np.ndarray
        Labels
    feature_names : list
        Feature names
    """
    X_list = []
    sfreq = epochs.info["sfreq"]

    for epoch_data in epochs.get_data():
        features = extract_all_features(epoch_data, sfreq, csp=csp)
        X_list.append(features)

    X = np.array(X_list)
    y = epochs.events[:, -1]

    # Generate feature names
    n_channels = len(epochs.ch_names)
    time_feat_names = ["mean", "std", "var", "max", "min", "ptp", "rms",
                       "skewness", "kurtosis", "median", "iqr", "zero_crossings"]
    freq_feat_names = [f"power_{b}" for b in FREQ_BANDS.keys()] + [
        "power_total", "peak_frequency", "spectral_entropy",
        "spectral_edge_50", "spectral_edge_75", "spectral_edge_90",
        "spectral_edge_95", "alpha_theta_ratio", "beta_alpha_ratio"
    ]

    feature_names = []
    for name in time_feat_names:
        feature_names.extend([f"{name}_ch{i}" for i in range(n_channels)])
    for name in freq_feat_names:
        feature_names.extend([f"{name}_ch{i}" for i in range(n_channels)])

    if csp is not None:
        feature_names.extend([f"csp_{i}" for i in range(csp.n_components)])

    return X, y, feature_names


def scale_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Scale features using StandardScaler.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray, optional
        Test features

    Returns
    -------
    X_train_scaled : np.ndarray
        Scaled training features
    X_test_scaled : np.ndarray or None
        Scaled test features
    scaler : StandardScaler
        Fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
