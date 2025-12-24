"""
EEG Preprocessing Pipeline

Functions for cleaning and preparing EEG signals for analysis:
- Filtering (bandpass, notch)
- Artifact removal (ICA)
- Re-referencing
- Epoch extraction
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import mne
from mne.io import read_raw_edf, concatenate_raws
from mne.preprocessing import ICA
from mne.datasets import eegbci


def load_subject_data(
    subject: int,
    runs: List[int],
    data_path: Optional[Path] = None
) -> mne.io.Raw:
    """
    Load and concatenate EEG data for a subject.

    Parameters
    ----------
    subject : int
        Subject ID (1-109)
    runs : list
        List of run numbers to load
    data_path : Path, optional
        Path to data directory

    Returns
    -------
    raw : mne.io.Raw
        Concatenated raw EEG data
    """
    raw_files = eegbci.load_data(
        subject=subject,
        runs=runs,
        path=str(data_path) if data_path else None,
        update_path=False
    )

    raws = [read_raw_edf(f, preload=True) for f in raw_files]
    raw = concatenate_raws(raws)

    # Standardize channel names to 10-20 system
    eegbci.standardize(raw)

    return raw


def set_montage(raw: mne.io.Raw, montage_name: str = "standard_1005") -> mne.io.Raw:
    """
    Set electrode montage for spatial information.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    montage_name : str
        Name of standard montage

    Returns
    -------
    raw : mne.io.Raw
        Raw data with montage set
    """
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage)
    return raw


def apply_filters(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freqs: Optional[List[float]] = None
) -> mne.io.Raw:
    """
    Apply bandpass and notch filters to EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low cutoff frequency (Hz)
    h_freq : float
        High cutoff frequency (Hz)
    notch_freqs : list, optional
        Frequencies for notch filter (e.g., [50, 60] for line noise)

    Returns
    -------
    raw : mne.io.Raw
        Filtered raw data
    """
    # Bandpass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")

    # Notch filter for line noise
    if notch_freqs:
        raw.notch_filter(freqs=notch_freqs)

    return raw


def apply_ica(
    raw: mne.io.Raw,
    n_components: int = 20,
    method: str = "fastica",
    random_state: int = 42,
    exclude_components: Optional[List[int]] = None
) -> Tuple[mne.io.Raw, ICA]:
    """
    Apply ICA for artifact removal.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    n_components : int
        Number of ICA components
    method : str
        ICA method ('fastica', 'infomax', 'picard')
    random_state : int
        Random seed for reproducibility
    exclude_components : list, optional
        Component indices to exclude (artifacts)

    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned raw data
    ica : ICA
        Fitted ICA object
    """
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto"
    )

    # Fit ICA
    ica.fit(raw)

    # Exclude artifact components
    if exclude_components:
        ica.exclude = exclude_components

    # Apply ICA
    raw_clean = ica.apply(raw.copy())

    return raw_clean, ica


def auto_detect_eog_components(
    ica: ICA,
    raw: mne.io.Raw,
    threshold: float = 3.0
) -> List[int]:
    """
    Automatically detect EOG artifact components.

    Parameters
    ----------
    ica : ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw EEG data
    threshold : float
        Z-score threshold for detection

    Returns
    -------
    eog_indices : list
        Indices of EOG components
    """
    # Try to find EOG components using frontal channels
    eog_indices = []

    # Use Fp1 and Fp2 as proxy for EOG
    try:
        eog_inds, scores = ica.find_bads_eog(
            raw,
            ch_name=['Fp1', 'Fp2'],
            threshold=threshold
        )
        eog_indices.extend(eog_inds)
    except Exception:
        # If automatic detection fails, return empty list
        pass

    return list(set(eog_indices))


def set_reference(
    raw: mne.io.Raw,
    ref_type: str = "average"
) -> mne.io.Raw:
    """
    Set EEG reference.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    ref_type : str
        Reference type: 'average', or channel name(s)

    Returns
    -------
    raw : mne.io.Raw
        Re-referenced data
    """
    if ref_type == "average":
        raw.set_eeg_reference("average", projection=False)
    else:
        raw.set_eeg_reference(ref_type)

    return raw


def create_epochs(
    raw: mne.io.Raw,
    event_id: Dict[str, int],
    tmin: float = -0.5,
    tmax: float = 4.0,
    baseline: Tuple[float, float] = (-0.5, 0),
    reject: Optional[Dict[str, float]] = None,
    picks: str = "eeg"
) -> mne.Epochs:
    """
    Create epochs from raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    event_id : dict
        Event name to code mapping
    tmin : float
        Start time before event (seconds)
    tmax : float
        End time after event (seconds)
    baseline : tuple
        Baseline correction window
    reject : dict, optional
        Rejection thresholds (e.g., {'eeg': 100e-6})
    picks : str
        Channel types to include

    Returns
    -------
    epochs : mne.Epochs
        Epoched data
    """
    # Extract events from annotations
    events, _ = mne.events_from_annotations(raw)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        picks=picks
    )

    return epochs


def preprocess_pipeline(
    subject: int,
    runs: List[int],
    data_path: Optional[Path] = None,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    apply_ica_cleaning: bool = True,
    event_id: Optional[Dict[str, int]] = None,
    tmin: float = -0.5,
    tmax: float = 4.0
) -> Tuple[mne.Epochs, Dict]:
    """
    Complete preprocessing pipeline for a subject.

    Parameters
    ----------
    subject : int
        Subject ID
    runs : list
        Run numbers to process
    data_path : Path, optional
        Data directory
    l_freq : float
        Bandpass low frequency
    h_freq : float
        Bandpass high frequency
    apply_ica_cleaning : bool
        Whether to apply ICA artifact removal
    event_id : dict, optional
        Event mapping
    tmin : float
        Epoch start time
    tmax : float
        Epoch end time

    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epochs
    info : dict
        Preprocessing information
    """
    info = {
        "subject": subject,
        "runs": runs,
        "steps": []
    }

    # Default event IDs for motor imagery
    if event_id is None:
        event_id = {"T1": 2, "T2": 3}  # Left and right hand

    # Load data
    raw = load_subject_data(subject, runs, data_path)
    info["steps"].append("load_data")
    info["sfreq"] = raw.info["sfreq"]
    info["n_channels"] = len(raw.ch_names)

    # Set montage
    raw = set_montage(raw)
    info["steps"].append("set_montage")

    # Apply filters
    raw = apply_filters(raw, l_freq=l_freq, h_freq=h_freq)
    info["steps"].append(f"filter_{l_freq}-{h_freq}Hz")

    # Set reference
    raw = set_reference(raw, ref_type="average")
    info["steps"].append("average_reference")

    # Apply ICA
    if apply_ica_cleaning:
        raw, ica = apply_ica(raw, n_components=20)
        eog_components = auto_detect_eog_components(ica, raw)
        if eog_components:
            ica.exclude = eog_components
            raw = ica.apply(raw.copy())
        info["steps"].append(f"ica_components_removed: {len(eog_components)}")

    # Create epochs
    epochs = create_epochs(
        raw,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax
    )
    info["steps"].append("create_epochs")
    info["n_epochs"] = len(epochs)
    info["event_id"] = event_id

    return epochs, info


def load_multiple_subjects(
    subjects: List[int],
    runs: List[int],
    data_path: Optional[Path] = None,
    **kwargs
) -> Tuple[List[mne.Epochs], List[Dict]]:
    """
    Load and preprocess data from multiple subjects.

    Parameters
    ----------
    subjects : list
        List of subject IDs
    runs : list
        Run numbers to process
    data_path : Path, optional
        Data directory
    **kwargs
        Additional arguments for preprocess_pipeline

    Returns
    -------
    all_epochs : list
        List of Epochs objects
    all_info : list
        List of preprocessing info dicts
    """
    all_epochs = []
    all_info = []

    for subject in subjects:
        print(f"Processing subject {subject}...")
        try:
            epochs, info = preprocess_pipeline(
                subject=subject,
                runs=runs,
                data_path=data_path,
                **kwargs
            )
            all_epochs.append(epochs)
            all_info.append(info)
            print(f"  Success: {len(epochs)} epochs")
        except Exception as e:
            print(f"  Error: {e}")
            all_info.append({"subject": subject, "error": str(e)})

    return all_epochs, all_info
