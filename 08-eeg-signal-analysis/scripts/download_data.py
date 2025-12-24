#!/usr/bin/env python3
"""
Download and prepare EEG Motor Movement/Imagery Dataset.

This script downloads the PhysioNet EEG BCI dataset via MNE-Python
and prepares it for analysis.

Dataset: EEG Motor Movement/Imagery Dataset
Source: https://physionet.org/content/eegmmidb/1.0.0/

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --subjects 5  # Download only 5 subjects
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import numpy as np


def download_eegbci_data(subjects: list, runs: list, data_path: Path) -> dict:
    """
    Download EEG BCI data for specified subjects and runs.

    Parameters
    ----------
    subjects : list
        List of subject IDs (1-109)
    runs : list
        List of run numbers to download
    data_path : Path
        Path to store downloaded data

    Returns
    -------
    dict
        Metadata about downloaded files
    """
    metadata = {
        "dataset": "EEG Motor Movement/Imagery Dataset",
        "source": "PhysioNet via MNE-Python",
        "download_date": datetime.now().isoformat(),
        "subjects": [],
        "runs": runs,
        "total_files": 0
    }

    print(f"Downloading EEG data for {len(subjects)} subjects...")
    print(f"Runs: {runs}")
    print(f"Data path: {data_path}")
    print("-" * 50)

    for subject in subjects:
        print(f"\nSubject {subject:03d}:")
        subject_files = []

        for run in runs:
            try:
                # Download data (returns list of file paths)
                files = eegbci.load_data(
                    subject=subject,
                    runs=[run],
                    path=str(data_path),
                    update_path=False
                )
                subject_files.extend(files)
                print(f"  Run {run:02d}: Downloaded")
            except Exception as e:
                print(f"  Run {run:02d}: Error - {e}")

        metadata["subjects"].append({
            "id": subject,
            "files": [str(f) for f in subject_files],
            "n_files": len(subject_files)
        })
        metadata["total_files"] += len(subject_files)

    return metadata


def create_sample_dataset(data_path: Path, sample_path: Path, n_epochs: int = 50):
    """
    Create a small sample dataset for quick testing.

    Parameters
    ----------
    data_path : Path
        Path to raw data
    sample_path : Path
        Path to save sample data
    n_epochs : int
        Number of epochs to include in sample
    """
    print("\nCreating sample dataset...")

    # Load data from first subject
    subject = 1
    runs = [4, 8, 12]  # Motor imagery runs

    raw_files = eegbci.load_data(subject, runs, path=str(data_path), update_path=False)
    raws = [read_raw_edf(f, preload=True) for f in raw_files]
    raw = concatenate_raws(raws)

    # Standardize channel names
    eegbci.standardize(raw)

    # Set montage
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)

    # Basic preprocessing
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin")

    # Extract events
    events, event_id = mne.events_from_annotations(raw)

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"T1": 2, "T2": 3},  # Left and right hand
        tmin=-0.5,
        tmax=4.0,
        baseline=(-0.5, 0),
        preload=True,
        picks="eeg"
    )

    # Select subset
    if len(epochs) > n_epochs:
        indices = np.random.choice(len(epochs), n_epochs, replace=False)
        epochs = epochs[sorted(indices)]

    # Save sample
    sample_file = sample_path / "sample_epochs.fif"
    epochs.save(sample_file, overwrite=True)
    print(f"Sample saved: {sample_file}")
    print(f"Sample contains {len(epochs)} epochs")

    return sample_file


def main():
    parser = argparse.ArgumentParser(description="Download EEG BCI dataset")
    parser.add_argument(
        "--subjects",
        type=int,
        default=10,
        help="Number of subjects to download (default: 10, max: 109)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory to store downloaded data"
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data_dir
    sample_path = project_root / "data" / "sample"

    # Create directories
    data_path.mkdir(parents=True, exist_ok=True)
    sample_path.mkdir(parents=True, exist_ok=True)

    # Define subjects and runs
    subjects = list(range(1, min(args.subjects + 1, 110)))

    # Runs for different tasks:
    # 3, 7, 11: Open/close left or right fist (real movement)
    # 4, 8, 12: Imagine opening/closing left or right fist (imagery)
    # 5, 9, 13: Open/close both fists or both feet (real)
    # 6, 10, 14: Imagine opening/closing both fists or both feet (imagery)

    runs_motor_imagery = [4, 8, 12]  # Left/right hand imagery
    runs_real_movement = [3, 7, 11]  # Left/right hand real movement

    all_runs = runs_motor_imagery + runs_real_movement

    # Download data
    print("=" * 60)
    print("EEG Motor Movement/Imagery Dataset Downloader")
    print("=" * 60)

    metadata = download_eegbci_data(subjects, all_runs, data_path)

    # Save metadata
    metadata_file = project_root / "data" / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {metadata_file}")

    # Create sample dataset
    create_sample_dataset(data_path, sample_path)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Total subjects: {len(subjects)}")
    print(f"Total files: {metadata['total_files']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
