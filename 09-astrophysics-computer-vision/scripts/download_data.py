"""
Download astronomical datasets from Kaggle.

This script downloads:
1. Galaxy Zoo 2 dataset (galaxy morphology classification)
2. PLAsTiCC dataset (transient classification)

Usage:
    python download_data.py --dataset all
    python download_data.py --dataset galaxy
    python download_data.py --dataset plasticc
    python download_data.py --sample  # Download only sample data

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Set up Kaggle API credentials:
       - Go to kaggle.com -> Account -> Create New API Token
       - Save kaggle.json to ~/.kaggle/kaggle.json
       - chmod 600 ~/.kaggle/kaggle.json
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Data directories
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"


def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        # Check environment variables
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            print("ERROR: Kaggle API credentials not found!")
            print("\nTo set up Kaggle API:")
            print("1. Go to kaggle.com -> Account -> Create New API Token")
            print("2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json")
            print("   OR set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
            return False

    return True


def download_galaxy_zoo(sample_only: bool = False):
    """
    Download Galaxy Zoo 2 dataset from Kaggle.

    The Galaxy Zoo project crowdsourced morphological classifications
    of galaxies from the Sloan Digital Sky Survey (SDSS).

    Args:
        sample_only: If True, download only a small sample
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    print("\n" + "="*60)
    print("GALAXY ZOO 2 DATASET")
    print("="*60)
    print("\nAbout this dataset:")
    print("- 61,578 galaxy images from SDSS")
    print("- 424x424 pixel RGB images")
    print("- Morphological classifications from citizen scientists")
    print("- Classes include: Elliptical, Spiral, Edge-on, Merger, etc.")

    galaxy_dir = RAW_DIR / "galaxy_zoo"
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading to: {galaxy_dir}")

    api = KaggleApi()
    api.authenticate()

    if sample_only:
        # Download just a subset for demo purposes
        print("\nDownloading sample images only...")
        api.competition_download_file(
            "galaxy-zoo-the-galaxy-challenge",
            "images_training_rev1.zip",
            path=str(galaxy_dir)
        )
    else:
        # Download full dataset
        print("\nDownloading full dataset (this may take a while)...")
        api.competition_download_files(
            "galaxy-zoo-the-galaxy-challenge",
            path=str(galaxy_dir)
        )

    # Extract zip files
    for zip_file in galaxy_dir.glob("*.zip"):
        print(f"\nExtracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(galaxy_dir)
        # Optionally remove zip after extraction
        # zip_file.unlink()

    print("\nGalaxy Zoo download complete!")
    return galaxy_dir


def download_plasticc(sample_only: bool = False):
    """
    Download PLAsTiCC dataset from Kaggle.

    PLAsTiCC (Photometric LSST Astronomical Time-Series Classification Challenge)
    contains simulated light curves for various transient and variable objects
    as they would be observed by the Vera C. Rubin Observatory (LSST).

    Args:
        sample_only: If True, download only test set (smaller)
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    print("\n" + "="*60)
    print("PLAsTiCC DATASET")
    print("="*60)
    print("\nAbout this dataset:")
    print("- Simulated LSST light curves")
    print("- 6 passbands: u, g, r, i, z, y")
    print("- 14 transient/variable classes")
    print("- Time-series flux measurements with uncertainties")
    print("\nClasses include:")
    print("  - Type Ia Supernovae (cosmological standard candles)")
    print("  - Core-collapse Supernovae (II, Ibc)")
    print("  - Kilonovae (neutron star mergers)")
    print("  - Active Galactic Nuclei (AGN)")
    print("  - RR Lyrae (pulsating variable stars)")
    print("  - And more...")

    plasticc_dir = RAW_DIR / "plasticc"
    plasticc_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading to: {plasticc_dir}")

    api = KaggleApi()
    api.authenticate()

    if sample_only:
        # Download just test set (smaller)
        print("\nDownloading sample data...")
        files_to_download = ["test_set_sample.csv", "test_set_metadata.csv"]
        for f in files_to_download:
            try:
                api.competition_download_file(
                    "PLAsTiCC-2018",
                    f,
                    path=str(plasticc_dir)
                )
            except Exception as e:
                print(f"Could not download {f}: {e}")
    else:
        # Download full dataset
        print("\nDownloading full dataset...")
        api.competition_download_files(
            "PLAsTiCC-2018",
            path=str(plasticc_dir)
        )

    # Extract zip files
    for zip_file in plasticc_dir.glob("*.zip"):
        print(f"\nExtracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(plasticc_dir)

    print("\nPLAsTiCC download complete!")
    return plasticc_dir


def create_sample_data():
    """Create small sample datasets for demos."""
    import numpy as np
    from PIL import Image

    print("\n" + "="*60)
    print("CREATING SAMPLE DATA")
    print("="*60)

    sample_galaxy_dir = SAMPLE_DIR / "galaxies"
    sample_transient_dir = SAMPLE_DIR / "transients"
    sample_galaxy_dir.mkdir(parents=True, exist_ok=True)
    sample_transient_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic galaxy images for demo
    print("\nGenerating sample galaxy images...")
    np.random.seed(42)

    galaxy_types = {
        "elliptical": lambda: create_elliptical_galaxy(),
        "spiral": lambda: create_spiral_galaxy(),
        "edge_on": lambda: create_edge_on_galaxy(),
        "irregular": lambda: create_irregular_galaxy()
    }

    for gtype, generator in galaxy_types.items():
        for i in range(5):
            img = generator()
            img_path = sample_galaxy_dir / f"{gtype}_{i+1}.png"
            Image.fromarray(img).save(img_path)

    print(f"  Created {4*5} sample galaxy images in {sample_galaxy_dir}")

    # Create synthetic light curves for demo
    print("\nGenerating sample light curves...")
    import pandas as pd

    transient_types = {
        90: "SNIa",
        67: "SNIa-91bg",
        52: "SNIax",
        42: "SNII",
        62: "SNIbc",
        95: "SLSN-I",
        15: "TDE",
        64: "KN",
        88: "AGN",
        92: "RRLyrae",
        65: "M-dwarf",
        16: "EB",
        53: "Mira",
        6: "Lens-Single"
    }

    light_curves = []
    for class_id, class_name in transient_types.items():
        for obj_id in range(10):  # 10 samples per class
            lc = create_synthetic_light_curve(class_id, obj_id)
            light_curves.append(lc)

    lc_df = pd.concat(light_curves, ignore_index=True)
    lc_df.to_csv(sample_transient_dir / "sample_light_curves.csv", index=False)

    # Create metadata
    metadata = []
    for class_id, class_name in transient_types.items():
        for obj_id in range(10):
            metadata.append({
                "object_id": class_id * 1000 + obj_id,
                "target": class_id,
                "class_name": class_name,
                "hostgal_photoz": np.random.uniform(0, 2),
                "mwebv": np.random.uniform(0, 0.5)
            })

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(sample_transient_dir / "sample_metadata.csv", index=False)

    print(f"  Created {len(transient_types)*10} sample light curves in {sample_transient_dir}")

    print("\nSample data creation complete!")


def create_elliptical_galaxy():
    """Generate synthetic elliptical galaxy image."""
    import numpy as np

    size = 128
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Create elliptical profile
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]

    # Random ellipticity
    a = np.random.uniform(20, 40)
    b = np.random.uniform(10, a)
    angle = np.random.uniform(0, np.pi)

    x_rot = x * np.cos(angle) + y * np.sin(angle)
    y_rot = -x * np.sin(angle) + y * np.cos(angle)

    r = np.sqrt((x_rot/a)**2 + (y_rot/b)**2)

    # Sersic profile (n~4 for ellipticals)
    intensity = 255 * np.exp(-7.67 * (r**(1/4) - 1))
    intensity = np.clip(intensity, 0, 255).astype(np.uint8)

    # Ellipticals are reddish
    img[:,:,0] = intensity  # Red
    img[:,:,1] = (intensity * 0.8).astype(np.uint8)  # Green
    img[:,:,2] = (intensity * 0.6).astype(np.uint8)  # Blue

    # Add noise
    noise = np.random.randint(0, 20, (size, size, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise - 10, 0, 255).astype(np.uint8)

    return img


def create_spiral_galaxy():
    """Generate synthetic spiral galaxy image."""
    import numpy as np

    size = 128
    img = np.zeros((size, size, 3), dtype=np.uint8)

    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Disk profile
    disk = 200 * np.exp(-r/30)

    # Spiral arms
    n_arms = np.random.choice([2, 4])
    arm_tightness = np.random.uniform(0.2, 0.5)
    arms = 80 * np.sin(n_arms * theta - arm_tightness * r) * np.exp(-r/40)
    arms = np.clip(arms, 0, 80)

    # Bulge
    bulge = 255 * np.exp(-r**2/100)

    intensity = np.clip(disk + arms + bulge, 0, 255).astype(np.uint8)

    # Spirals are blueish in arms, yellow in center
    img[:,:,0] = np.clip(intensity * (0.7 + 0.3*np.exp(-r/20)), 0, 255).astype(np.uint8)
    img[:,:,1] = intensity
    img[:,:,2] = np.clip(intensity * (0.5 + 0.5*(1-np.exp(-r/30))), 0, 255).astype(np.uint8)

    # Add noise
    noise = np.random.randint(0, 15, (size, size, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise - 7, 0, 255).astype(np.uint8)

    return img


def create_edge_on_galaxy():
    """Generate synthetic edge-on galaxy image."""
    import numpy as np

    size = 128
    img = np.zeros((size, size, 3), dtype=np.uint8)

    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]

    # Edge-on disk (very flattened)
    a = np.random.uniform(40, 55)
    b = np.random.uniform(3, 8)
    angle = np.random.uniform(-0.3, 0.3)

    x_rot = x * np.cos(angle) + y * np.sin(angle)
    y_rot = -x * np.sin(angle) + y * np.cos(angle)

    r = np.sqrt((x_rot/a)**2 + (y_rot/b)**2)

    # Disk profile
    disk = 200 * np.exp(-r)

    # Dust lane (dark stripe)
    dust = 1 - 0.7 * np.exp(-y_rot**2/4) * (np.abs(x_rot) < 35)

    intensity = (disk * dust).astype(np.uint8)

    # Edge-on galaxies with dust
    img[:,:,0] = intensity
    img[:,:,1] = (intensity * 0.9).astype(np.uint8)
    img[:,:,2] = (intensity * 0.7).astype(np.uint8)

    # Add noise
    noise = np.random.randint(0, 15, (size, size, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise - 7, 0, 255).astype(np.uint8)

    return img


def create_irregular_galaxy():
    """Generate synthetic irregular galaxy image."""
    import numpy as np

    size = 128
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Multiple random clumps
    n_clumps = np.random.randint(3, 7)

    for _ in range(n_clumps):
        cx = np.random.randint(size//4, 3*size//4)
        cy = np.random.randint(size//4, 3*size//4)
        sigma = np.random.uniform(5, 15)
        brightness = np.random.randint(100, 200)

        y, x = np.ogrid[0:size, 0:size]
        clump = brightness * np.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))

        # Random color (blue for star-forming regions)
        color_idx = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
        img[:,:,color_idx] = np.clip(img[:,:,color_idx] + clump, 0, 255).astype(np.uint8)

    # Add noise
    noise = np.random.randint(0, 20, (size, size, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise - 10, 0, 255).astype(np.uint8)

    return img


def create_synthetic_light_curve(class_id: int, obj_id: int):
    """Generate synthetic light curve for a transient class."""
    import numpy as np
    import pandas as pd

    object_id = class_id * 1000 + obj_id

    # Time points (simulating LSST cadence)
    n_obs = np.random.randint(50, 200)
    mjd = np.sort(np.random.uniform(59000, 60000, n_obs))

    # Passbands (LSST ugrizy)
    passbands = np.random.choice([0, 1, 2, 3, 4, 5], n_obs)

    # Generate flux based on class
    if class_id in [90, 67, 52]:  # Type Ia SNe
        peak_mjd = np.random.uniform(59300, 59700)
        t = mjd - peak_mjd
        peak_flux = np.random.uniform(500, 2000)
        flux = peak_flux * np.exp(-0.5 * (t/15)**2) * (t < 100)
        flux += peak_flux * 0.3 * np.exp(-t/40) * (t > 0) * (t < 100)

    elif class_id in [42, 62]:  # Core-collapse SNe
        peak_mjd = np.random.uniform(59300, 59700)
        t = mjd - peak_mjd
        peak_flux = np.random.uniform(300, 1500)
        # Plateau phase for Type II
        flux = peak_flux * np.exp(-0.5 * (t/20)**2) * (t < 50)
        flux += peak_flux * 0.5 * (t >= 50) * (t < 100)
        flux *= np.exp(-t/100) * (t > 0)

    elif class_id == 88:  # AGN (stochastic variability)
        base_flux = np.random.uniform(200, 800)
        flux = base_flux + 100 * np.cumsum(np.random.randn(n_obs)) / np.sqrt(n_obs)

    elif class_id == 92:  # RR Lyrae (periodic)
        period = np.random.uniform(0.4, 0.9)  # days
        amplitude = np.random.uniform(100, 300)
        flux = 500 + amplitude * np.sin(2 * np.pi * mjd / period)

    elif class_id == 64:  # Kilonova (fast transient)
        peak_mjd = np.random.uniform(59300, 59700)
        t = mjd - peak_mjd
        peak_flux = np.random.uniform(200, 800)
        flux = peak_flux * np.exp(-t/2) * (t > 0) * (t < 20)

    else:  # Other classes
        peak_mjd = np.random.uniform(59300, 59700)
        t = mjd - peak_mjd
        peak_flux = np.random.uniform(200, 1000)
        flux = peak_flux * np.exp(-0.5 * (t/30)**2) * (t < 150)

    flux = np.maximum(flux, 0)

    # Add noise
    flux_err = np.sqrt(flux + 100) * np.random.uniform(0.8, 1.2, n_obs)
    flux += np.random.randn(n_obs) * flux_err

    # Passband-dependent color
    color_offset = (passbands - 2) * 50 * np.random.uniform(0.5, 1.5)
    flux += color_offset
    flux = np.maximum(flux, 1)

    return pd.DataFrame({
        "object_id": object_id,
        "mjd": mjd,
        "passband": passbands,
        "flux": flux,
        "flux_err": flux_err,
        "detected": (flux > 3 * flux_err).astype(int)
    })


def main():
    parser = argparse.ArgumentParser(
        description="Download astronomical datasets for the astrophysics CV project"
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "galaxy", "plasticc"],
        default="all",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create synthetic sample data only (no Kaggle download)"
    )
    parser.add_argument(
        "--sample-only-kaggle",
        action="store_true",
        help="Download only sample/subset from Kaggle"
    )

    args = parser.parse_args()

    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if args.sample:
        # Generate synthetic sample data (no Kaggle needed)
        create_sample_data()
        return

    # Check Kaggle credentials
    if not check_kaggle_credentials():
        print("\nFalling back to synthetic sample data...")
        create_sample_data()
        return

    # Download datasets
    if args.dataset in ["all", "galaxy"]:
        download_galaxy_zoo(sample_only=args.sample_only_kaggle)

    if args.dataset in ["all", "plasticc"]:
        download_plasticc(sample_only=args.sample_only_kaggle)

    # Always create sample data for demos
    create_sample_data()

    print("\n" + "="*60)
    print("ALL DOWNLOADS COMPLETE")
    print("="*60)
    print(f"\nData saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
