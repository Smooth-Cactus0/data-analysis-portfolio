"""
Preprocessing module for astronomical data.

This module provides preprocessing functions for:
1. Galaxy images: cropping, normalization, augmentation
2. Light curves: interpolation, feature extraction, normalization

The preprocessing pipeline is designed to handle the unique challenges
of astronomical data:
- High dynamic range (bright cores, faint outskirts)
- Non-uniform sampling in time-series data
- Missing observations due to weather/moon
- Varying signal-to-noise ratios
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Union
from pathlib import Path
import warnings

# Image processing
try:
    from PIL import Image
    import cv2
except ImportError:
    warnings.warn("PIL or OpenCV not installed. Image preprocessing unavailable.")

# Scientific computing
from scipy import ndimage
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# =============================================================================
# GALAXY IMAGE PREPROCESSING
# =============================================================================

class GalaxyPreprocessor:
    """
    Preprocessor for galaxy images from imaging surveys.

    Galaxy images require special handling because:
    1. Galaxies have high dynamic range (log scaling often needed)
    2. Background sky must be subtracted
    3. Objects can appear at any orientation (rotation invariance)
    4. Size varies significantly (need consistent cropping/resizing)

    Parameters
    ----------
    target_size : tuple
        Output image size (height, width)
    normalize : bool
        Whether to normalize pixel values to [0, 1]
    augment : bool
        Whether to apply data augmentation
    log_scale : bool
        Whether to apply logarithmic scaling (helps with dynamic range)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (128, 128),
        normalize: bool = True,
        augment: bool = False,
        log_scale: bool = True
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        self.log_scale = log_scale

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to a galaxy image.

        Parameters
        ----------
        image : np.ndarray
            Input image (H, W, C) or (H, W)

        Returns
        -------
        np.ndarray
            Preprocessed image
        """
        # Ensure 3 channels
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Convert to float
        image = image.astype(np.float32)

        # Background subtraction (estimate from corners)
        image = self._subtract_background(image)

        # Center crop around brightest region
        image = self._center_crop(image)

        # Resize to target size
        image = cv2.resize(image, self.target_size)

        # Logarithmic scaling for dynamic range
        if self.log_scale:
            image = self._log_scale(image)

        # Normalize to [0, 1]
        if self.normalize:
            image = self._normalize(image)

        # Data augmentation
        if self.augment:
            image = self._augment(image)

        return image

    def _subtract_background(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate and subtract sky background.

        Uses corner regions to estimate background level,
        assuming galaxy is centered and corners show sky only.
        """
        h, w = image.shape[:2]
        corner_size = min(h, w) // 10

        # Sample corners
        corners = [
            image[:corner_size, :corner_size],
            image[:corner_size, -corner_size:],
            image[-corner_size:, :corner_size],
            image[-corner_size:, -corner_size:]
        ]

        # Median background estimate (robust to cosmic rays)
        background = np.median(np.concatenate([c.flatten() for c in corners]))

        # Subtract and clip
        image = image - background
        image = np.clip(image, 0, None)

        return image

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image to center on brightest region (galaxy core).
        """
        h, w = image.shape[:2]

        # Find center of mass of brightness
        gray = np.mean(image, axis=2) if image.ndim == 3 else image

        # Use threshold to find bright regions
        threshold = np.percentile(gray, 90)
        binary = gray > threshold

        if np.sum(binary) > 0:
            # Calculate centroid
            cy, cx = ndimage.center_of_mass(binary)
            cy, cx = int(cy), int(cx)
        else:
            cy, cx = h // 2, w // 2

        # Crop around centroid
        crop_size = min(h, w) // 2
        y1 = max(0, cy - crop_size)
        y2 = min(h, cy + crop_size)
        x1 = max(0, cx - crop_size)
        x2 = min(w, cx + crop_size)

        return image[y1:y2, x1:x2]

    def _log_scale(self, image: np.ndarray) -> np.ndarray:
        """
        Apply logarithmic scaling to compress dynamic range.

        Galaxies have bright cores and faint halos/arms.
        Log scaling helps visualize and learn from both.
        """
        # Asinh (inverse hyperbolic sine) is commonly used in astronomy
        # It behaves like log for large values but linear near zero
        return np.arcsinh(image / 10) * 10

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] range."""
        min_val = np.min(image)
        max_val = np.max(image)

        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)

        return image

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation.

        For galaxies, we can use:
        - Rotation (any angle, galaxies have no preferred orientation)
        - Flips (horizontal and vertical)
        - Small translations

        We avoid:
        - Heavy scaling (changes galaxy size which is meaningful)
        - Color jittering (colors are physically meaningful)
        """
        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)

        # Random flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        if np.random.random() > 0.5:
            image = np.flipud(image)

        return image

    def batch_preprocess(
        self,
        images: List[np.ndarray],
        n_jobs: int = -1
    ) -> np.ndarray:
        """
        Preprocess a batch of images.

        Parameters
        ----------
        images : list of np.ndarray
            List of input images
        n_jobs : int
            Number of parallel jobs (-1 for all cores)

        Returns
        -------
        np.ndarray
            Batch of preprocessed images (N, H, W, C)
        """
        preprocessed = [self.preprocess(img) for img in images]
        return np.stack(preprocessed, axis=0)


# =============================================================================
# LIGHT CURVE PREPROCESSING
# =============================================================================

class LightCurvePreprocessor:
    """
    Preprocessor for astronomical light curves (time-series photometry).

    Light curves present unique challenges:
    1. Non-uniform time sampling (observations depend on weather, moon phase)
    2. Gaps in coverage (seasonal gaps, telescope downtime)
    3. Multiple passbands with different sensitivities
    4. Varying observation lengths between objects

    Our approach:
    - Interpolate to a regular grid for CNN/LSTM input
    - Extract statistical features for classical ML
    - Normalize flux to make comparison across objects possible

    Parameters
    ----------
    n_time_bins : int
        Number of time bins for interpolated light curve
    passbands : list
        List of passbands to process (e.g., [0, 1, 2, 3, 4, 5] for ugrizy)
    normalize : bool
        Whether to normalize flux values
    """

    def __init__(
        self,
        n_time_bins: int = 100,
        passbands: List[int] = None,
        normalize: bool = True
    ):
        self.n_time_bins = n_time_bins
        self.passbands = passbands if passbands else [0, 1, 2, 3, 4, 5]
        self.normalize = normalize

    def preprocess(
        self,
        lc_df: pd.DataFrame,
        object_id: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess a single light curve.

        Parameters
        ----------
        lc_df : pd.DataFrame
            Light curve data with columns: mjd, passband, flux, flux_err
        object_id : int, optional
            Object ID for filtering (if df contains multiple objects)

        Returns
        -------
        dict
            Dictionary with 'interpolated' and 'features' arrays
        """
        if object_id is not None:
            lc_df = lc_df[lc_df['object_id'] == object_id]

        # Interpolate to regular grid
        interpolated = self._interpolate(lc_df)

        # Extract statistical features
        features = self._extract_features(lc_df)

        return {
            'interpolated': interpolated,
            'features': features
        }

    def _interpolate(self, lc_df: pd.DataFrame) -> np.ndarray:
        """
        Interpolate light curve to regular time grid.

        Returns a 2D array: (n_passbands, n_time_bins)
        Each passband is interpolated separately.
        """
        # Get time range
        t_min = lc_df['mjd'].min()
        t_max = lc_df['mjd'].max()
        t_grid = np.linspace(t_min, t_max, self.n_time_bins)

        interpolated = np.zeros((len(self.passbands), self.n_time_bins))

        for i, pb in enumerate(self.passbands):
            pb_data = lc_df[lc_df['passband'] == pb]

            if len(pb_data) < 2:
                # Not enough points - leave as zeros
                continue

            # Sort by time
            pb_data = pb_data.sort_values('mjd')

            # Interpolate
            try:
                f = interp1d(
                    pb_data['mjd'].values,
                    pb_data['flux'].values,
                    kind='linear',
                    bounds_error=False,
                    fill_value=0
                )
                interpolated[i] = f(t_grid)
            except Exception:
                continue

        # Normalize
        if self.normalize:
            max_flux = np.max(np.abs(interpolated))
            if max_flux > 0:
                interpolated = interpolated / max_flux

        return interpolated

    def _extract_features(self, lc_df: pd.DataFrame) -> np.ndarray:
        """
        Extract statistical features from light curve.

        These features capture:
        - Variability (amplitude, standard deviation)
        - Shape (skewness, kurtosis)
        - Time behavior (duration, rise time)
        - Color (flux ratios between bands)

        Returns
        -------
        np.ndarray
            Feature vector
        """
        features = []

        # Global features
        flux = lc_df['flux'].values
        mjd = lc_df['mjd'].values

        features.extend([
            np.mean(flux),           # Mean flux
            np.std(flux),            # Flux variability
            np.max(flux) - np.min(flux),  # Amplitude
            np.median(flux),         # Median flux
            mjd.max() - mjd.min(),   # Duration
        ])

        # Peak-finding features
        if np.max(flux) > 0:
            peak_idx = np.argmax(flux)
            peak_mjd = mjd[peak_idx]

            # Time from start to peak
            features.append(peak_mjd - mjd.min())

            # Time from peak to end
            features.append(mjd.max() - peak_mjd)
        else:
            features.extend([0, 0])

        # Per-passband features
        for pb in self.passbands:
            pb_data = lc_df[lc_df['passband'] == pb]

            if len(pb_data) > 0:
                pb_flux = pb_data['flux'].values
                features.extend([
                    np.mean(pb_flux),
                    np.std(pb_flux),
                    np.max(pb_flux) if len(pb_flux) > 0 else 0
                ])
            else:
                features.extend([0, 0, 0])

        # Color features (flux ratios between bands)
        pb_means = {}
        for pb in self.passbands:
            pb_data = lc_df[lc_df['passband'] == pb]
            pb_means[pb] = np.mean(pb_data['flux']) if len(pb_data) > 0 else 0

        # Add color ratios (e.g., g-r, r-i)
        for i in range(len(self.passbands) - 1):
            if pb_means[self.passbands[i+1]] > 0:
                ratio = pb_means[self.passbands[i]] / pb_means[self.passbands[i+1]]
                features.append(np.log10(max(ratio, 0.01)))
            else:
                features.append(0)

        return np.array(features)

    def batch_preprocess(
        self,
        lc_df: pd.DataFrame,
        object_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess multiple light curves.

        Parameters
        ----------
        lc_df : pd.DataFrame
            DataFrame with all light curves
        object_ids : list
            List of object IDs to process

        Returns
        -------
        tuple
            (interpolated_batch, features_batch)
        """
        interpolated_list = []
        features_list = []

        for obj_id in object_ids:
            result = self.preprocess(lc_df, object_id=obj_id)
            interpolated_list.append(result['interpolated'])
            features_list.append(result['features'])

        return np.stack(interpolated_list), np.stack(features_list)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_galaxy_image(filepath: Union[str, Path]) -> np.ndarray:
    """Load a galaxy image from disk."""
    image = Image.open(filepath)
    return np.array(image)


def load_light_curves(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load light curve data from CSV."""
    return pd.read_csv(filepath)


def train_test_split_stratified(
    data: Union[np.ndarray, pd.DataFrame],
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple:
    """
    Stratified train-test split for imbalanced classes.

    Important for astronomical classification where some
    classes (e.g., kilonovae) are rare.
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        data, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced classification.

    Rare transient types (kilonovae, TDEs) need higher weight
    to prevent the model from ignoring them.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )

    return dict(zip(classes, weights))


# =============================================================================
# DATA AUGMENTATION FOR ASTRONOMY
# =============================================================================

class AstronomyAugmenter:
    """
    Data augmentation strategies specific to astronomy.

    Key insight: Not all augmentations are valid for astronomy!

    Valid augmentations:
    - Rotation: Galaxies have no preferred orientation on sky
    - Flips: No physical difference between mirror images
    - Small additive noise: Simulates varying seeing conditions
    - PSF blurring: Simulates atmospheric effects

    INVALID augmentations:
    - Strong color jittering: Colors are physically meaningful!
    - Large scaling: Size correlates with distance/luminosity
    - Cropping that removes galaxy: Loses morphological info

    For light curves:
    - Time shifting: Changes phase, acceptable for periodic sources
    - Adding noise: Simulates varying conditions
    - Dropping observations: Simulates realistic gaps

    INVALID for light curves:
    - Flux scaling: Absolute flux is meaningful
    - Time stretching: Changes physical timescales
    """

    def __init__(self, mode: str = 'galaxy'):
        """
        Parameters
        ----------
        mode : str
            Either 'galaxy' or 'lightcurve'
        """
        self.mode = mode

    def augment_galaxy(self, image: np.ndarray) -> np.ndarray:
        """Apply valid augmentations to galaxy image."""
        # Random rotation
        angle = np.random.uniform(0, 360)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

        # Random flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)

        # Small Gaussian noise (simulates sky background variation)
        noise_level = np.random.uniform(0, 0.02)
        noise = np.random.randn(*image.shape) * noise_level
        image = np.clip(image + noise, 0, 1)

        # Slight blur (simulates seeing variation)
        if np.random.random() > 0.7:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        return image

    def augment_lightcurve(
        self,
        lc: np.ndarray,
        drop_fraction: float = 0.1
    ) -> np.ndarray:
        """
        Apply valid augmentations to light curve.

        Parameters
        ----------
        lc : np.ndarray
            Interpolated light curve (n_bands, n_time)
        drop_fraction : float
            Fraction of time points to randomly drop

        Returns
        -------
        np.ndarray
            Augmented light curve
        """
        augmented = lc.copy()

        # Random observation dropout (simulates weather gaps)
        n_drop = int(lc.shape[1] * drop_fraction)
        drop_indices = np.random.choice(lc.shape[1], n_drop, replace=False)
        augmented[:, drop_indices] = 0

        # Add realistic noise
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.randn(*augmented.shape) * noise_level
        augmented = augmented + noise

        return augmented
