"""
Generate visualization images for the README documentation.

This script creates all the plots and diagrams needed for the
research-paper quality documentation.

Usage:
    python scripts/generate_images.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from pathlib import Path

# Create images directory
IMAGES_DIR = Path(__file__).parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')

# Colors
PASSBAND_COLORS = {
    0: '#8b5cf6', 1: '#3b82f6', 2: '#22c55e',
    3: '#f97316', 4: '#ef4444', 5: '#7f1d1d'
}


def generate_galaxy_examples():
    """Generate example galaxy images for each morphology type."""
    np.random.seed(42)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    galaxy_types = ['Elliptical', 'Spiral', 'Edge-on', 'Irregular']
    descriptions = [
        'Smooth, featureless\nOld stars (red)',
        'Central bulge + spiral arms\nActive star formation',
        'Disk viewed from side\nOften shows dust lane',
        'No regular structure\nOften from interactions'
    ]

    for i, (gtype, desc) in enumerate(zip(galaxy_types, descriptions)):
        # Generate galaxy
        size = 128
        img = np.zeros((size, size, 3))
        y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        if gtype == 'Elliptical':
            a, b = 30, 20
            angle = 0.5
            x_rot = x * np.cos(angle) + y * np.sin(angle)
            y_rot = -x * np.sin(angle) + y * np.cos(angle)
            r_ell = np.sqrt((x_rot/a)**2 + (y_rot/b)**2)
            intensity = np.exp(-7.67 * (r_ell**(1/4) - 1))
            img[:,:,0] = intensity
            img[:,:,1] = intensity * 0.8
            img[:,:,2] = intensity * 0.6

        elif gtype == 'Spiral':
            disk = 0.8 * np.exp(-r/30)
            arms = 0.3 * np.sin(2 * theta - 0.3 * r) * np.exp(-r/40)
            arms = np.clip(arms, 0, 0.3)
            bulge = np.exp(-r**2/100)
            intensity = np.clip(disk + arms + bulge, 0, 1)
            img[:,:,0] = intensity * (0.7 + 0.3*np.exp(-r/20))
            img[:,:,1] = intensity
            img[:,:,2] = intensity * (0.5 + 0.5*(1-np.exp(-r/30)))

        elif gtype == 'Edge-on':
            a, b = 50, 5
            r_ell = np.sqrt((x/a)**2 + (y/b)**2)
            disk = 0.8 * np.exp(-r_ell)
            dust = 1 - 0.7 * np.exp(-y**2/4) * (np.abs(x) < 35)
            intensity = disk * dust
            img[:,:,0] = intensity
            img[:,:,1] = intensity * 0.9
            img[:,:,2] = intensity * 0.7

        else:  # Irregular
            for _ in range(5):
                cx = np.random.randint(30, 98)
                cy = np.random.randint(30, 98)
                sigma = np.random.uniform(8, 15)
                brightness = np.random.uniform(0.4, 0.8)
                clump = brightness * np.exp(-((x-cx+64)**2 + (y-cy+64)**2)/(2*sigma**2))
                channel = np.random.choice([0, 1, 2])
                img[:,:,channel] = np.clip(img[:,:,channel] + clump, 0, 1)

        # Add noise
        noise = np.random.randn(size, size, 3) * 0.03
        img = np.clip(img + noise, 0, 1)

        # Plot
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(gtype, fontsize=14, fontweight='bold')

        # Description
        axes[1, i].text(0.5, 0.5, desc, ha='center', va='center',
                        fontsize=11, transform=axes[1, i].transAxes)
        axes[1, i].axis('off')

    plt.suptitle('Galaxy Morphology Classification', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'galaxy_morphology_types.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: galaxy_morphology_types.png")


def generate_transient_examples():
    """Generate example light curves for different transient types."""
    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    transient_types = [
        ('Type Ia Supernova', 'Thermonuclear explosion\nCosmological standard candle'),
        ('Type II Supernova', 'Core-collapse with H envelope\nShows plateau phase'),
        ('Kilonova', 'Neutron star merger\nVery fast (days), very red'),
        ('RR Lyrae', 'Pulsating variable star\nPeriod < 1 day'),
        ('AGN', 'Active galactic nucleus\nStochastic variability'),
        ('Eclipsing Binary', 'Two stars orbiting\nPeriodic eclipses')
    ]

    for i, (name, desc) in enumerate(transient_types):
        ax = axes[i]

        n_obs = 100
        mjd = np.sort(np.random.uniform(0, 200, n_obs))

        if 'Ia' in name:
            peak = 50
            t = mjd - peak
            flux = 1000 * np.exp(-0.5 * (t/15)**2) * (t < 100)
            flux += 300 * np.exp(-t/40) * (t > 0) * (t < 100)

        elif 'II' in name:
            peak = 50
            t = mjd - peak
            flux = 800 * np.exp(-0.5 * (t/20)**2) * (t < 50)
            flux += 400 * (t >= 50) * (t < 100)
            flux *= np.exp(-t/100) * (t > 0)

        elif 'Kilonova' in name:
            peak = 50
            t = mjd - peak
            flux = 600 * np.exp(-t/2) * (t > 0) * (t < 20)

        elif 'RR Lyrae' in name:
            period = 0.6
            flux = 500 + 200 * np.sin(2 * np.pi * mjd / period)

        elif 'AGN' in name:
            flux = 500 + 150 * np.cumsum(np.random.randn(n_obs)) / np.sqrt(n_obs)

        else:  # Eclipsing Binary
            period = 3
            phase = (mjd % period) / period
            flux = 800 - 200 * (np.abs(phase - 0.5) < 0.08).astype(float)
            flux -= 100 * (np.abs(phase) < 0.04).astype(float)

        flux = np.maximum(flux, 0)
        flux_err = np.sqrt(flux + 50) * 0.5
        flux += np.random.randn(n_obs) * flux_err * 0.3

        # Assign random passbands and plot
        passbands = np.random.choice([0, 1, 2, 3, 4, 5], n_obs)

        for pb in range(6):
            mask = passbands == pb
            if np.sum(mask) > 0:
                ax.scatter(mjd[mask], flux[mask], c=PASSBAND_COLORS[pb],
                          s=20, alpha=0.7, label=f"{'ugrizy'[pb]}")

        ax.set_xlabel('Days', fontsize=10)
        ax.set_ylabel('Flux', fontsize=10)
        ax.set_title(f'{name}\n{desc}', fontsize=11)

        if i == 0:
            ax.legend(title='Band', fontsize=8, loc='upper right')

    plt.suptitle('Transient Light Curve Examples', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'transient_light_curve_types.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: transient_light_curve_types.png")


def generate_architecture_diagram():
    """Generate model architecture comparison diagram."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Colors
    input_color = '#3498db'
    conv_color = '#2ecc71'
    lstm_color = '#9b59b6'
    fc_color = '#e74c3c'
    output_color = '#f39c12'

    # 1. Galaxy CNN
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    boxes = [
        (1, 8, 'Galaxy Image\n(128×128×3)', input_color),
        (1, 6.5, 'Conv Block 1\n32 filters', conv_color),
        (1, 5, 'Conv Block 2\n64 filters', conv_color),
        (1, 3.5, 'Conv Block 3\n128 filters', conv_color),
        (1, 2, 'FC Layer\n512 units', fc_color),
        (1, 0.5, 'Output\n4 classes', output_color),
    ]

    for x, y, text, color in boxes:
        rect = FancyBboxPatch((x, y), 3, 1.2, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+1.5, y+0.6, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows
    for i in range(len(boxes)-1):
        ax.annotate('', xy=(2.5, boxes[i+1][1]+1.2), xytext=(2.5, boxes[i][1]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_title('Galaxy Classification\n(2D CNN)', fontsize=14, fontweight='bold')
    ax.axis('off')

    # 2. Light Curve 1D-CNN
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    boxes = [
        (1, 8, 'Light Curve\n(6 bands × 100 time)', input_color),
        (1, 6.5, '1D Conv\n32 filters, k=7', conv_color),
        (1, 5, '1D Conv\n64 filters, k=5', conv_color),
        (1, 3.5, '1D Conv\n128 filters, k=3', conv_color),
        (1, 2, 'Global Pool + FC\n128 units', fc_color),
        (1, 0.5, 'Output\n14 classes', output_color),
    ]

    for x, y, text, color in boxes:
        rect = FancyBboxPatch((x, y), 3, 1.2, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+1.5, y+0.6, text, ha='center', va='center', fontsize=9, fontweight='bold')

    for i in range(len(boxes)-1):
        ax.annotate('', xy=(2.5, boxes[i+1][1]+1.2), xytext=(2.5, boxes[i][1]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_title('Transient Classification\n(1D CNN)', fontsize=14, fontweight='bold')
    ax.axis('off')

    # 3. Hybrid CNN-LSTM
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    boxes = [
        (1, 8, 'Light Curve\n(6 bands × 100 time)', input_color),
        (1, 6.5, '1D Conv Layers\nLocal features', conv_color),
        (1, 5, 'Bi-LSTM\n128 hidden', lstm_color),
        (1, 3.5, 'Attention\nWeighted sum', lstm_color),
        (1, 2, 'FC Layer\n128 units', fc_color),
        (1, 0.5, 'Output\n14 classes', output_color),
    ]

    for x, y, text, color in boxes:
        rect = FancyBboxPatch((x, y), 3, 1.2, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+1.5, y+0.6, text, ha='center', va='center', fontsize=9, fontweight='bold')

    for i in range(len(boxes)-1):
        ax.annotate('', xy=(2.5, boxes[i+1][1]+1.2), xytext=(2.5, boxes[i][1]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_title('Hybrid Model\n(CNN + LSTM)', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=input_color, label='Input'),
        mpatches.Patch(facecolor=conv_color, label='Convolution'),
        mpatches.Patch(facecolor=lstm_color, label='LSTM/Attention'),
        mpatches.Patch(facecolor=fc_color, label='Fully Connected'),
        mpatches.Patch(facecolor=output_color, label='Output'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)

    plt.suptitle('Model Architectures for Astrophysics CV', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(IMAGES_DIR / 'model_architectures.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: model_architectures.png")


def generate_pipeline_overview():
    """Generate complete pipeline overview diagram."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)

    # Colors
    data_color = '#3498db'
    process_color = '#2ecc71'
    model_color = '#9b59b6'
    output_color = '#e74c3c'

    # Galaxy pipeline (top)
    y_galaxy = 6

    ax.text(0.5, y_galaxy + 1, 'Galaxy Classification Pipeline', fontsize=14,
            fontweight='bold', va='bottom')

    galaxy_boxes = [
        (0.5, y_galaxy, 'Raw\nImages', data_color),
        (3, y_galaxy, 'Preprocessing\n• Background sub\n• Centering\n• Log scale', process_color),
        (6.5, y_galaxy, '2D CNN /\nTransfer\nLearning', model_color),
        (10, y_galaxy, 'Galaxy\nMorphology\nClass', output_color),
    ]

    for x, y, text, color in galaxy_boxes:
        rect = FancyBboxPatch((x, y-0.8), 2.2, 1.6, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+1.1, y, text, ha='center', va='center', fontsize=9)

    for i in range(len(galaxy_boxes)-1):
        ax.annotate('', xy=(galaxy_boxes[i+1][0], y_galaxy),
                   xytext=(galaxy_boxes[i][0]+2.2, y_galaxy),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Transient pipeline (middle)
    y_transient = 3

    ax.text(0.5, y_transient + 1, 'Transient Classification Pipeline', fontsize=14,
            fontweight='bold', va='bottom')

    transient_boxes = [
        (0.5, y_transient, 'Raw Light\nCurves', data_color),
        (3, y_transient, 'Preprocessing\n• Interpolation\n• Normalization\n• Feature extract', process_color),
        (6.5, y_transient, '1D-CNN /\nLSTM /\nHybrid', model_color),
        (10, y_transient, 'Transient\nType\nClass', output_color),
    ]

    for x, y, text, color in transient_boxes:
        rect = FancyBboxPatch((x, y-0.8), 2.2, 1.6, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+1.1, y, text, ha='center', va='center', fontsize=9)

    for i in range(len(transient_boxes)-1):
        ax.annotate('', xy=(transient_boxes[i+1][0], y_transient),
                   xytext=(transient_boxes[i][0]+2.2, y_transient),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Anomaly detection branch
    y_anomaly = 0.5

    ax.text(6.5, y_anomaly + 0.8, 'Anomaly Detection', fontsize=12, fontweight='bold')

    anomaly_box = FancyBboxPatch((6.5, y_anomaly-0.5), 2.2, 1, boxstyle="round,pad=0.1",
                                  facecolor='#f39c12', edgecolor='black', alpha=0.8)
    ax.add_patch(anomaly_box)
    ax.text(7.6, y_anomaly, 'Autoencoder\n(Unsupervised)', ha='center', va='center', fontsize=9)

    # Arrows to anomaly
    ax.annotate('', xy=(6.5, y_anomaly), xytext=(5.2, y_transient-0.8),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))

    output_box = FancyBboxPatch((10, y_anomaly-0.5), 2.2, 1, boxstyle="round,pad=0.1",
                                 facecolor=output_color, edgecolor='black', alpha=0.8)
    ax.add_patch(output_box)
    ax.text(11.1, y_anomaly, 'Novel/Rare\nObjects', ha='center', va='center', fontsize=9)

    ax.annotate('', xy=(10, y_anomaly), xytext=(8.7, y_anomaly),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=data_color, label='Input Data'),
        mpatches.Patch(facecolor=process_color, label='Preprocessing'),
        mpatches.Patch(facecolor=model_color, label='Deep Learning Model'),
        mpatches.Patch(facecolor=output_color, label='Output'),
        mpatches.Patch(facecolor='#f39c12', label='Anomaly Detection'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.axis('off')
    plt.title('Astrophysics Computer Vision: Complete Pipeline', fontsize=16, y=0.98)
    plt.savefig(IMAGES_DIR / 'pipeline_overview.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: pipeline_overview.png")


def generate_why_1d_cnn():
    """Generate diagram explaining why 1D-CNN for light curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Light curve as 1D signal
    ax = axes[0]
    np.random.seed(42)
    t = np.linspace(0, 100, 200)
    flux = 500 * np.exp(-0.5 * ((t-50)/15)**2) + np.random.randn(200) * 30

    ax.plot(t, flux, 'b-', linewidth=2, label='Light curve (1D signal)')
    ax.fill_between(t, flux-50, flux+50, alpha=0.2)

    # Show 1D convolution
    kernel_center = 50
    kernel_width = 15
    ax.axvspan(kernel_center-kernel_width/2, kernel_center+kernel_width/2,
               alpha=0.3, color='red', label='1D conv kernel')

    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title('Light Curve = 1D Time Series\n→ Use 1D Convolutions', fontsize=14)
    ax.legend()

    # Right: Why not 2D
    ax = axes[1]
    ax.text(0.5, 0.8, 'Why NOT 2D-CNN for Light Curves?', fontsize=14,
            fontweight='bold', ha='center', transform=ax.transAxes)

    reasons = [
        '• Light curves are 1D signals, not 2D images',
        '• 2D-CNN would require creating artificial images',
        '  (e.g., spectrograms, recurrence plots)',
        '• This loses the natural temporal ordering',
        '• Adds complexity without adding information',
        '',
        '1D-CNN advantages:',
        '• Directly processes sequential data',
        '• Detects temporal patterns (rise, peak, decline)',
        '• Translation-invariant across time',
        '• Fewer parameters than 2D equivalent'
    ]

    for i, line in enumerate(reasons):
        color = 'darkgreen' if line.startswith('1D') else 'black'
        ax.text(0.1, 0.65 - i*0.06, line, fontsize=11, color=color,
                transform=ax.transAxes)

    ax.axis('off')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'why_1d_cnn.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: why_1d_cnn.png")


def generate_anomaly_detection_diagram():
    """Generate anomaly detection concept diagram."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Normal reconstruction
    ax = axes[0]
    np.random.seed(42)

    # Simulated normal galaxy
    size = 64
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    r = np.sqrt(x**2 + y**2)
    intensity = np.exp(-r**2/200)
    noise = np.random.randn(size, size) * 0.05
    original = np.clip(intensity + noise, 0, 1)
    reconstructed = np.clip(intensity + noise * 0.5, 0, 1)  # Less noise

    ax.imshow(original, cmap='gray')
    ax.set_title('Normal Object\nLow reconstruction error', fontsize=12)
    ax.axis('off')

    # Add error value
    error = np.mean((original - reconstructed)**2)
    ax.text(0.5, -0.1, f'Error: {error:.4f}', fontsize=11, ha='center',
            transform=ax.transAxes, color='green')

    # 2. Anomaly reconstruction
    ax = axes[1]

    # Simulated anomaly (ring galaxy)
    r_ring = 20
    ring = np.exp(-(r-r_ring)**2/20)
    noise = np.random.randn(size, size) * 0.08
    original = np.clip(ring + noise, 0, 1)
    # Poor reconstruction (autoencoder hasn't seen rings)
    reconstructed = np.clip(intensity * 0.5, 0, 1)

    ax.imshow(original, cmap='gray')
    ax.set_title('Anomaly (Ring Galaxy)\nHigh reconstruction error', fontsize=12)
    ax.axis('off')

    error = np.mean((original - reconstructed)**2)
    ax.text(0.5, -0.1, f'Error: {error:.4f}', fontsize=11, ha='center',
            transform=ax.transAxes, color='red')

    # 3. Score distribution
    ax = axes[2]

    normal_scores = np.random.exponential(0.01, 500)
    anomaly_scores = np.random.exponential(0.04, 50) + 0.02

    ax.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='green')
    ax.hist(anomaly_scores, bins=15, alpha=0.7, label='Anomaly', color='red')

    threshold = np.percentile(np.concatenate([normal_scores, anomaly_scores]), 95)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold:.3f})')

    ax.set_xlabel('Reconstruction Error', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Anomaly Score Distribution', fontsize=12)
    ax.legend()

    plt.suptitle('Autoencoder Anomaly Detection', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'anomaly_detection.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: anomaly_detection.png")


def main():
    """Generate all images."""
    print("Generating images for README documentation...\n")

    generate_galaxy_examples()
    generate_transient_examples()
    generate_architecture_diagram()
    generate_pipeline_overview()
    generate_why_1d_cnn()
    generate_anomaly_detection_diagram()

    print(f"\nAll images saved to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
