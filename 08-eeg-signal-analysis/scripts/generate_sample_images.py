"""
Generate sample visualization images for the EEG project.
These are representative examples of what the analysis produces.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_raw_signals():
    """Generate sample raw EEG signals plot."""
    fig, axes = plt.subplots(8, 1, figsize=(14, 10), sharex=True)

    channels = ['Fc3', 'Fc4', 'C3', 'Cz', 'C4', 'Cp3', 'Cp4', 'Pz']
    t = np.linspace(0, 5, 800)  # 5 seconds at 160 Hz

    for i, (ax, ch) in enumerate(zip(axes, channels)):
        # Simulate EEG: mix of frequencies + noise
        signal = (10 * np.sin(2 * np.pi * 10 * t) +  # Alpha
                  5 * np.sin(2 * np.pi * 20 * t) +   # Beta
                  np.random.randn(len(t)) * 15)      # Noise
        ax.plot(t, signal, 'b-', linewidth=0.5)
        ax.set_ylabel(ch, rotation=0, ha='right', fontsize=10)
        ax.set_ylim([-80, 80])
        if i < 7:
            ax.set_xticklabels([])

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Raw EEG Signals (Sensorimotor Cortex)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'raw_signals.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: raw_signals.png")

def generate_psd_comparison():
    """Generate PSD comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    freqs = np.linspace(1, 40, 100)

    # Simulate PSD with alpha peak
    psd_left = 10 / (1 + (freqs - 10)**2 / 20) + 2 / (1 + (freqs - 20)**2 / 30) + 0.5
    psd_right = 8 / (1 + (freqs - 10)**2 / 25) + 2.5 / (1 + (freqs - 20)**2 / 25) + 0.5

    colors = {'Left Hand': '#2ecc71', 'Right Hand': '#e74c3c'}

    for ax in axes:
        ax.plot(freqs, psd_left, color=colors['Left Hand'], label='Left Hand', linewidth=2)
        ax.plot(freqs, psd_right, color=colors['Right Hand'], label='Right Hand', linewidth=2)

        # Band annotations
        bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
        for band_name, (fmin, fmax) in bands.items():
            ax.axvspan(fmin, fmax, alpha=0.1, color='gray')

    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power (μV²/Hz)')
    axes[0].set_title('PSD - Linear Scale')
    axes[0].legend()

    axes[1].semilogy(freqs, psd_left, color=colors['Left Hand'], label='Left Hand', linewidth=2)
    axes[1].semilogy(freqs, psd_right, color=colors['Right Hand'], label='Right Hand', linewidth=2)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power (μV²/Hz)')
    axes[1].set_title('PSD - Log Scale')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'psd_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: psd_comparison.png")

def generate_erp_comparison():
    """Generate ERP comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    channels = ['C3', 'Cz', 'C4']
    times = np.linspace(-0.5, 4, 720)
    colors = {'Left Hand': '#2ecc71', 'Right Hand': '#e74c3c'}

    for ax, ch in zip(axes, channels):
        for label, color in colors.items():
            # Simulate ERP with movement-related potential
            baseline = np.zeros_like(times)
            # Add movement-related negativity
            mrn = -5 * np.exp(-((times - 0.5)**2) / 0.1) * (times > 0)
            # Add some oscillation
            osc = 2 * np.sin(2 * np.pi * 2 * times) * np.exp(-times/2) * (times > 0)

            if label == 'Left Hand' and ch == 'C4':
                mrn *= 1.5  # Contralateral effect
            elif label == 'Right Hand' and ch == 'C3':
                mrn *= 1.5

            mean = baseline + mrn + osc + np.random.randn() * 0.5
            sem = np.abs(np.random.randn(len(times))) * 0.5 + 0.3

            ax.plot(times, mean, color=color, label=label, linewidth=2)
            ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.2)

        ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title(f'Channel {ch}')
        ax.legend()

    fig.suptitle('Event-Related Potentials (Motor Cortex)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'erp_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: erp_comparison.png")

def generate_confusion_matrix():
    """Generate confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Sample confusion matrix
    cm = np.array([[38, 12], [10, 40]])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    labels = ['Left Hand', 'Right Hand']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = f'{cm_norm[i, j]:.1%}\n(n={cm[i, j]})'
            ax.text(j, i, text, ha='center', va='center', fontsize=11)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (CSP + LDA)')

    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: confusion_matrix.png")

def generate_model_comparison():
    """Generate model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = ['LDA', 'SVM-RBF', 'SVM-Linear', 'Random Forest', 'CSP+LDA']
    accuracies = [0.72, 0.74, 0.71, 0.69, 0.78]
    stds = [0.05, 0.04, 0.06, 0.05, 0.04]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(models, accuracies, yerr=stds, color=colors, edgecolor='black', capsize=5)

    ax.axhline(0.5, color='red', linestyle='--', label='Chance level')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison - Cross-Validation Accuracy')
    ax.set_ylim([0, 1])
    ax.legend()

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: model_comparison.png")

def generate_subject_variability():
    """Generate subject variability plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    subjects = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    accuracies = [0.78, 0.65, 0.82, 0.71, 0.68, 0.75, 0.62, 0.79, 0.73, 0.70]
    stds = [0.04, 0.06, 0.03, 0.05, 0.07, 0.04, 0.08, 0.04, 0.05, 0.06]

    colors = plt.cm.Set2(np.linspace(0, 1, len(subjects)))
    bars = ax.bar(subjects, accuracies, yerr=stds, color=colors, edgecolor='black', capsize=4)

    mean_acc = np.mean(accuracies)
    ax.axhline(0.5, color='red', linestyle='--', label='Chance level')
    ax.axhline(mean_acc, color='blue', linestyle='-.', label=f'Mean = {mean_acc:.2f}')

    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Motor Imagery Classification - Subject Variability')
    ax.set_ylim([0, 1])
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'subject_variability.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: subject_variability.png")

def generate_csp_patterns():
    """Generate CSP patterns visualization."""
    fig, axes = plt.subplots(1, 6, figsize=(14, 3))

    # Simulate topographic patterns (simplified circles)
    theta = np.linspace(0, 2*np.pi, 100)

    for idx, ax in enumerate(axes):
        # Create a simple circular head outline
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
        ax.plot([0], [1.05], 'k^', markersize=8)  # Nose

        # Simulate spatial pattern with gradient
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)

        # Create different patterns for each component
        if idx < 3:
            Z = np.exp(-((X - 0.3)**2 + Y**2) / 0.3) - np.exp(-((X + 0.3)**2 + Y**2) / 0.3)
        else:
            Z = np.exp(-((X + 0.3)**2 + Y**2) / 0.3) - np.exp(-((X - 0.3)**2 + Y**2) / 0.3)

        Z = Z * (X**2 + Y**2 < 0.95)  # Mask outside head

        im = ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r')
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.1, 1.2])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'CSP {idx + 1}')

    fig.suptitle('Common Spatial Patterns', fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'csp_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: csp_patterns.png")

def generate_time_frequency():
    """Generate time-frequency plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    times = np.linspace(-0.5, 4, 200)
    freqs = np.linspace(4, 35, 100)
    T, F = np.meshgrid(times, freqs)

    conditions = ['Left Hand', 'Right Hand']
    channels = ['C3', 'Cz', 'C4']

    for row, cond in enumerate(conditions):
        for col, ch in enumerate(channels):
            ax = axes[row, col]

            # Simulate ERD/ERS pattern
            # Alpha/beta desynchronization after stimulus
            erd = -20 * np.exp(-((F - 12)**2) / 50) * np.exp(-((T - 1.5)**2) / 1) * (T > 0)

            # Add contralateral effect
            if (cond == 'Left Hand' and ch == 'C4') or (cond == 'Right Hand' and ch == 'C3'):
                erd *= 1.5

            # Add some noise
            Z = erd + np.random.randn(*erd.shape) * 2

            im = ax.pcolormesh(T, F, Z, cmap='RdBu_r', vmin=-30, vmax=30, shading='auto')
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'{cond} - {ch}')

    fig.suptitle('Time-Frequency Representations (% change from baseline)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_frequency.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: time_frequency.png")

def generate_csp_features():
    """Generate CSP feature space plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulate CSP features
    n_left = 45
    n_right = 45

    # Left hand (class 1)
    left_x = np.random.randn(n_left) * 0.3 + 1.5
    left_y = np.random.randn(n_left) * 0.3 - 1.0

    # Right hand (class 2)
    right_x = np.random.randn(n_right) * 0.3 - 1.0
    right_y = np.random.randn(n_right) * 0.3 + 1.5

    ax.scatter(left_x, left_y, c='#2ecc71', alpha=0.7, s=80, label='Left Hand', edgecolors='black')
    ax.scatter(right_x, right_y, c='#e74c3c', alpha=0.7, s=80, label='Right Hand', edgecolors='black')

    ax.set_xlabel('CSP Component 1')
    ax.set_ylabel('CSP Component 6')
    ax.set_title('CSP Feature Space')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'csp_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: csp_features.png")

def main():
    print("Generating sample visualization images...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    generate_raw_signals()
    generate_psd_comparison()
    generate_erp_comparison()
    generate_confusion_matrix()
    generate_model_comparison()
    generate_subject_variability()
    generate_csp_patterns()
    generate_time_frequency()
    generate_csp_features()

    print("\nAll images generated successfully!")

if __name__ == "__main__":
    main()
