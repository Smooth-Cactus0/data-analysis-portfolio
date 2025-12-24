"""
Generate preprocessing visualization images for the EEG project.
Shows the transformation from raw to processed signals with explanations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_raw_vs_filtered():
    """
    Generate comparison of raw EEG vs filtered EEG signal.
    Shows how bandpass filtering removes noise while preserving brain signals.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Time vector (2 seconds at 160 Hz)
    sfreq = 160
    t = np.linspace(0, 2, 2 * sfreq)

    # Create realistic raw EEG signal
    # Components: slow drift + brain activity + high-freq noise + 50Hz line noise
    slow_drift = 30 * np.sin(2 * np.pi * 0.3 * t)  # < 1 Hz drift
    alpha_wave = 15 * np.sin(2 * np.pi * 10 * t)   # 10 Hz alpha
    beta_wave = 8 * np.sin(2 * np.pi * 22 * t)     # 22 Hz beta
    line_noise = 10 * np.sin(2 * np.pi * 50 * t)   # 50 Hz power line
    high_freq_noise = np.random.randn(len(t)) * 8  # High frequency noise
    muscle_artifact = np.random.randn(len(t)) * 15 * (np.random.rand(len(t)) > 0.95)  # Occasional spikes

    raw_signal = slow_drift + alpha_wave + beta_wave + line_noise + high_freq_noise + muscle_artifact

    # Simulated filtered signal (1-40 Hz bandpass removes drift and high-freq)
    filtered_signal = alpha_wave + beta_wave + np.random.randn(len(t)) * 3

    # Plot 1: Raw signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, raw_signal, 'b-', linewidth=0.7, alpha=0.8)
    ax1.set_title('Raw EEG Signal (Unprocessed)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude (μV)')
    ax1.set_ylim([-80, 80])
    ax1.set_xlim([0, 2])
    ax1.axhline(0, color='k', linewidth=0.5, alpha=0.3)

    # Plot 2: Filtered signal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, filtered_signal, 'g-', linewidth=0.7, alpha=0.8)
    ax2.set_title('After Bandpass Filter (1-40 Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amplitude (μV)')
    ax2.set_ylim([-80, 80])
    ax2.set_xlim([0, 2])
    ax2.axhline(0, color='k', linewidth=0.3, alpha=0.3)

    # Plot 3: Power spectrum - Raw
    ax3 = fig.add_subplot(gs[1, 0])
    freqs = np.fft.rfftfreq(len(t), 1/sfreq)
    raw_fft = np.abs(np.fft.rfft(raw_signal)) / len(t)
    ax3.semilogy(freqs, raw_fft, 'b-', linewidth=1)
    ax3.axvline(1, color='r', linestyle='--', alpha=0.7, label='Filter cutoffs (1-40 Hz)')
    ax3.axvline(40, color='r', linestyle='--', alpha=0.7)
    ax3.axvspan(0, 1, alpha=0.2, color='red', label='Removed: slow drift')
    ax3.axvspan(40, 80, alpha=0.2, color='red')
    ax3.axvline(50, color='orange', linestyle=':', linewidth=2, label='50 Hz line noise')
    ax3.set_title('Power Spectrum - Raw', fontsize=11)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.set_xlim([0, 80])
    ax3.legend(fontsize=8, loc='upper right')

    # Plot 4: Power spectrum - Filtered
    ax4 = fig.add_subplot(gs[1, 1])
    filtered_fft = np.abs(np.fft.rfft(filtered_signal)) / len(t)
    ax4.semilogy(freqs, filtered_fft, 'g-', linewidth=1)
    ax4.axvline(1, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(40, color='gray', linestyle='--', alpha=0.5)
    ax4.axvspan(8, 13, alpha=0.3, color='blue', label='Alpha band (8-13 Hz)')
    ax4.axvspan(13, 30, alpha=0.3, color='green', label='Beta band (13-30 Hz)')
    ax4.set_title('Power Spectrum - Filtered', fontsize=11)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power')
    ax4.set_xlim([0, 80])
    ax4.legend(fontsize=8, loc='upper right')

    # Plot 5: Text explanation
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    explanation = """
    PREPROCESSING RATIONALE:

    1. BANDPASS FILTER (1-40 Hz): Removes slow electrode drift (<1 Hz) and high-frequency noise (>40 Hz).
       Motor imagery signals of interest are in the alpha (8-13 Hz) and beta (13-30 Hz) frequency bands.

    2. NOTCH FILTER (50 Hz): Removes power line interference that contaminates recordings.

    3. RESULT: Clean signal where brain oscillations (alpha/beta waves) are preserved while artifacts are removed.
       This enables reliable detection of motor imagery patterns.
    """
    ax5.text(0.5, 0.5, explanation, transform=ax5.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(os.path.join(OUTPUT_DIR, 'preprocessing_filtering.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: preprocessing_filtering.png")


def generate_epoching_visualization():
    """
    Generate visualization showing how continuous EEG is segmented into epochs.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1], hspace=0.3)

    # Continuous signal with event markers
    sfreq = 160
    duration = 15  # seconds
    t = np.linspace(0, duration, duration * sfreq)

    # Create continuous EEG
    signal = (12 * np.sin(2 * np.pi * 10 * t) +
              6 * np.sin(2 * np.pi * 22 * t) +
              np.random.randn(len(t)) * 5)

    # Event times (motor imagery cues)
    events = [2.0, 6.5, 11.0]  # seconds
    event_labels = ['Left Hand\nImagery', 'Right Hand\nImagery', 'Left Hand\nImagery']
    colors = ['#2ecc71', '#e74c3c', '#2ecc71']

    # Top plot: Continuous signal with events
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, signal, 'b-', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('Amplitude (μV)', fontsize=11)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_title('Continuous EEG Recording with Motor Imagery Cues', fontsize=12, fontweight='bold')
    ax1.set_xlim([0, duration])
    ax1.set_ylim([-40, 40])

    # Mark events and epoch windows
    for i, (event_time, label, color) in enumerate(zip(events, event_labels, colors)):
        # Event marker
        ax1.axvline(event_time, color=color, linewidth=2, linestyle='-', alpha=0.8)
        ax1.text(event_time, 35, label, ha='center', fontsize=9, color=color, fontweight='bold')

        # Epoch window (-0.5 to 4.0 seconds)
        epoch_start = event_time - 0.5
        epoch_end = event_time + 4.0
        ax1.axvspan(epoch_start, epoch_end, alpha=0.15, color=color)

        # Bracket annotation
        ax1.annotate('', xy=(epoch_start, -35), xytext=(epoch_end, -35),
                     arrowprops=dict(arrowstyle='<->', color=color, lw=1.5))
        ax1.text((epoch_start + epoch_end) / 2, -38, f'Epoch {i+1}\n(4.5s)',
                 ha='center', fontsize=8, color=color)

    # Bottom: Extracted epochs
    ax2 = fig.add_subplot(gs[1])
    epoch_t = np.linspace(-0.5, 4.0, int(4.5 * sfreq))

    for i, (event_time, color) in enumerate(zip(events, colors)):
        start_idx = int((event_time - 0.5) * sfreq)
        end_idx = start_idx + len(epoch_t)
        if end_idx <= len(signal):
            epoch_data = signal[start_idx:end_idx]
            # Offset for visualization
            offset = i * 50
            ax2.plot(epoch_t, epoch_data + offset, color=color, linewidth=0.8, alpha=0.9)
            ax2.text(-0.7, offset, f'Epoch {i+1}', ha='right', fontsize=9, color=color)

    ax2.axvline(0, color='k', linestyle='--', linewidth=1.5, label='Cue onset (t=0)')
    ax2.set_xlabel('Time relative to cue (seconds)', fontsize=11)
    ax2.set_ylabel('Epochs (stacked)', fontsize=11)
    ax2.set_title('Extracted Epochs Aligned to Motor Imagery Cues', fontsize=12, fontweight='bold')
    ax2.set_xlim([-0.5, 4.0])
    ax2.legend(loc='upper right')
    ax2.set_yticks([])

    plt.savefig(os.path.join(OUTPUT_DIR, 'preprocessing_epoching.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: preprocessing_epoching.png")


def generate_pipeline_diagram():
    """
    Generate a visual pipeline diagram showing the complete analysis workflow.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    colors = {
        'data': '#3498db',
        'preprocess': '#9b59b6',
        'features': '#e67e22',
        'model': '#27ae60',
        'output': '#e74c3c'
    }

    def draw_box(x, y, w, h, text, color, fontsize=10):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', zorder=3,
                wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2),
                   zorder=1)

    # Title
    ax.text(8, 9.5, 'EEG Motor Imagery Classification Pipeline', fontsize=16,
            fontweight='bold', ha='center')

    # Row 1: Input Data
    draw_box(0.5, 7, 3, 1.5, 'Raw EEG\n(64 channels\n160 Hz)', colors['data'])
    ax.text(2, 6.5, 'INPUT', fontsize=9, ha='center', style='italic')

    # Arrow to preprocessing
    draw_arrow(3.5, 7.75, 4.5, 7.75)

    # Row 1: Preprocessing steps
    draw_box(4.5, 7, 2.5, 1.5, 'Bandpass\nFilter\n(1-40 Hz)', colors['preprocess'])
    draw_arrow(7, 7.75, 7.5, 7.75)
    draw_box(7.5, 7, 2.5, 1.5, 'Re-reference\n(Common\nAverage)', colors['preprocess'])
    draw_arrow(10, 7.75, 10.5, 7.75)
    draw_box(10.5, 7, 2.5, 1.5, 'Epoch\nExtraction\n(-0.5 to 4s)', colors['preprocess'])

    ax.text(8.5, 6.5, 'PREPROCESSING', fontsize=9, ha='center', style='italic')

    # Arrow down to features
    draw_arrow(11.75, 7, 11.75, 5.5)
    draw_arrow(11.75, 5.5, 8, 5.5)

    # Row 2: Feature Extraction (3 branches)
    draw_box(1, 4, 3.5, 1.2, 'Time Features\n(mean, std, variance)', colors['features'])
    draw_box(5, 4, 3.5, 1.2, 'Frequency Features\n(alpha, beta power)', colors['features'])
    draw_box(9, 4, 3.5, 1.2, 'Spatial Features\n(CSP patterns)', colors['features'])

    ax.text(6.75, 3.5, 'FEATURE EXTRACTION', fontsize=9, ha='center', style='italic')

    # Arrows from epoch to features
    draw_arrow(8, 5.5, 2.75, 5.2)
    draw_arrow(8, 5.5, 6.75, 5.2)
    draw_arrow(8, 5.5, 10.75, 5.2)

    # Merge arrows to classification
    draw_arrow(2.75, 4, 6.75, 2.7)
    draw_arrow(6.75, 4, 6.75, 2.7)
    draw_arrow(10.75, 4, 6.75, 2.7)

    # Row 3: Classification
    draw_box(4.5, 1.5, 4.5, 1.2, 'Classification Model\n(CSP + LDA / EEGNet)', colors['model'])
    ax.text(6.75, 1, 'CLASSIFICATION', fontsize=9, ha='center', style='italic')

    # Arrow to output
    draw_arrow(9, 2.1, 11, 2.1)

    # Output
    draw_box(11, 1.5, 3.5, 1.2, 'Prediction:\nLeft / Right\nHand', colors['output'])
    ax.text(12.75, 1, 'OUTPUT', fontsize=9, ha='center', style='italic')

    # Legend / Explanation box
    legend_text = """
    WHAT WE CLASSIFY:
    • Input: 4.5-second EEG epochs recorded during motor imagery
    • Target: Which hand the subject imagined moving (Left or Right)
    • Application: Brain-Computer Interface (BCI) control
    """
    ax.text(0.5, 0.3, legend_text, fontsize=9, fontfamily='monospace',
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(os.path.join(OUTPUT_DIR, 'pipeline_diagram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: pipeline_diagram.png")


def generate_feature_explanation():
    """
    Generate visualization explaining feature extraction approaches.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    sfreq = 160
    t = np.linspace(0, 2, 2 * sfreq)

    # Sample epoch signal
    signal = 15 * np.sin(2 * np.pi * 10 * t) + 8 * np.sin(2 * np.pi * 22 * t) + np.random.randn(len(t)) * 4

    # 1. Time-domain features
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, signal, 'b-', linewidth=0.8)
    ax1.axhline(np.mean(signal), color='r', linestyle='--', linewidth=2, label=f'Mean = {np.mean(signal):.1f}')
    ax1.fill_between(t, np.mean(signal) - np.std(signal), np.mean(signal) + np.std(signal),
                     alpha=0.3, color='orange', label=f'Std = {np.std(signal):.1f}')
    ax1.set_title('Time-Domain Features', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (μV)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.text(0.05, 0.05, 'Extract: mean, std,\nvariance, skewness,\nkurtosis per channel',
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 2. Frequency-domain features
    ax2 = fig.add_subplot(gs[0, 1])
    freqs = np.fft.rfftfreq(len(t), 1/sfreq)
    psd = np.abs(np.fft.rfft(signal))**2 / len(t)
    ax2.semilogy(freqs, psd, 'b-', linewidth=1)

    # Highlight bands
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    ax2.fill_between(freqs, psd, where=alpha_mask, alpha=0.5, color='blue', label='Alpha (8-13 Hz)')
    ax2.fill_between(freqs, psd, where=beta_mask, alpha=0.5, color='green', label='Beta (13-30 Hz)')
    ax2.set_title('Frequency-Domain Features', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power')
    ax2.set_xlim([0, 45])
    ax2.legend(loc='upper right', fontsize=8)
    ax2.text(0.05, 0.05, 'Extract: power in each\nfrequency band\n(α, β most important)',
             transform=ax2.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 3. CSP spatial features
    ax3 = fig.add_subplot(gs[0, 2])
    # Simulate CSP-transformed signal variance
    left_var = np.array([0.8, 0.6, 0.4, 0.3, 0.35, 0.5])
    right_var = np.array([0.3, 0.35, 0.45, 0.55, 0.7, 0.85])
    x = np.arange(6) + 1
    width = 0.35
    ax3.bar(x - width/2, left_var, width, label='Left Hand', color='#2ecc71')
    ax3.bar(x + width/2, right_var, width, label='Right Hand', color='#e74c3c')
    ax3.set_title('CSP Features', fontsize=12, fontweight='bold')
    ax3.set_xlabel('CSP Component')
    ax3.set_ylabel('Log-Variance')
    ax3.legend(loc='upper center', fontsize=8)
    ax3.text(0.05, 0.05, 'CSP maximizes\nvariance difference\nbetween classes',
             transform=ax3.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 4. Why these features work (bottom spanning plot)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    explanation = """
    WHY THESE FEATURES ENABLE MOTOR IMAGERY DECODING:

    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  NEUROSCIENCE BASIS:                                                                                            │
    │  When you imagine moving your LEFT hand, the RIGHT motor cortex (electrode C4) shows decreased alpha/beta      │
    │  power - a phenomenon called Event-Related Desynchronization (ERD). The opposite occurs for right hand imagery.│
    │                                                                                                                 │
    │  FEATURE ENGINEERING RATIONALE:                                                                                 │
    │  • Time features capture overall signal amplitude patterns                                                      │
    │  • Frequency features (especially 8-30 Hz) capture the ERD phenomenon                                          │
    │  • CSP learns optimal spatial filters that maximize the left/right difference across electrodes                 │
    │                                                                                                                 │
    │  WHAT THE CLASSIFIER LEARNS:                                                                                    │
    │  The model learns that: Left hand imagery → ERD at C4 (right hemisphere)                                       │
    │                         Right hand imagery → ERD at C3 (left hemisphere)                                       │
    │  This contralateral pattern is the key discriminative feature.                                                  │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    ax4.text(0.5, 0.5, explanation, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_extraction_explained.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: feature_extraction_explained.png")


def main():
    print("Generating preprocessing and pipeline images...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    generate_raw_vs_filtered()
    generate_epoching_visualization()
    generate_pipeline_diagram()
    generate_feature_explanation()

    print("\nAll preprocessing images generated successfully!")


if __name__ == "__main__":
    main()
