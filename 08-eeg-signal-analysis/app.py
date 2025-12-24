"""
EEG Signal Analysis - Streamlit Demo Application

Interactive visualization and classification of EEG brain signals
for Brain-Computer Interface (BCI) motor imagery tasks.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import json

# MNE and ML imports
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Suppress MNE info messages
mne.set_log_level('WARNING')

# Page configuration
st.set_page_config(
    page_title="EEG Brain Signal Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Cache data loading
@st.cache_data(show_spinner=False)
def load_sample_data(subject=1, runs=[4, 8]):
    """Load sample EEG data from MNE dataset."""
    try:
        raw_files = eegbci.load_data(subject, runs, update_path=False)
        raws = [read_raw_edf(f, preload=True) for f in raw_files]
        raw = concatenate_raws(raws)
        eegbci.standardize(raw)
        return raw
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data(show_spinner=False)
def preprocess_raw(_raw, l_freq=1.0, h_freq=40.0):
    """Apply basic preprocessing."""
    raw = _raw.copy()
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")
    raw.set_eeg_reference("average", projection=False)
    return raw


@st.cache_data(show_spinner=False)
def create_epochs(_raw, tmin=-0.5, tmax=4.0):
    """Create epochs from raw data."""
    events, _ = mne.events_from_annotations(_raw)
    event_id = {"T1": 2, "T2": 3}  # Left and right hand
    epochs = mne.Epochs(
        _raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=(-0.5, 0),
        preload=True, picks="eeg"
    )
    return epochs


def plot_raw_signals_plotly(raw, duration=5, start=0, n_channels=8):
    """Plot raw EEG signals using Plotly."""
    sfreq = raw.info['sfreq']
    start_idx = int(start * sfreq)
    end_idx = int((start + duration) * sfreq)
    times = np.arange(start_idx, end_idx) / sfreq

    data = raw.get_data()[:n_channels, start_idx:end_idx]
    ch_names = raw.ch_names[:n_channels]

    fig = make_subplots(rows=n_channels, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02)

    colors = px.colors.qualitative.Set2

    for i, (ch_data, ch_name) in enumerate(zip(data, ch_names)):
        fig.add_trace(
            go.Scatter(x=times, y=ch_data * 1e6, name=ch_name,
                       line=dict(color=colors[i % len(colors)], width=1)),
            row=i + 1, col=1
        )
        fig.update_yaxes(title_text=ch_name, row=i + 1, col=1,
                         range=[-100, 100], title_font_size=10)

    fig.update_layout(
        height=100 * n_channels,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
        title="Raw EEG Signals"
    )
    fig.update_xaxes(title_text="Time (s)", row=n_channels, col=1)

    return fig


def plot_psd_plotly(epochs, event_id):
    """Plot PSD comparison using Plotly."""
    colors = {'T1': '#2ecc71', 'T2': '#e74c3c'}

    fig = go.Figure()

    for event_name in event_id.keys():
        epochs_cond = epochs[event_name]
        spectrum = epochs_cond.compute_psd(method='welch', fmin=1, fmax=40)
        psds, freqs = spectrum.get_data(return_freqs=True)
        psds_mean = psds.mean(axis=(0, 1)) * 1e12

        label = "Left Hand" if event_name == "T1" else "Right Hand"
        fig.add_trace(go.Scatter(
            x=freqs, y=psds_mean,
            name=label,
            line=dict(color=colors[event_name], width=2)
        ))

    # Add frequency band annotations
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
             'Beta': (13, 30), 'Gamma': (30, 40)}
    colors_band = ['rgba(255,0,0,0.1)', 'rgba(255,165,0,0.1)',
                   'rgba(255,255,0,0.1)', 'rgba(0,255,0,0.1)',
                   'rgba(0,0,255,0.1)']

    for (band_name, (fmin, fmax)), color in zip(bands.items(), colors_band):
        fig.add_vrect(x0=fmin, x1=fmax, fillcolor=color, layer="below",
                      line_width=0, annotation_text=band_name,
                      annotation_position="top")

    fig.update_layout(
        title="Power Spectral Density Comparison",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (μV²/Hz)",
        yaxis_type="log",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    return fig


def plot_erp_plotly(epochs, event_id, channel='C3'):
    """Plot ERP comparison using Plotly."""
    colors = {'T1': '#2ecc71', 'T2': '#e74c3c'}
    times = epochs.times

    ch_idx = epochs.ch_names.index(channel) if channel in epochs.ch_names else 0

    fig = go.Figure()

    for event_name in event_id.keys():
        epochs_cond = epochs[event_name]
        data = epochs_cond.get_data()[:, ch_idx, :]

        mean = data.mean(axis=0) * 1e6
        std = data.std(axis=0) * 1e6

        label = "Left Hand" if event_name == "T1" else "Right Hand"

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill='toself',
            fillcolor=colors[event_name].replace(')', ',0.2)').replace('#', 'rgba(').replace('2ecc71', '46,204,113').replace('e74c3c', '231,76,60'),
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{label} CI'
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=times, y=mean,
            name=label,
            line=dict(color=colors[event_name], width=2)
        ))

    # Add vertical line at stimulus onset
    fig.add_vline(x=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title=f"Event-Related Potentials - Channel {channel}",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (μV)",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    return fig


def plot_topomap_at_time(epochs, t, event_name):
    """Create topomap at specific time point."""
    epochs_cond = epochs[event_name]
    t_idx = np.argmin(np.abs(epochs.times - t))
    data = epochs_cond.get_data()[:, :, t_idx].mean(axis=0)

    fig, ax = plt.subplots(figsize=(4, 4))
    mne.viz.plot_topomap(data, epochs.info, axes=ax, show=False,
                         cmap='RdBu_r', contours=6)
    label = "Left Hand" if event_name == "T1" else "Right Hand"
    ax.set_title(f'{label} at t={t:.2f}s')
    plt.tight_layout()
    return fig


def extract_simple_features(epochs):
    """Extract simple features for classification."""
    X = []
    y = []

    sfreq = epochs.info['sfreq']

    for epoch_data, event in zip(epochs.get_data(), epochs.events[:, -1]):
        features = []

        # Time features per channel (mean, std, var)
        features.extend(np.mean(epoch_data, axis=1))
        features.extend(np.std(epoch_data, axis=1))

        # Band power (rough estimate)
        from scipy.signal import welch
        for ch in range(epoch_data.shape[0]):
            freqs, psd = welch(epoch_data[ch], fs=sfreq, nperseg=128)
            # Alpha band power (8-13 Hz)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_power = np.sum(psd[alpha_mask])
            # Beta band power (13-30 Hz)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            beta_power = np.sum(psd[beta_mask])
            features.extend([alpha_power, beta_power])

        X.append(features)
        y.append(event)

    return np.array(X), np.array(y)


def main():
    # Header
    st.markdown('<p class="main-header">🧠 EEG Brain Signal Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Visualization & Classification of Motor Imagery BCI Signals</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Settings")

    # Data Loading Section
    st.sidebar.header("1. Data Selection")
    subject_id = st.sidebar.slider("Subject ID", 1, 10, 1)
    st.sidebar.caption("PhysioNet EEG Motor Imagery Dataset")

    # Load data
    with st.spinner("Loading EEG data..."):
        raw = load_sample_data(subject=subject_id)

    if raw is None:
        st.error("Failed to load data. Please check your internet connection.")
        return

    # Preprocessing settings
    st.sidebar.header("2. Preprocessing")
    l_freq = st.sidebar.slider("Low-pass filter (Hz)", 0.5, 5.0, 1.0, 0.5)
    h_freq = st.sidebar.slider("High-pass filter (Hz)", 20, 50, 40, 5)

    # Preprocess
    with st.spinner("Preprocessing..."):
        raw_filtered = preprocess_raw(raw, l_freq, h_freq)
        epochs = create_epochs(raw_filtered)

    # Display dataset info
    st.sidebar.header("3. Dataset Info")
    st.sidebar.metric("Channels", len(raw.ch_names))
    st.sidebar.metric("Sampling Rate", f"{raw.info['sfreq']:.0f} Hz")
    st.sidebar.metric("Total Epochs", len(epochs))
    st.sidebar.metric("Left Hand Epochs", len(epochs['T1']))
    st.sidebar.metric("Right Hand Epochs", len(epochs['T2']))

    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Raw Signals",
        "⚡ Frequency Analysis",
        "🎯 ERPs & Topomaps",
        "🤖 Classification",
        "ℹ️ About"
    ])

    # Tab 1: Raw Signals
    with tab1:
        st.header("Raw EEG Signal Visualization")

        col1, col2 = st.columns(2)
        with col1:
            start_time = st.slider("Start time (s)", 0.0, float(raw.times[-1]) - 10, 0.0, 1.0)
        with col2:
            duration = st.slider("Duration (s)", 2.0, 20.0, 5.0, 1.0)

        n_channels = st.slider("Number of channels", 4, 16, 8)

        fig_raw = plot_raw_signals_plotly(raw_filtered, duration, start_time, n_channels)
        st.plotly_chart(fig_raw, use_container_width=True)

        st.info("""
        **What you're seeing:** Raw EEG signals from electrodes placed on the scalp.
        Each trace represents voltage fluctuations (in microvolts) from a different brain region.
        These signals contain information about neural activity during motor imagery tasks.
        """)

    # Tab 2: Frequency Analysis
    with tab2:
        st.header("Power Spectral Density Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_psd = plot_psd_plotly(epochs, {"T1": 2, "T2": 3})
            st.plotly_chart(fig_psd, use_container_width=True)

        with col2:
            st.subheader("Frequency Bands")
            st.markdown("""
            | Band | Range | Associated with |
            |------|-------|-----------------|
            | **Delta** | 1-4 Hz | Deep sleep |
            | **Theta** | 4-8 Hz | Drowsiness, meditation |
            | **Alpha** | 8-13 Hz | Relaxed, idle |
            | **Beta** | 13-30 Hz | Active thinking, focus |
            | **Gamma** | 30-40 Hz | Cognitive processing |
            """)

        st.info("""
        **Key Insight:** During motor imagery, we expect to see differences in the **alpha** (8-13 Hz)
        and **beta** (13-30 Hz) bands, especially over motor cortex regions. This phenomenon is known
        as **Event-Related Desynchronization (ERD)** and is the basis for motor imagery BCIs.
        """)

    # Tab 3: ERPs & Topomaps
    with tab3:
        st.header("Event-Related Potentials & Topographic Maps")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ERPs")
            channel = st.selectbox("Select Channel", ['C3', 'Cz', 'C4', 'FC3', 'FC4'])
            if channel in epochs.ch_names:
                fig_erp = plot_erp_plotly(epochs, {"T1": 2, "T2": 3}, channel)
                st.plotly_chart(fig_erp, use_container_width=True)
            else:
                st.warning(f"Channel {channel} not available")

        with col2:
            st.subheader("Topographic Maps")
            time_point = st.slider("Time point (s)", float(epochs.tmin), float(epochs.tmax), 1.0, 0.1)
            event_select = st.radio("Condition", ["Left Hand (T1)", "Right Hand (T2)"])
            event_name = "T1" if "T1" in event_select else "T2"

            fig_topo = plot_topomap_at_time(epochs, time_point, event_name)
            st.pyplot(fig_topo)

        st.info("""
        **ERPs** show the average brain response time-locked to events. **Topographic maps**
        show the spatial distribution of activity across the scalp. For motor imagery,
        look for differences over the sensorimotor cortex (C3 for right hand, C4 for left hand
        - due to contralateral organization of the motor cortex).
        """)

    # Tab 4: Classification
    with tab4:
        st.header("Motor Imagery Classification")

        st.markdown("""
        Train machine learning models to classify **Left Hand** vs **Right Hand** motor imagery
        based on EEG features.
        """)

        if st.button("Train Classifiers", type="primary"):
            with st.spinner("Extracting features and training models..."):
                # Extract features
                X, y = extract_simple_features(epochs)

                # Handle NaN/Inf
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train models
                models = {
                    "LDA": LinearDiscriminantAnalysis(),
                    "SVM": SVC(kernel='rbf', C=1.0),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
                }

                results = {}
                for name, model in models.items():
                    scores = cross_val_score(model, X_scaled, y, cv=5)
                    results[name] = {
                        "mean": scores.mean(),
                        "std": scores.std(),
                        "scores": scores
                    }

                # Display results
                st.subheader("Cross-Validation Results (5-fold)")

                cols = st.columns(len(models))
                for col, (name, res) in zip(cols, results.items()):
                    with col:
                        st.metric(
                            name,
                            f"{res['mean']*100:.1f}%",
                            f"± {res['std']*100:.1f}%"
                        )

                # Bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(results.keys()),
                    y=[r['mean'] * 100 for r in results.values()],
                    error_y=dict(
                        type='data',
                        array=[r['std'] * 100 for r in results.values()]
                    ),
                    marker_color=['#3498db', '#2ecc71', '#e74c3c']
                ))
                fig.add_hline(y=50, line_dash="dash", line_color="gray",
                              annotation_text="Chance level")
                fig.update_layout(
                    title="Classification Accuracy Comparison",
                    yaxis_title="Accuracy (%)",
                    yaxis_range=[0, 100],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                st.success("""
                Models trained successfully! Accuracy above 50% indicates the classifier
                can distinguish between left and right hand motor imagery better than chance.
                """)

                # Feature info
                st.subheader("Features Used")
                st.markdown(f"""
                - **Time-domain features:** Mean and standard deviation per channel ({len(epochs.ch_names) * 2} features)
                - **Frequency-domain features:** Alpha and beta band power per channel ({len(epochs.ch_names) * 2} features)
                - **Total features:** {X.shape[1]}
                """)

    # Tab 5: About
    with tab5:
        st.header("About This Project")

        st.markdown("""
        ### EEG Motor Imagery Brain-Computer Interface

        This project demonstrates the analysis and classification of EEG signals
        for Brain-Computer Interface (BCI) applications, specifically focusing on
        motor imagery tasks.

        #### Dataset
        - **Source:** PhysioNet EEG Motor Movement/Imagery Dataset
        - **Subjects:** 109 volunteers
        - **Channels:** 64 EEG electrodes (10-20 system)
        - **Tasks:** Motor imagery (imagining left/right hand movement)
        - **Sampling Rate:** 160 Hz

        #### Methods

        **Preprocessing Pipeline:**
        1. Bandpass filtering (1-40 Hz)
        2. Average reference
        3. Epoch extraction around events

        **Feature Extraction:**
        - Time-domain statistics (mean, variance)
        - Frequency-domain power (alpha, beta bands)
        - Common Spatial Patterns (CSP)

        **Classification Models:**
        - Linear Discriminant Analysis (LDA)
        - Support Vector Machine (SVM)
        - Random Forest
        - Deep Learning (EEGNet)

        #### References

        1. Schalk, G. et al. (2004). BCI2000: A General-Purpose BCI System
        2. Lawhern, V. J. et al. (2018). EEGNet: A Compact CNN for EEG-based BCIs
        3. Blankertz, B. et al. (2008). CSP for Motor Imagery EEG Classification

        ---

        **Author:** Alexy Louis

        *Part of the Data Analysis Portfolio*
        """)


if __name__ == "__main__":
    main()
