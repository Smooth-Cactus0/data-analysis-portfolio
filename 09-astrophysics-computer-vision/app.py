"""
Astrophysics Computer Vision - Interactive Demo

A Streamlit application for exploring galaxy morphology classification
and transient light curve classification using deep learning.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Astrophysics CV Demo",
    page_icon="🔭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c5282;
        border-bottom: 2px solid #4299e1;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #ebf8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4299e1;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Passband colors
PASSBAND_COLORS = {
    0: '#8b5cf6',  # u
    1: '#3b82f6',  # g
    2: '#22c55e',  # r
    3: '#f97316',  # i
    4: '#ef4444',  # z
    5: '#7f1d1d',  # y
}

PASSBAND_NAMES = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'}


def generate_sample_galaxy():
    """Generate a synthetic galaxy image for demo."""
    np.random.seed(None)
    size = 128

    galaxy_type = np.random.choice(['elliptical', 'spiral', 'edge_on'])

    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    if galaxy_type == 'elliptical':
        a = np.random.uniform(20, 40)
        b = np.random.uniform(10, a)
        angle = np.random.uniform(0, np.pi)
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)
        r_ell = np.sqrt((x_rot/a)**2 + (y_rot/b)**2)
        intensity = 255 * np.exp(-7.67 * (r_ell**(1/4) - 1))

        img = np.zeros((size, size, 3))
        img[:,:,0] = intensity
        img[:,:,1] = intensity * 0.8
        img[:,:,2] = intensity * 0.6

    elif galaxy_type == 'spiral':
        disk = 200 * np.exp(-r/30)
        n_arms = np.random.choice([2, 4])
        arms = 80 * np.sin(n_arms * theta - 0.3 * r) * np.exp(-r/40)
        arms = np.clip(arms, 0, 80)
        bulge = 255 * np.exp(-r**2/100)
        intensity = np.clip(disk + arms + bulge, 0, 255)

        img = np.zeros((size, size, 3))
        img[:,:,0] = intensity * (0.7 + 0.3*np.exp(-r/20))
        img[:,:,1] = intensity
        img[:,:,2] = intensity * (0.5 + 0.5*(1-np.exp(-r/30)))

    else:  # edge_on
        a = np.random.uniform(40, 55)
        b = np.random.uniform(3, 8)
        angle = np.random.uniform(-0.3, 0.3)
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)
        r_ell = np.sqrt((x_rot/a)**2 + (y_rot/b)**2)
        disk = 200 * np.exp(-r_ell)
        dust = 1 - 0.7 * np.exp(-y_rot**2/4) * (np.abs(x_rot) < 35)
        intensity = disk * dust

        img = np.zeros((size, size, 3))
        img[:,:,0] = intensity
        img[:,:,1] = intensity * 0.9
        img[:,:,2] = intensity * 0.7

    # Add noise
    noise = np.random.randn(size, size, 3) * 10
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img, galaxy_type


def generate_sample_lightcurve(transient_type):
    """Generate a synthetic light curve for demo."""
    np.random.seed(None)

    n_obs = np.random.randint(80, 150)
    mjd = np.sort(np.random.uniform(59000, 60000, n_obs))
    passbands = np.random.choice([0, 1, 2, 3, 4, 5], n_obs)

    if transient_type == 'SNIa':
        peak_mjd = np.random.uniform(59300, 59700)
        t = mjd - peak_mjd
        peak_flux = np.random.uniform(500, 2000)
        flux = peak_flux * np.exp(-0.5 * (t/15)**2) * (t < 100)
        flux += peak_flux * 0.3 * np.exp(-t/40) * (t > 0) * (t < 100)

    elif transient_type == 'SNII':
        peak_mjd = np.random.uniform(59300, 59700)
        t = mjd - peak_mjd
        peak_flux = np.random.uniform(300, 1500)
        flux = peak_flux * np.exp(-0.5 * (t/20)**2) * (t < 50)
        flux += peak_flux * 0.5 * (t >= 50) * (t < 100)
        flux *= np.exp(-t/100) * (t > 0)

    elif transient_type == 'Kilonova':
        peak_mjd = np.random.uniform(59300, 59700)
        t = mjd - peak_mjd
        peak_flux = np.random.uniform(200, 800)
        flux = peak_flux * np.exp(-t/2) * (t > 0) * (t < 20)

    elif transient_type == 'RRLyrae':
        period = np.random.uniform(0.4, 0.9)
        amplitude = np.random.uniform(100, 300)
        flux = 500 + amplitude * np.sin(2 * np.pi * mjd / period)

    else:  # AGN
        base_flux = np.random.uniform(200, 800)
        flux = base_flux + 100 * np.cumsum(np.random.randn(n_obs)) / np.sqrt(n_obs)

    flux = np.maximum(flux, 0)
    flux_err = np.sqrt(flux + 100) * np.random.uniform(0.8, 1.2, n_obs)
    flux += np.random.randn(n_obs) * flux_err * 0.5

    color_offset = (passbands - 2) * 50 * np.random.uniform(0.5, 1.5)
    flux += color_offset
    flux = np.maximum(flux, 1)

    return pd.DataFrame({
        'mjd': mjd,
        'passband': passbands,
        'flux': flux,
        'flux_err': flux_err
    })


def main():
    st.markdown('<h1 class="main-header">🔭 Astrophysics Computer Vision</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    This interactive demo showcases machine learning applications in astronomy:
    <ul>
        <li><b>Galaxy Morphology Classification</b>: Identifying galaxy types from images</li>
        <li><b>Transient Classification</b>: Classifying variable objects from light curves</li>
        <li><b>Anomaly Detection</b>: Finding unusual astronomical objects</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Demo",
        ["🌌 Galaxy Classification", "⭐ Transient Classification", "🔍 Anomaly Detection", "📚 Learn More"]
    )

    if page == "🌌 Galaxy Classification":
        galaxy_classification_page()
    elif page == "⭐ Transient Classification":
        transient_classification_page()
    elif page == "🔍 Anomaly Detection":
        anomaly_detection_page()
    else:
        learn_more_page()


def galaxy_classification_page():
    st.markdown('<h2 class="section-header">Galaxy Morphology Classification</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Galaxy Types")

        st.markdown("""
        **Elliptical Galaxies** 🔴
        - Smooth, featureless appearance
        - Old stellar populations (red color)
        - Little gas or dust
        - Formed through galaxy mergers

        **Spiral Galaxies** 🔵
        - Central bulge with spiral arms
        - Active star formation (blue arms)
        - Disk structure with rotation
        - Our Milky Way is a spiral galaxy

        **Edge-on Galaxies** 🟢
        - Disk galaxies viewed from the side
        - Often show dark dust lane
        - Structure hidden by viewing angle
        """)

    with col2:
        st.subheader("Generate Sample Galaxy")

        if st.button("🎲 Generate Random Galaxy", key="gen_galaxy"):
            img, gtype = generate_sample_galaxy()
            st.session_state['galaxy_img'] = img
            st.session_state['galaxy_type'] = gtype

        if 'galaxy_img' in st.session_state:
            st.image(
                st.session_state['galaxy_img'],
                caption=f"Generated: {st.session_state['galaxy_type'].title()} Galaxy",
                use_container_width=True
            )

            # Simulated classification
            probs = {
                'Elliptical': np.random.uniform(0.1, 0.3),
                'Spiral': np.random.uniform(0.1, 0.3),
                'Edge-on': np.random.uniform(0.1, 0.3)
            }
            probs[st.session_state['galaxy_type'].replace('_', '-').title()] = np.random.uniform(0.7, 0.95)

            # Normalize
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}

            fig = px.bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                labels={'x': 'Galaxy Type', 'y': 'Probability'},
                title='Classification Probabilities'
            )
            fig.update_traces(marker_color=['#e74c3c', '#3498db', '#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)

    # Model explanation
    st.markdown('<h3 class="section-header">How It Works</h3>', unsafe_allow_html=True)

    st.markdown("""
    Our galaxy classifier uses a **Convolutional Neural Network (CNN)** with transfer learning:

    1. **Input**: 128×128 RGB galaxy image
    2. **Backbone**: ResNet-18 pretrained on ImageNet
    3. **Fine-tuning**: Top layers trained on Galaxy Zoo data
    4. **Output**: Probability distribution over morphology classes

    **Why CNNs for Galaxy Classification?**
    - Galaxies have characteristic spatial patterns (spiral arms, bulges)
    - CNNs naturally learn hierarchical features
    - Transfer learning provides robust low-level features (edges, textures)
    """)


def transient_classification_page():
    st.markdown('<h2 class="section-header">Transient & Variable Star Classification</h2>', unsafe_allow_html=True)

    # Transient type selector
    transient_type = st.selectbox(
        "Select Transient Type to Simulate",
        ["SNIa", "SNII", "Kilonova", "RRLyrae", "AGN"]
    )

    transient_descriptions = {
        "SNIa": "Type Ia Supernova - Thermonuclear explosion of white dwarf. Used as cosmological standard candles.",
        "SNII": "Type II Supernova - Core-collapse of massive star with hydrogen envelope. Shows plateau phase.",
        "Kilonova": "Neutron star merger - Rapid transient (days). Source of heavy elements like gold.",
        "RRLyrae": "RR Lyrae Variable - Pulsating star with period < 1 day. Distance indicator.",
        "AGN": "Active Galactic Nucleus - Supermassive black hole accretion. Stochastic variability."
    }

    st.info(transient_descriptions[transient_type])

    if st.button("🎲 Generate Light Curve", key="gen_lc"):
        lc = generate_sample_lightcurve(transient_type)
        st.session_state['lightcurve'] = lc
        st.session_state['lc_type'] = transient_type

    if 'lightcurve' in st.session_state:
        lc = st.session_state['lightcurve']

        # Create plotly figure
        fig = go.Figure()

        for pb in sorted(lc['passband'].unique()):
            pb_data = lc[lc['passband'] == pb].sort_values('mjd')
            color = PASSBAND_COLORS.get(pb, '#333333')
            name = PASSBAND_NAMES.get(pb, str(pb))

            fig.add_trace(go.Scatter(
                x=pb_data['mjd'],
                y=pb_data['flux'],
                mode='markers',
                name=f'{name}-band',
                marker=dict(color=color, size=6),
                error_y=dict(
                    type='data',
                    array=pb_data['flux_err'],
                    visible=True,
                    color=color
                )
            ))

        fig.update_layout(
            title=f'Simulated {st.session_state["lc_type"]} Light Curve',
            xaxis_title='MJD (Modified Julian Date)',
            yaxis_title='Flux',
            legend_title='Passband',
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Classification results
        st.subheader("Classification Result")

        classes = ["SNIa", "SNII", "Kilonova", "RRLyrae", "AGN"]
        probs = np.random.uniform(0.05, 0.2, len(classes))
        probs[classes.index(st.session_state['lc_type'])] = np.random.uniform(0.6, 0.9)
        probs = probs / probs.sum()

        col1, col2 = st.columns(2)

        with col1:
            fig_bar = px.bar(
                x=classes,
                y=probs,
                labels={'x': 'Class', 'y': 'Probability'},
                title='Classification Probabilities'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            pred_class = classes[np.argmax(probs)]
            confidence = np.max(probs)

            st.metric("Predicted Class", pred_class)
            st.metric("Confidence", f"{confidence:.1%}")

    # Model explanation
    st.markdown('<h3 class="section-header">How It Works</h3>', unsafe_allow_html=True)

    st.markdown("""
    Our transient classifier uses a **Hybrid CNN-LSTM** architecture:

    1. **Input**: Multi-band light curve (6 passbands × time)
    2. **1D-CNN**: Extracts local temporal features (rise, peak, decline shapes)
    3. **Bidirectional LSTM**: Captures long-range dependencies
    4. **Output**: Probability distribution over 14 transient classes

    **Why 1D-CNN + LSTM, not 2D-CNN?**
    - Light curves are 1D time series, not 2D images
    - 1D convolutions respect the temporal nature of data
    - LSTM captures dependencies between early and late phases
    """)


def anomaly_detection_page():
    st.markdown('<h2 class="section-header">Anomaly Detection</h2>', unsafe_allow_html=True)

    st.markdown("""
    Anomaly detection helps discover:
    - **Rare objects**: Gravitational lenses, peculiar supernovae
    - **Novel physics**: Previously unknown transient types
    - **Data issues**: Instrument artifacts, cosmic rays

    Our approach uses **autoencoders** - neural networks that learn to compress and reconstruct data.
    Objects that can't be well-reconstructed are flagged as anomalies.
    """)

    # Simulated anomaly scores
    np.random.seed(42)
    n_normal = 500
    n_anomaly = 50

    normal_scores = np.random.exponential(0.01, n_normal)
    anomaly_scores = np.random.exponential(0.05, n_anomaly) + 0.03

    all_scores = np.concatenate([normal_scores, anomaly_scores])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=normal_scores,
        name='Normal Objects',
        opacity=0.7,
        marker_color='#3498db'
    ))

    fig.add_trace(go.Histogram(
        x=anomaly_scores,
        name='Anomalies',
        opacity=0.7,
        marker_color='#e74c3c'
    ))

    # Threshold line
    threshold = np.percentile(all_scores, 95)
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Threshold: {threshold:.3f}")

    fig.update_layout(
        title='Distribution of Reconstruction Errors',
        xaxis_title='Reconstruction Error (Anomaly Score)',
        yaxis_title='Count',
        barmode='overlay'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        detected = np.sum(anomaly_scores > threshold)
        recall = detected / n_anomaly
        st.metric("Anomalies Detected", f"{detected}/{n_anomaly}", f"{recall:.0%} recall")

    with col2:
        false_positives = np.sum(normal_scores > threshold)
        precision = detected / (detected + false_positives) if (detected + false_positives) > 0 else 0
        st.metric("False Positives", false_positives, f"{precision:.0%} precision")

    with col3:
        st.metric("Threshold", f"{threshold:.4f}", "95th percentile")


def learn_more_page():
    st.markdown('<h2 class="section-header">Learn More</h2>', unsafe_allow_html=True)

    st.markdown("""
    ## The Science Behind the Models

    ### Galaxy Morphology
    Galaxy morphology classification has a rich history, from Edwin Hubble's original "tuning fork"
    diagram (1926) to modern machine learning approaches. The physical appearance of a galaxy
    encodes information about:

    - **Formation history**: Mergers produce ellipticals, disk instabilities create spiral arms
    - **Stellar populations**: Blue regions have young stars, red regions have old stars
    - **Environment**: Cluster galaxies tend to be elliptical (ram pressure stripping removes gas)

    ### Transient Astronomy
    The study of variable and transient objects is undergoing a revolution with surveys like:

    - **ZTF (Zwicky Transient Facility)**: Currently scanning the sky every 2 nights
    - **Rubin Observatory (LSST)**: Will detect ~10 million alerts per night starting 2025

    Machine learning is essential for:
    1. **Real-time classification**: Deciding which alerts need immediate follow-up
    2. **Photometric classification**: When spectra aren't available
    3. **Anomaly detection**: Finding the unexpected

    ### Why These Model Architectures?

    | Task | Best Architecture | Why |
    |------|------------------|-----|
    | Galaxy images | 2D CNN / Transfer Learning | Spatial patterns, hierarchical features |
    | Light curves | 1D CNN + LSTM | Temporal patterns, long-range dependencies |
    | Anomaly detection | Autoencoder | Unsupervised, learns normal distribution |

    ## Datasets Used

    1. **Galaxy Zoo 2**: 61,578 galaxies with citizen science morphology labels
    2. **PLAsTiCC**: 3.5 million simulated LSST light curves across 14 classes

    ## References

    1. Willett et al. (2013) - Galaxy Zoo 2
    2. The PLAsTiCC Team (2018) - PLAsTiCC Challenge
    3. Dieleman et al. (2015) - Deep Learning for Galaxy Morphology
    """)


if __name__ == "__main__":
    main()
