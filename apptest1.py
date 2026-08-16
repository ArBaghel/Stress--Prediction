import os
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
os.environ['OMP_NUM_THREADS']       = '16'
os.environ['KMP_BLOCKTIME']         = '1'
os.environ['KMP_AFFINITY']          = 'granularity=fine,compact,1,0'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(12)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

st.set_page_config(
    page_title="ECG Stress Detector",
    page_icon="🫀",
    layout="wide"
)

FS          = 700
WINDOW_SIZE = 250
STEP_SIZE   = 125
THRESHOLD   = 65.15   # optimal threshold from validation set (Youden's J)
MODEL_PATH  = os.path.join(os.path.dirname(__file__),
                           'models', 'stress_ecg_model.h5')

from tensorflow.keras.layers import LSTM as KerasLSTM, Bidirectional

class PatchedLSTM(KerasLSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_model():
    from tensorflow.keras import backend as K
    custom_objects = {
        'tf': tf,
        'K': K,
        'LSTM': PatchedLSTM,
        'Bidirectional': Bidirectional
    }
    return tf.keras.models.load_model(MODEL_PATH,
                                      custom_objects=custom_objects)

def bandpass_filter(ecg):
    nyq  = FS / 2
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    return filtfilt(b, a, ecg).astype(np.float32)

def pan_tompkins_rpeak(ecg, fs=700):
    """
    Pan-Tompkins R-peak detector — identical to training pipeline (notebook Cell 1).
    Bug fix: threshold uses np.percentile(mwi, 99.5) instead of mwi.max() to
    prevent a single motion artifact from raising the threshold above all real beats.
    """
    from scipy.signal import find_peaks
    nyq      = fs / 2
    b, a     = butter(2, [5/nyq, 15/nyq], btype='band')
    sig_bp   = filtfilt(b, a, ecg)
    sig_d    = np.gradient(sig_bp)
    sig_sq   = sig_d ** 2
    win      = max(1, int(0.150 * fs))
    sig_mwi  = np.convolve(sig_sq, np.ones(win) / win, mode='same')
    # ── Robust threshold: clip outlier spikes before computing 0.35×max ──
    mwi_ref   = np.clip(sig_mwi, 0, np.percentile(sig_mwi, 99.5))
    min_dist  = int(0.200 * fs)
    threshold = 0.35 * mwi_ref.max()
    peaks, _  = find_peaks(sig_mwi, height=threshold, distance=min_dist)
    return peaks

def make_windows(ecg):
    """
    R-peak centred windowing — matches training pipeline (notebook Cell 6).
    One window per R-peak, centred on QRS complex, z-score normalised.
    Falls back to sliding window only when <3 peaks detected.
    """
    filtered = bandpass_filter(ecg)
    peaks    = pan_tompkins_rpeak(filtered, fs=FS)
    half     = WINDOW_SIZE // 2

    if len(peaks) < 3:
        # Fallback: sliding window (signal quality too low for R-peak detection)
        windows = []
        for start in range(0, len(filtered) - WINDOW_SIZE, STEP_SIZE):
            seg = filtered[start:start + WINDOW_SIZE]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            windows.append(seg)
        return np.array(windows, dtype=np.float32)[..., np.newaxis] \
               if windows else None

    # R-peak centred windows — same as training
    windows = []
    for peak in peaks:
        start = peak - half
        end   = peak + half
        if start < 0 or end > len(filtered):
            continue
        seg = filtered[start:end]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        windows.append(seg)

    if len(windows) == 0:
        return None
    return np.array(windows, dtype=np.float32)[..., np.newaxis]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
* { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background: #050d1a; }
.header {
    background: linear-gradient(135deg, #0d1f3c, #0a1628);
    border: 1px solid #1a3050; border-radius: 14px;
    padding: 1.8rem 2.2rem; margin-bottom: 1.8rem;
}
.header h1 { color: white; font-size: 2rem; font-weight: 800; margin: 0; }
.header p { color: #6a8caa; margin: 0.3rem 0 0; font-size: 0.9rem; }
.cyan { color: #00e5ff; }
.result-stressed {
    background: #1f0a10; border: 2px solid #ff4b6e;
    border-radius: 14px; padding: 2rem; text-align: center;
}
.result-nostress {
    background: #061818; border: 2px solid #00e5ff;
    border-radius: 14px; padding: 2rem; text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800;
}
.result-sub { color: #6a8caa; font-size: 0.85rem; margin-top: 0.3rem; }
.meter-wrap {
    background: #0d1828; border: 1px solid #1a3050;
    border-radius: 10px; padding: 1rem 1.2rem; margin: 1rem 0;
}
.meter-label {
    color: #6a8caa; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;
}
.meter-track {
    background: #1a2d45; border-radius: 100px; height: 10px; overflow: hidden;
}
.meter-fill-stressed {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #c2185b, #ff4b6e);
}
.meter-fill-nostress {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #0097a7, #00e5ff);
}
.meter-value { color: white; font-size: 1rem; font-weight: 700; text-align: right; margin-top: 0.3rem; }
.stat-card {
    background: #0d1828; border: 1px solid #1a3050;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.stat-label { color: #6a8caa; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
.stat-value { color: white; font-size: 1.4rem; font-weight: 700; font-family: 'Syne', sans-serif; margin-top: 0.2rem; }
.rec-item {
    background: #0d1828; border-left: 3px solid;
    border-radius: 0 8px 8px 0; padding: 0.7rem 1rem;
    margin: 0.4rem 0; color: #c0d0e0; font-size: 0.88rem;
}
.disclaimer {
    background: #0f0e00; border-left: 3px solid #ff9800;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    color: #9a8060; font-size: 0.8rem; margin-top: 1rem;
}
.section-title {
    color: white; font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700; margin: 1.4rem 0 0.6rem;
}
.info-box {
    background: #0d1828; border: 1px solid #1a3050;
    border-radius: 10px; padding: 2.5rem 1rem;
    text-align: center; color: #6a8caa;
}
.stButton > button {
    background: linear-gradient(135deg, #0097a7, #00e5ff) !important;
    color: #050d1a !important; border: none !important;
    border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; width: 100% !important; padding: 0.6rem !important;
}
footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🫀 ECG <span class="cyan">Stress</span> Detector</h1>
    <p>CNN-BiLSTM-DACAM+ — Upload ECG signal or use demo &nbsp;|&nbsp; Running on CPU (oneDNN)</p>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info(f"Expected path: {MODEL_PATH}")
    st.stop()

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown("<div class='section-title'>Input ECG Signal</div>",
                unsafe_allow_html=True)

    mode = st.radio("", ["Upload CSV", "Demo Signal"],
                    horizontal=True, label_visibility="collapsed")
    ecg  = None

    if mode == "Upload CSV":
        file = st.file_uploader(
            "CSV with one ECG value per row at 700 Hz",
            type=['csv', 'txt'], label_visibility="visible")
        if file:
            try:
                import pandas as pd
                ecg = pd.read_csv(file, header=None).iloc[:, 0]\
                        .values.astype(np.float32)
                st.success(f"{len(ecg):,} samples — "
                           f"{len(ecg)/FS:.1f}s")
            except Exception as e:
                st.error(f"File error: {e}")
    else:
        sig_type = st.selectbox(
            "Signal type",
            ["Normal ECG (~65 BPM)", "Stressed ECG (~95 BPM)"])
        duration = st.slider("Duration (seconds)", 10, 60, 30, 5)

        if st.button("Generate Demo"):
            np.random.seed(42)
            n   = duration * FS
            t   = np.linspace(0, duration, n)
            hr  = 65 if "Normal" in sig_type else 95
            nz  = 0.04 if "Normal" in sig_type else 0.12
            sig = np.zeros(n)
            for bt in np.arange(0, t[-1], 60.0/hr):
                sig += np.exp(-0.5 * ((t - bt) / 0.02) ** 2)
            sig += 0.1 * np.sin(2 * np.pi * 0.15 * t)
            sig += nz * np.random.randn(n)
            st.session_state['demo_ecg'] = sig.astype(np.float32)
            st.success(f"Generated — HR: {hr} BPM, {duration}s")

        if 'demo_ecg' in st.session_state:
            ecg = st.session_state['demo_ecg']

    if ecg is not None:
        st.markdown("<div class='section-title'>Preview (5s)</div>",
                    unsafe_allow_html=True)
        preview = min(len(ecg), 5 * FS)
        t_p     = np.arange(preview) / FS
        fig, ax = plt.subplots(figsize=(7, 2.2), facecolor='#0d1828')
        ax.set_facecolor('#0d1828')
        ax.plot(t_p, ecg[:preview], color='#00e5ff', linewidth=0.8)
        ax.fill_between(t_p, ecg[:preview], alpha=0.07, color='#00e5ff')
        ax.set_xlabel('Time (s)', color='#6a8caa', fontsize=8)
        ax.tick_params(colors='#6a8caa', labelsize=7)
        for s in ax.spines.values():
            s.set_color('#1a3050')
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.button("🔬  Analyze ECG", key="run_btn")
    else:
        st.markdown("""
        <div class="info-box">
            <div style="font-size:2.5rem;">📂</div>
            <div style="color:white; margin-top:0.5rem; font-weight:500;">
                No signal loaded</div>
            <div style="margin-top:0.3rem; font-size:0.82rem;">
                Upload CSV or use demo</div>
        </div>""", unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-title'>Results</div>",
                unsafe_allow_html=True)
    run = st.session_state.get("run_btn", False)

    if ecg is not None and run:
        with st.spinner("Analyzing..."):
            windows = make_windows(ecg)

        if windows is None:
            st.error("Signal too short — need at least 0.36s of data.")
        else:
            probs = model.predict(windows, verbose=0).flatten()

            # Fraction of beats above threshold — matches training's majority-vote
            # labelling (extract_rpeak_windows uses label.mean() > 0.5 per window)
            stressed_w = int((probs > THRESHOLD / 100).sum())
            stress_pct = stressed_w / len(probs) * 100
            label      = "Stressed" if stress_pct > THRESHOLD else "No Stress"
            confidence = stress_pct if label == "Stressed" \
                         else 100 - stress_pct
            color      = "#ff4b6e" if label == "Stressed" else "#00e5ff"
            card_cls   = "result-stressed" if label == "Stressed" \
                         else "result-nostress"
            bar_cls    = "meter-fill-stressed" if label == "Stressed" \
                         else "meter-fill-nostress"
            icon       = "⚠️" if label == "Stressed" else "✅"

            st.markdown(f"""
            <div class="{card_cls}">
                <div style="font-size:2.2rem;">{icon}</div>
                <div class="result-title" style="color:{color};">{label}</div>
                <div class="result-sub">
                    {len(windows)} beats analysed
                    ({len(windows)*WINDOW_SIZE/FS:.1f}s equivalent)
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="meter-wrap">
                <div class="meter-label">Model Confidence</div>
                <div class="meter-track">
                    <div class="{bar_cls}" style="width:{confidence:.1f}%"></div>
                </div>
                <div class="meter-value">{confidence:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            for col, lbl, val, col_override in [
                (c1, "Stress Level",    f"{stress_pct:.1f}%", color),
                (c2, "Beats Analysed",  str(len(windows)),    "white"),
                (c3, "Stressed Beats",  str(stressed_w),      "#ff4b6e"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">{lbl}</div>
                        <div class="stat-value"
                             style="color:{col_override};">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div class='section-title'>ECG Analysis</div>",
                        unsafe_allow_html=True)

            fig, axes = plt.subplots(2, 1, figsize=(9, 5),
                                     facecolor='#0d1828')
            fig.subplots_adjust(hspace=0.45)

            ax1 = axes[0]
            ax1.set_facecolor('#0d1828')
            t   = np.arange(len(ecg)) / FS
            ax1.plot(t, ecg, color=color, linewidth=0.7, alpha=0.9)
            ax1.fill_between(t, ecg, alpha=0.06, color=color)
            ax1.set_title('ECG Signal', color='white',
                          fontsize=9, fontweight='bold', pad=6)
            ax1.set_xlabel('Time (s)', color='#6a8caa', fontsize=8)
            ax1.tick_params(colors='#6a8caa', labelsize=7)
            for s in ax1.spines.values():
                s.set_color('#1a3050')

            ax2 = axes[1]
            ax2.set_facecolor('#0d1828')
            ax2.bar(range(len(probs)), probs * 100,
                    color=['#ff4b6e' if p > THRESHOLD / 100 else '#00e5ff'
                           for p in probs],
                    alpha=0.8, width=0.8)
            # threshold line
            ax2.axhline(y=THRESHOLD, color='#ffeb3b', linewidth=1.0,
                        linestyle='--', alpha=0.7)
            ax2.text(len(probs) * 0.01, THRESHOLD + 1.5,
                     f'threshold ({THRESHOLD}%)',
                     color='#ffeb3b', fontsize=7)
            ax2.set_title('Stress Probability per Beat (R-peak centred)',
                          color='white', fontsize=9,
                          fontweight='bold', pad=6)
            ax2.set_xlabel('Beat', color='#6a8caa', fontsize=8)
            ax2.set_ylabel('%',      color='#6a8caa', fontsize=8)
            ax2.set_ylim(0, 100)
            ax2.tick_params(colors='#6a8caa', labelsize=7)
            for s in ax2.spines.values():
                s.set_color('#1a3050')

            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown("<div class='section-title'>Recommendations</div>",
                        unsafe_allow_html=True)

            if label == "Stressed":
                recs = [
                    "Take an immediate break from your current activity.",
                    "Practice deep breathing — inhale 4s, hold 4s, exhale 6s.",
                    "Avoid caffeine for the next few hours.",
                    "Consider speaking with a healthcare professional.",
                    "Try progressive muscle relaxation from feet upward.",
                ] if stress_pct > 75 else [
                    "Take short breaks every 30 minutes.",
                    "Practice light stretching or mindfulness.",
                    "Stay hydrated — aim for 8 glasses of water daily.",
                    "Avoid screens for 10 minutes to rest your eyes.",
                    "A short walk can help reset your mental state.",
                ]
            else:
                recs = [
                    "Your stress levels look healthy — keep it up!",
                    "Maintain your current routine and sleep schedule.",
                    "Regular exercise helps sustain low stress levels.",
                    "Continue any relaxation practices that work for you.",
                    "Good cardiovascular health supports low stress.",
                ]

            for i, rec in enumerate(recs):
                st.markdown(f"""
                <div class="rec-item" style="border-color:{color};">
                    <b style="color:{color};">
                        {['①','②','③','④','⑤'][i]}
                    </b>&nbsp; {rec}
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
                <b style="color:#ff9800;">Disclaimer</b> — Research
                purposes only. Consult a healthcare professional
                for clinical decisions.
            </div>""", unsafe_allow_html=True)

    elif ecg is not None and not run:
        st.markdown("""
        <div class="info-box" style="padding:3rem;">
            <div style="font-size:2.5rem;">🔬</div>
            <div style="color:white; font-weight:600; margin-top:0.6rem;">
                Signal ready</div>
            <div style="margin-top:0.3rem; font-size:0.82rem;">
                Click Analyze ECG to run the model</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box" style="padding:3rem;">
            <div style="font-size:2.5rem;">🫀</div>
            <div style="color:white; font-weight:600; margin-top:0.6rem;">
                Awaiting input</div>
            <div style="margin-top:0.3rem; font-size:0.82rem;">
                Load a signal on the left to begin</div>
        </div>""", unsafe_allow_html=True)
