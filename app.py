import streamlit as st
import numpy as np
import joblib
import os
import io
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drone Audio Detector",
    page_icon="🚁",
    layout="centered"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 900;
        color: #1976d2;
        margin-bottom: 4px;
        letter-spacing: 1px;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 28px;
    }
    .drone-box {
        background: linear-gradient(135deg, #0d47a1, #1565c0);
        border-radius: 18px;
        padding: 34px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: 900;
        margin: 24px 0 16px 0;
        box-shadow: 0 6px 28px rgba(21,101,192,0.45);
        letter-spacing: 1px;
    }
    .unknown-box {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 18px;
        padding: 34px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: 900;
        margin: 24px 0 16px 0;
        box-shadow: 0 6px 28px rgba(46,125,50,0.45);
        letter-spacing: 1px;
    }
    .bar-label {
        font-weight: 700;
        font-size: 0.95rem;
        margin: 10px 0 3px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #1976d2);
        color: white;
        font-size: 1.15rem;
        font-weight: 800;
        border: none;
        border-radius: 12px;
        padding: 15px;
        width: 100%;
        margin-top: 8px;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover { opacity: 0.87; }
    .upload-hint {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        color: #7f8c8d;
        margin: 10px 0 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Audio feature extraction (exact same as notebook) ────────────────────────
SR       = 22050
DURATION = 3
N_MFCC   = 40

def extract_features(audio_bytes):
    try:
        import librosa
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR,
                             duration=DURATION, mono=True)
        target_len = SR * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]

        feat = []

        # 1. MFCCs mean + std  →  80 features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        feat.extend(np.mean(mfcc, axis=1))
        feat.extend(np.std(mfcc,  axis=1))

        # 2. Delta MFCCs mean  →  40 features
        delta = librosa.feature.delta(mfcc)
        feat.extend(np.mean(delta, axis=1))

        # 3. Chroma mean + std  →  24 features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feat.extend(np.mean(chroma, axis=1))
        feat.extend(np.std(chroma,  axis=1))

        # 4. Mel-Spectrogram mean + std  →  2 features
        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feat.append(np.mean(mel_db))
        feat.append(np.std(mel_db))

        # 5. Spectral Contrast mean  →  7 features
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        feat.extend(np.mean(sc, axis=1))

        # 6. Rolloff, ZCR, RMS  →  6 features
        feat.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        feat.append(np.std( librosa.feature.spectral_rolloff(y=y, sr=sr)))
        feat.append(np.mean(librosa.feature.zero_crossing_rate(y)))
        feat.append(np.std( librosa.feature.zero_crossing_rate(y)))
        rms = librosa.feature.rms(y=y)
        feat.append(np.mean(rms))
        feat.append(np.std(rms))

        return np.array(feat, dtype=np.float32)
    except:
        return None

# ─── Load models from folder (automatic — no user upload needed) ───────────────
@st.cache_resource
def load_models():
    """Auto-load pkl files from same folder as app.py"""
    base = os.path.dirname(os.path.abspath(__file__))
    errors = []

    # Scaler & Label Encoder — required
    scaler_path = os.path.join(base, 'scaler.pkl')
    le_path     = os.path.join(base, 'label_encoder.pkl')

    if not os.path.exists(scaler_path):
        return None, None, None, None, None, ["❌ scaler.pkl not found in app folder"]
    if not os.path.exists(le_path):
        return None, None, None, None, None, ["❌ label_encoder.pkl not found in app folder"]

    scaler = joblib.load(scaler_path)
    le     = joblib.load(le_path)

    # Models — at least one required
    rf, mlp, cnn = None, None, None

    rf_path = os.path.join(base, 'random_forest.pkl')
    if os.path.exists(rf_path):
        try:    rf = joblib.load(rf_path)
        except: errors.append("RF load failed")

    mlp_path = os.path.join(base, 'mlp.pkl')
    if os.path.exists(mlp_path):
        try:    mlp = joblib.load(mlp_path)
        except: errors.append("MLP load failed")

    cnn_path = os.path.join(base, 'cnn_model.h5')
    if os.path.exists(cnn_path):
        try:
            import tensorflow as tf
            cnn = tf.keras.models.load_model(cnn_path)
        except: errors.append("CNN load failed")

    if rf is None and mlp is None and cnn is None:
        errors.append("❌ No model file found (random_forest.pkl / mlp.pkl / cnn_model.h5)")

    return scaler, le, rf, mlp, cnn, errors

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🚁 Drone Audio Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an audio file — instantly know if a drone is present</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Load models silently ───────────────────────────────────────────────────────
scaler, le, rf, mlp, cnn, load_errors = load_models()

# If models missing — show friendly one-time instruction
if scaler is None or (rf is None and mlp is None and cnn is None):
    st.error("⚠️ Model files not found in the app folder!")
    st.markdown("""
### 📋 Ek baar karna hai — bas ye files app folder mein rakh do:

| File | Kahan se milegi |
|------|----------------|
| `scaler.pkl` | Google Colab se download karo |
| `label_encoder.pkl` | Google Colab se download karo |
| `random_forest.pkl` | Google Colab se download karo |
| `mlp.pkl` | Google Colab se download karo |
| `cnn_model.h5` | Google Colab se download karo |

**Colab mein ye run karo:**
```python
from google.colab import files
files.download('scaler.pkl')
files.download('label_encoder.pkl')
files.download('random_forest.pkl')
files.download('mlp.pkl')
files.download('cnn_model.h5')
```

**Phir saari files is folder mein rakh do:**
```
AudioDataSet/
├── app.py          ← ye file
├── scaler.pkl      ← yahan
├── label_encoder.pkl
├── random_forest.pkl
├── mlp.pkl
└── cnn_model.h5
```
**Phir terminal mein:** `streamlit run app.py`
    """)
    st.stop()

# ── Build available model list ─────────────────────────────────────────────────
available = {}
if rf  is not None: available["🌲 Random Forest"] = "rf"
if mlp is not None: available["🧠 MLP Neural Net"] = "mlp"
if cnn is not None: available["🔷 1D CNN"]         = "cnn"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI — Audio Upload + Predict
# ══════════════════════════════════════════════════════════════════════════════

# Model selector (small, clean)
model_label  = st.radio("Model:", list(available.keys()),
                         horizontal=True, label_visibility="collapsed")
model_choice = available[model_label]

st.markdown("---")

# Audio uploader
audio_file = st.file_uploader(
    "🎵 Audio file upload karo (.wav / .mp3)",
    type=["wav", "mp3", "ogg", "flac"],
    label_visibility="visible"
)

if audio_file:
    # Play the audio
    st.audio(audio_file, format="audio/wav")
    audio_bytes = audio_file.read()
    st.markdown("---")

    # Predict button
    if st.button("🔍 Detect Drone"):
        # Feature extraction
        with st.spinner("🔊 Audio analyse ho rahi hai..."):
            features = extract_features(audio_bytes)

        if features is None:
            st.error("❌ Audio file process nahi ho payi. Dobaara try karo.")
        else:
            feat_scaled = scaler.transform([features])

            # Predict with chosen model
            with st.spinner("🤖 Detecting..."):
                if model_choice == "rf":
                    pred  = rf.predict(feat_scaled)[0]
                    proba = rf.predict_proba(feat_scaled)[0]
                elif model_choice == "mlp":
                    pred  = mlp.predict(feat_scaled)[0]
                    proba = mlp.predict_proba(feat_scaled)[0]
                elif model_choice == "cnn":
                    fc    = feat_scaled.reshape(1, feat_scaled.shape[1], 1)
                    proba = cnn.predict(fc, verbose=0)[0]
                    pred  = int(np.argmax(proba))

            label      = le.inverse_transform([pred])[0]
            confidence = proba[pred] * 100
            classes    = le.classes_

            # ── Result ────────────────────────────────────────────────────────
            if "drone" in label.lower():
                st.markdown(
                    '<div class="drone-box">🚁 DRONE SOUND DETECTED</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="unknown-box">✅ NO DRONE DETECTED</div>',
                    unsafe_allow_html=True)

            # ── 3 metrics ─────────────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence",        f"{confidence:.1f}%")
            c2.metric(f"P({classes[0]})",  f"{proba[0]*100:.1f}%")
            if len(classes) > 1:
                c3.metric(f"P({classes[1]})", f"{proba[1]*100:.1f}%")

            # ── Probability bars ──────────────────────────────────────────────
            st.markdown("---")
            for i, cls in enumerate(classes):
                pct   = int(proba[i] * 100)
                color = "#1565c0" if "drone" in cls.lower() else "#2e7d32"
                icon  = "🚁" if "drone" in cls.lower() else "✅"
                st.markdown(
                    f'<div class="bar-label" style="color:{color};">'
                    f'{icon} {cls} — {pct}%</div>'
                    f'<div style="background:#1e1e2e;border-radius:8px;'
                    f'overflow:hidden;height:24px;margin-bottom:10px;">'
                    f'<div style="width:{pct}%;background:{color};'
                    f'height:100%;border-radius:8px;"></div></div>',
                    unsafe_allow_html=True)

            # ── Final verdict ─────────────────────────────────────────────────
            st.markdown("---")
            if "drone" in label.lower():
                st.error(f"⚠️ Drone sound hai is audio mein! ({confidence:.1f}% confident)")
            else:
                st.success(f"🟢 Koi drone nahi — background/unknown sound hai. ({confidence:.1f}% confident)")

else:
    st.markdown("""
    <div class="upload-hint">
        <div style="font-size:3.5rem;">🎵</div>
        <div style="font-size:1.15rem;margin-top:12px;color:#bdc3c7;">
            Upar se audio file upload karo
        </div>
        <div style="font-size:0.88rem;margin-top:6px;">
            Supported: .wav &nbsp;·&nbsp; .mp3 &nbsp;·&nbsp; .ogg &nbsp;·&nbsp; .flac
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><sub>🚁 Drone Audio Detector · RF / MLP / CNN · Streamlit</sub></center>",
    unsafe_allow_html=True)
