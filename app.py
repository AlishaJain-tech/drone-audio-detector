import streamlit as st
import numpy as np
import joblib
import os
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Drone Audio Detector",
    page_icon="🚁",
    layout="centered"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 900;
        color: #1976d2;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 20px;
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
        font-size: 1.1rem;
        font-weight: 800;
        border: none;
        border-radius: 12px;
        padding: 14px;
        width: 100%;
        margin-top: 8px;
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

# ─── Google Drive file IDs ────────────────────────────────────────────────────
# Folder: https://drive.google.com/drive/folders/17iZkBjDfyy-l6Co0qZ1Afrt-Dwn2rC69
DRIVE_FILES = {
    'scaler.pkl':          None,
    'label_encoder.pkl':   None,
    'random_forest.pkl':   None,
    'mlp.pkl':             None,
}
FOLDER_ID = "17iZkBjDfyy-l6Co0qZ1Afrt-Dwn2rC69"

@st.cache_resource(show_spinner=False)
def load_models_from_drive():
    import requests
    import pickle

    def download_file(filename):
        # Google Drive folder listing API
        api_url = (
            f"https://www.googleapis.com/drive/v3/files"
            f"?q='{FOLDER_ID}'+in+parents+and+name='{filename}'"
            f"&key=AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY"
            f"&fields=files(id,name)"
        )
        # Direct download using file ID from known mapping
        download_url = f"https://drive.google.com/uc?export=download&id="
        return download_url

    def gdrive_download(file_id):
        """Download a file from Google Drive by ID."""
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True)
        # Handle large file confirmation
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
                response = session.get(url, stream=True)
                break
        return response.content

    def list_folder_files():
        """List all files in the shared folder."""
        url = (
            f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        )
        # Use public folder API
        api = (
            f"https://www.googleapis.com/drive/v3/files"
            f"?q=%27{FOLDER_ID}%27+in+parents"
            f"&fields=files(id,name)"
            f"&supportsAllDrives=true"
        )
        # Fallback: use gdown
        return None

    # Use gdown to download files from folder
    import subprocess
    import sys
    import tempfile

    # Install gdown if not present
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    tmpdir = tempfile.mkdtemp()

    try:
        # Download entire folder
        gdown.download_folder(
            id=FOLDER_ID,
            output=tmpdir,
            quiet=True,
            use_cookies=False
        )
    except Exception as e:
        return None, None, None, None, str(e)

    # Load models
    scaler, le, rf, mlp = None, None, None, None

    try:
        scaler_path = os.path.join(tmpdir, 'scaler.pkl')
        le_path     = os.path.join(tmpdir, 'label_encoder.pkl')
        if os.path.exists(scaler_path): scaler = joblib.load(scaler_path)
        if os.path.exists(le_path):     le     = joblib.load(le_path)
    except Exception as e:
        return None, None, None, None, f"Scaler/LE load failed: {e}"

    try:
        rf_path = os.path.join(tmpdir, 'random_forest.pkl')
        if os.path.exists(rf_path): rf = joblib.load(rf_path)
    except: pass

    try:
        mlp_path = os.path.join(tmpdir, 'mlp.pkl')
        if os.path.exists(mlp_path): mlp = joblib.load(mlp_path)
    except: pass

    if scaler is None or le is None:
        return None, None, None, None, "scaler.pkl ya label_encoder.pkl nahi mila Drive mein"

    if rf is None and mlp is None:
        return None, None, None, None, "Koi model file nahi mili Drive mein"

    return scaler, le, rf, mlp, None

# ─── Feature extraction ───────────────────────────────────────────────────────
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
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        feat.extend(np.mean(mfcc, axis=1))
        feat.extend(np.std(mfcc,  axis=1))
        delta = librosa.feature.delta(mfcc)
        feat.extend(np.mean(delta, axis=1))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feat.extend(np.mean(chroma, axis=1))
        feat.extend(np.std(chroma,  axis=1))
        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feat.append(np.mean(mel_db))
        feat.append(np.std(mel_db))
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        feat.extend(np.mean(sc, axis=1))
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

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🚁 Drone Audio Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an audio file — instantly know if a drone is present</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ─── Load models ──────────────────────────────────────────────────────────────
with st.spinner("🔄 Models load ho rahe hain... (pehli baar thoda time lagega)"):
    scaler, le, rf, mlp, error = load_models_from_drive()

if error or scaler is None:
    st.error(f"❌ Models load nahi hue: {error}")
    st.stop()

# ─── Model selector ───────────────────────────────────────────────────────────
available = {}
if rf  is not None: available["🌲 Random Forest"] = "rf"
if mlp is not None: available["🧠 MLP Neural Net"] = "mlp"

model_label  = st.radio("Model:", list(available.keys()),
                         horizontal=True, label_visibility="collapsed")
model_choice = available[model_label]
st.markdown("---")

# ─── Audio upload ─────────────────────────────────────────────────────────────
audio_file = st.file_uploader(
    "🎵 Audio file upload karo (.wav / .mp3)",
    type=["wav", "mp3", "ogg", "flac"],
    label_visibility="visible"
)

if audio_file:
    st.audio(audio_file, format="audio/wav")
    audio_bytes = audio_file.read()
    st.markdown("---")

    if st.button("🔍 Detect Drone"):
        with st.spinner("🔊 Audio analyse ho rahi hai..."):
            features = extract_features(audio_bytes)

        if features is None:
            st.error("❌ Audio process nahi ho payi. Dobaara try karo.")
        else:
            feat_scaled = scaler.transform([features])

            with st.spinner("🤖 Detecting..."):
                if model_choice == "rf":
                    pred  = rf.predict(feat_scaled)[0]
                    proba = rf.predict_proba(feat_scaled)[0]
                else:
                    pred  = mlp.predict(feat_scaled)[0]
                    proba = mlp.predict_proba(feat_scaled)[0]

            label      = le.inverse_transform([pred])[0]
            confidence = proba[pred] * 100
            classes    = le.classes_

            if "drone" in label.lower():
                st.markdown(
                    '<div class="drone-box">🚁 DRONE SOUND DETECTED</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="unknown-box">✅ NO DRONE DETECTED</div>',
                    unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence",       f"{confidence:.1f}%")
            c2.metric(f"P({classes[0]})", f"{proba[0]*100:.1f}%")
            if len(classes) > 1:
                c3.metric(f"P({classes[1]})", f"{proba[1]*100:.1f}%")

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

            st.markdown("---")
            if "drone" in label.lower():
                st.error(f"⚠️ Drone sound hai! ({confidence:.1f}% confident)")
            else:
                st.success(f"🟢 Koi drone nahi! ({confidence:.1f}% confident)")

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

st.markdown("---")
st.markdown(
    "<center><sub>🚁 Drone Audio Detector · RF / MLP · Streamlit</sub></center>",
    unsafe_allow_html=True)
