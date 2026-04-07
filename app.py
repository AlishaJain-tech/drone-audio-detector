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
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');
 
    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
 
    .stApp {
        background: #030712;
        background-image:
            radial-gradient(ellipse at 20% 20%, rgba(0,80,200,0.18) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 80%, rgba(0,200,255,0.10) 0%, transparent 60%),
            repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(0,120,255,0.03) 40px, rgba(0,120,255,0.03) 41px),
            repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(0,120,255,0.03) 40px, rgba(0,120,255,0.03) 41px);
    }
 
    .hero-wrap { text-align: center; padding: 40px 0 10px 0; }
    .hero-icon {
        font-size: 4.5rem; display: block;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 0 24px rgba(0,160,255,0.7));
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-10px); }
    }
    .hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.6rem; font-weight: 900;
        background: linear-gradient(90deg, #00aaff, #00e5ff, #ffffff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin: 12px 0 6px 0; letter-spacing: 2px;
    }
    .hero-sub { color: #4a9eff; font-size: 1.05rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 8px; }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #0066ff, #00aaff, #0066ff, transparent);
        margin: 24px 0; border: none;
    }
    .model-label { font-family: 'Orbitron', monospace; font-size: 0.7rem; color: #4a9eff; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
    div[data-testid="stRadio"] label { font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important; color: #a0c8ff !important; font-size: 1rem !important; }
    div[data-testid="stFileUploader"] { background: rgba(0,40,100,0.25) !important; border: 1.5px dashed rgba(0,150,255,0.4) !important; border-radius: 16px !important; padding: 12px !important; }
    div[data-testid="stFileUploader"] label { color: #4a9eff !important; font-weight: 700 !important; font-size: 1rem !important; font-family: 'Rajdhani', sans-serif !important; }
 
    .stButton > button {
        background: linear-gradient(135deg, #0050c8, #0090ff) !important;
        color: white !important; font-family: 'Orbitron', monospace !important;
        font-size: 0.95rem !important; font-weight: 700 !important;
        letter-spacing: 2px !important; border: none !important;
        border-radius: 12px !important; padding: 16px !important;
        width: 100% !important; margin-top: 10px !important;
        box-shadow: 0 0 24px rgba(0,120,255,0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover { box-shadow: 0 0 40px rgba(0,160,255,0.7) !important; transform: translateY(-2px) !important; }
 
    .result-drone {
        background: linear-gradient(135deg, rgba(0,50,180,0.6), rgba(0,100,255,0.4));
        border: 1.5px solid rgba(0,150,255,0.6); border-radius: 20px;
        padding: 36px; text-align: center; color: white;
        font-family: 'Orbitron', monospace; font-size: 1.8rem; font-weight: 900;
        margin: 20px 0; letter-spacing: 2px;
        box-shadow: 0 0 40px rgba(0,120,255,0.35), inset 0 0 40px rgba(0,60,180,0.2);
        animation: pulse-blue 2s ease-in-out infinite;
    }
    @keyframes pulse-blue {
        0%, 100% { box-shadow: 0 0 40px rgba(0,120,255,0.35), inset 0 0 40px rgba(0,60,180,0.2); }
        50%       { box-shadow: 0 0 60px rgba(0,160,255,0.55), inset 0 0 40px rgba(0,80,200,0.3); }
    }
    .result-none {
        background: linear-gradient(135deg, rgba(0,80,40,0.6), rgba(0,160,80,0.3));
        border: 1.5px solid rgba(0,200,100,0.5); border-radius: 20px;
        padding: 36px; text-align: center; color: white;
        font-family: 'Orbitron', monospace; font-size: 1.8rem; font-weight: 900;
        margin: 20px 0; letter-spacing: 2px; box-shadow: 0 0 40px rgba(0,180,80,0.25);
    }
    .result-sub { font-family: 'Rajdhani', sans-serif; font-size: 1rem; font-weight: 600; margin-top: 8px; opacity: 0.85; letter-spacing: 1px; }
    .bar-label { font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 0.98rem; margin: 12px 0 4px 0; letter-spacing: 1px; }
 
    div[data-testid="stMetricValue"] { font-family: 'Orbitron', monospace !important; font-size: 1.6rem !important; font-weight: 700 !important; color: #00aaff !important; }
    div[data-testid="stMetricLabel"] { font-family: 'Rajdhani', sans-serif !important; color: #4a9eff !important; font-weight: 600 !important; letter-spacing: 1px !important; }
    div[data-testid="metric-container"] { background: rgba(0,40,100,0.3) !important; border: 1px solid rgba(0,100,200,0.3) !important; border-radius: 12px !important; padding: 16px !important; }
 
    .upload-hint { background: rgba(0,30,80,0.35); border: 1.5px dashed rgba(0,100,200,0.35); border-radius: 20px; padding: 48px 32px; text-align: center; margin: 16px 0 24px 0; }
    .upload-hint-icon { font-size: 4rem; display: block; margin-bottom: 14px; filter: drop-shadow(0 0 12px rgba(0,150,255,0.5)); }
    .upload-hint-text { font-family: 'Rajdhani', sans-serif; color: #4a9eff; font-size: 1.15rem; font-weight: 600; letter-spacing: 1px; }
    .upload-hint-formats { font-family: 'Rajdhani', sans-serif; color: #2a5a8a; font-size: 0.88rem; margin-top: 8px; letter-spacing: 2px; }
    .footer { text-align: center; font-family: 'Orbitron', monospace; font-size: 0.6rem; color: #1a3a6a; letter-spacing: 3px; padding: 20px 0 10px 0; }
</style>
""", unsafe_allow_html=True)
 
FILE_IDS = {
    'scaler.pkl':        '1faY2oxqqMaidiLc457oXX4a2hSNGIqY6',
    'label_encoder.pkl': '1qeU8d14PGK3hHeGo6q0XDzJlfGsxoL_Q',
    'random_forest.pkl': '1YHuxArWBovqv7qegwLgtQvRWbpP-MT77',
    'mlp.pkl':           '1hmZyg2Ro2dblb5Akwu5pIlYhzq1Auny7',
}
 
@st.cache_resource(show_spinner=False)
def load_models():
    import tempfile, subprocess, sys
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
    tmpdir = tempfile.mkdtemp()
    for fname, fid in FILE_IDS.items():
        out_path = os.path.join(tmpdir, fname)
        try:
            gdown.download(f"https://drive.google.com/uc?id={fid}", out_path, quiet=True, fuzzy=True)
        except Exception as e:
            return None, None, None, None, f"{fname} download failed: {e}"
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
            return None, None, None, None, f"{fname} missing or empty"
    try:
        scaler = joblib.load(os.path.join(tmpdir, 'scaler.pkl'))
        le     = joblib.load(os.path.join(tmpdir, 'label_encoder.pkl'))
        rf     = joblib.load(os.path.join(tmpdir, 'random_forest.pkl'))
        mlp    = joblib.load(os.path.join(tmpdir, 'mlp.pkl'))
        return scaler, le, rf, mlp, None
    except Exception as e:
        return None, None, None, None, f"Model load failed: {e}"
 
SR = 22050
DURATION = 3
N_MFCC = 40
 
def extract_features(audio_bytes):
    try:
        import librosa
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR, duration=DURATION, mono=True)
        target_len = SR * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
        feat = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        feat.extend(np.mean(mfcc, axis=1)); feat.extend(np.std(mfcc, axis=1))
        feat.extend(np.mean(librosa.feature.delta(mfcc), axis=1))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feat.extend(np.mean(chroma, axis=1)); feat.extend(np.std(chroma, axis=1))
        mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
        feat.append(np.mean(mel_db)); feat.append(np.std(mel_db))
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        feat.extend(np.mean(sc, axis=1))
        feat.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        feat.append(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        feat.append(np.mean(librosa.feature.zero_crossing_rate(y)))
        feat.append(np.std(librosa.feature.zero_crossing_rate(y)))
        rms = librosa.feature.rms(y=y)
        feat.append(np.mean(rms)); feat.append(np.std(rms))
        return np.array(feat, dtype=np.float32)
    except:
        return None
 
# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <span class="hero-icon">🚁</span>
    <div class="hero-title">DRONE AUDIO DETECTOR</div>
    <div class="hero-sub">⚡ AI-POWERED ACOUSTIC THREAT ANALYSIS ⚡</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)
 
# ── Silent model load ────────────────────────────────────────────────────────
with st.spinner("Initializing detection system..."):
    scaler, le, rf, mlp, error = load_models()
 
if error or scaler is None:
    st.error(f"❌ System initialization failed: {error}")
    st.stop()
 
# ── Model selector ───────────────────────────────────────────────────────────
available = {}
if rf  is not None: available["🌲 Random Forest"] = "rf"
if mlp is not None: available["🧠 Neural Network"] = "mlp"
 
st.markdown('<div class="model-label">▸ SELECT DETECTION ENGINE</div>', unsafe_allow_html=True)
model_label  = st.radio("", list(available.keys()), horizontal=True, label_visibility="collapsed")
model_choice = available[model_label]
st.markdown('<hr class="divider">', unsafe_allow_html=True)
 
# ── Audio upload ─────────────────────────────────────────────────────────────
audio_file = st.file_uploader("🎵 Upload Audio File", type=["wav","mp3","ogg","flac"], label_visibility="visible")
 
if audio_file:
    st.audio(audio_file, format="audio/wav")
    audio_bytes = audio_file.read()
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
 
    if st.button("⚡ ANALYZE AUDIO"):
        with st.spinner("🔊 Processing audio signal..."):
            features = extract_features(audio_bytes)
        if features is None:
            st.error("❌ Audio processing failed. Please try again.")
        else:
            feat_scaled = scaler.transform([features])
            with st.spinner("🤖 Running detection model..."):
                if model_choice == "rf":
                    pred = rf.predict(feat_scaled)[0]; proba = rf.predict_proba(feat_scaled)[0]
                else:
                    pred = mlp.predict(feat_scaled)[0]; proba = mlp.predict_proba(feat_scaled)[0]
 
            label      = le.inverse_transform([pred])[0]
            confidence = proba[pred] * 100
            classes    = le.classes_
 
            if "drone" in label.lower():
                st.markdown(f'<div class="result-drone">🚁 DRONE DETECTED<div class="result-sub">Acoustic signature matched · {confidence:.1f}% confidence</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-none">✅ NO DRONE DETECTED<div class="result-sub">Environment clear · {confidence:.1f}% confidence</div></div>', unsafe_allow_html=True)
 
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{confidence:.1f}%")
            c2.metric(f"P({classes[0]})", f"{proba[0]*100:.1f}%")
            if len(classes) > 1:
                c3.metric(f"P({classes[1]})", f"{proba[1]*100:.1f}%")
 
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            for i, cls in enumerate(classes):
                pct   = int(proba[i] * 100)
                color = "#0090ff" if "drone" in cls.lower() else "#00c864"
                glow  = "rgba(0,144,255,0.4)" if "drone" in cls.lower() else "rgba(0,200,100,0.3)"
                icon  = "🚁" if "drone" in cls.lower() else "✅"
                st.markdown(f"""
                <div class="bar-label" style="color:{color};">{icon} {cls.upper()} — {pct}%</div>
                <div style="background:rgba(0,20,60,0.5);border-radius:8px;overflow:hidden;height:22px;margin-bottom:14px;border:1px solid rgba(0,80,180,0.2);">
                    <div style="width:{pct}%;background:linear-gradient(90deg,{color},{color}aa);height:100%;border-radius:8px;box-shadow:0 0 12px {glow};"></div>
                </div>""", unsafe_allow_html=True)
 
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            if "drone" in label.lower():
                st.error(f"⚠️ ALERT: Drone acoustic signature detected with {confidence:.1f}% confidence!")
            else:
                st.success(f"🟢 CLEAR: No drone detected. Confidence: {confidence:.1f}%")
else:
    st.markdown("""
    <div class="upload-hint">
        <span class="upload-hint-icon">🎵</span>
        <div class="upload-hint-text">Upload an audio file to begin analysis</div>
        <div class="upload-hint-formats">WAV &nbsp;·&nbsp; MP3 &nbsp;·&nbsp; OGG &nbsp;·&nbsp; FLAC</div>
    </div>""", unsafe_allow_html=True)
 
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="footer">🚁 DRONE AUDIO DETECTOR · RF / MLP · ACOUSTIC AI SYSTEM</div>', unsafe_allow_html=True)