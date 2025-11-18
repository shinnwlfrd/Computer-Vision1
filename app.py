import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image

# ============================
# PAGE SETTINGS
# ============================
st.set_page_config(
    page_title="Deteksi Kematangan Tomat",
    layout="wide"
)

# ============================
# RESPONSIVE THEME (AUTO DARK/LIGHT BASED ON SYSTEM)
# ============================
st.markdown("""
<style>
/* HAPUS LATAR BELAKANG BAWAAN STREAMLIT */
*, .css-1d391kg, .css-1iyw2u1, .css-12oz5g7, .css-18e3th9,
.st-emotion-cache-1v0mbdj, .st-emotion-cache-12w0qpk, section {
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

/* TARGET KONTAINER UTAMA STREAMLIT */
[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* DARK MODE (DEFAULT FALLBACK + SYSTEM DARK) */
@media (prefers-color-scheme: dark) {
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
    }
    .main-title {
        color: #60a5fa;
    }
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        color: #e2e8f0;
    }
    .result-box {
        background: rgba(59,130,246,0.15);
        border-left: 4px solid #3b82f6;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    .stMarkdown, p, div, span, li, ul, ol, h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }
    input, textarea, .stFileUploader {
        color: #e2e8f0 !important;
        background-color: rgba(0,0,0,0.2) !important;
    }
}

/* LIGHT MODE (SYSTEM LIGHT) */
@media (prefers-color-scheme: light) {
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }
    .main-title {
        color: #2563eb;
    }
    .card {
        background: rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.1);
        color: #1e293b;
    }
    .result-box {
        background: rgba(37,99,235,0.1);
        border-left: 4px solid #2563eb;
        color: #1e293b;
    }
    [data-testid="stSidebar"] {
        background-color: #e2e8f0 !important;
        color: #1e293b !important;
    }
    .stMarkdown, p, div, span, li, ul, ol, h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
    input, textarea, .stFileUploader {
        color: #1e293b !important;
        background-color: white !important;
    }
}

/* GLOBAL STYLING */
.main-title {
    font-size: 34px;
    font-weight: bold;
    text-align: center;
    padding: 10px 0 30px 0;
}

.card {
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.result-box {
    padding: 18px;
    border-radius: 8px;
}

.stButton > button {
    width: 100%;
    background: #2563eb !important;
    color: white !important;
    padding: 10px;
    border-radius: 10px;
    font-size: 16px;
    border: none;
}
.stButton > button:hover {
    background: #1e40af !important;
}
</style>
""", unsafe_allow_html=True)

# ============================
# LOAD MODELS
# ============================
@st.cache_resource
def load_models():
    try:
        cnn_model = tf.keras.models.load_model("cnn_feature_extractor.h5")
        knn_model = joblib.load("knn_classifier.pkl")
        return cnn_model, knn_model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

cnn, knn = load_models()
IMG_SIZE = 128

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE)).convert("L")
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=[0, -1])
    return arr

def hybrid_predict(arr):
    feature = cnn.predict(arr, verbose=0)
    label = knn.predict(feature)[0]
    # Asumsi: 1 = ripe/mature, 0 = unripe/immature
    # Anda bisa sesuaikan berdasarkan label asli model Anda
    return "matang" if label == 1 else "belum matang"

# ============================
# UI LAYOUT
# ============================
st.markdown("<div class='main-title'>Deteksi Kematangan Tomat (Hybrid CNN + KNN)</div>", unsafe_allow_html=True)

left, right = st.columns([1.3, 1])

# LEFT SIDE
with left:
    st.markdown("<div class='card'><b>📤 Upload Gambar Tomat</b></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Pilih gambar Tomat", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar diupload", width=300)

        st.markdown("<div class='card'><b>🔍 Ekstraksi Ciri Citra</b></div>", unsafe_allow_html=True)
        arr = preprocess_image(img)
        st.success("Ekstraksi selesai.")

        st.markdown("<div class='card'><b>🤖 Klasifikasi</b></div>", unsafe_allow_html=True)
        if st.button("Prediksi"):
            result = hybrid_predict(arr)
            st.markdown(f"""
            <div class='result-box'>
                <h4>Hasil Prediksi:</h4>
                <p><b>Tomat terdeteksi sebagai:</b> <code>{result.upper()}</code></p>
            </div>
            """, unsafe_allow_html=True)

# RIGHT SIDE
with right:
    st.markdown("<div class='card'><b>ℹ️ Informasi Aplikasi</b></div>", unsafe_allow_html=True)
    st.write("""
Aplikasi ini mendeteksi kematangan tomat menggunakan Hybrid CNN + KNN:
- CNN sebagai extractor fitur
- KNN sebagai classifier
- Input citra grayscale 128×128  
- Model dimuat sekali (cached)
""")
