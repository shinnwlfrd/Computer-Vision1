import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image

# ============================
# PAGE SETTINGS
# ============================
st.set_page_config(
    page_title="🍅 Deteksi Kematangan Tomat",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# ENHANCED RESPONSIVE THEME
# ============================
st.markdown("""
<style>
/* RESET STREAMLIT DEFAULTS */
*, .css-1d391kg, .css-1iyw2u1, .css-12oz5g7, .css-18e3th9,
.st-emotion-cache-1v0mbdj, .st-emotion-cache-12w0qpk, section {
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

/* MAIN CONTAINER */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}

/* DARK MODE */
@media (prefers-color-scheme: dark) {
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    }
    .hero-section {
        background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(139,92,246,0.15) 100%);
        border: 1px solid rgba(59,130,246,0.3);
    }
    .card {
        background: rgba(30,41,59,0.6);
        border: 1px solid rgba(100,116,139,0.3);
        backdrop-filter: blur(10px);
    }
    .upload-zone {
        background: rgba(15,23,42,0.4);
        border: 2px dashed rgba(100,116,139,0.5);
    }
    .result-success {
        background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(22,163,74,0.2) 100%);
        border-left: 4px solid #22c55e;
    }
    .result-warning {
        background: linear-gradient(135deg, rgba(251,146,60,0.2) 0%, rgba(249,115,22,0.2) 100%);
        border-left: 4px solid #fb923c;
    }
    .feature-box {
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(139,92,246,0.2) 100%);
        border: 1px solid rgba(139,92,246,0.3);
    }
}

/* LIGHT MODE */
@media (prefers-color-scheme: light) {
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
        color: #0f172a !important;
    }
    .hero-section {
        background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(59,130,246,0.2);
    }
    .card {
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(226,232,240,0.8);
        backdrop-filter: blur(10px);
    }
    .upload-zone {
        background: rgba(248,250,252,0.6);
        border: 2px dashed rgba(148,163,184,0.5);
    }
    .result-success {
        background: linear-gradient(135deg, rgba(34,197,94,0.15) 0%, rgba(22,163,74,0.15) 100%);
        border-left: 4px solid #16a34a;
        color: #0f172a !important;
    }
    .result-warning {
        background: linear-gradient(135deg, rgba(251,146,60,0.15) 0%, rgba(249,115,22,0.15) 100%);
        border-left: 4px solid #ea580c;
        color: #0f172a !important;
    }
    .feature-box {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.15);
        color: #0f172a !important;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(139,92,246,0.2);
        color: #0f172a !important;
    }
    .stMarkdown, p, div, span, li, h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }
}

/* HERO SECTION */
.hero-section {
    text-align: center;
    padding: 40px 20px;
    border-radius: 20px;
    margin-bottom: 40px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.8;
    margin-bottom: 0;
}

/* CARD STYLES */
.card {
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* UPLOAD ZONE */
.upload-zone {
    padding: 40px;
    border-radius: 16px;
    text-align: center;
    margin: 20px 0;
    transition: all 0.3s ease;
}

.upload-zone:hover {
    border-color: #3b82f6;
    background: rgba(59,130,246,0.05);
}

/* RESULT BOXES */
.result-success, .result-warning {
    padding: 24px;
    border-radius: 12px;
    margin: 20px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}

.result-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 12px;
}

.result-value {
    font-size: 32px;
    font-weight: 800;
    margin: 16px 0;
}

/* FEATURE BOXES */
.feature-box {
    padding: 20px;
    border-radius: 12px;
    margin: 12px 0;
    transition: transform 0.2s ease;
}

.feature-box:hover {
    transform: translateX(8px);
}

.feature-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
}

/* METRIC CARDS */
.metric-card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin: 10px 0;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    margin: 8px 0;
}

.metric-label {
    font-size: 14px;
    opacity: 0.7;
}

/* BUTTON STYLING */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    padding: 16px 32px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 16px rgba(59,130,246,0.3);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.4);
}

/* IMAGE CONTAINER */
.image-container {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    margin: 20px 0;
}

/* PROCESS STEPS */
.process-step {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    margin: 12px 0;
    border-radius: 12px;
    background: rgba(59,130,246,0.08);
    border-left: 4px solid #3b82f6;
}

.step-number {
    font-size: 24px;
    font-weight: 800;
    color: #3b82f6;
    min-width: 40px;
}

/* SPINNER OVERRIDE */
.stSpinner > div {
    border-top-color: #3b82f6 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: transparent !important;
}

/* HIDE STREAMLIT BRANDING */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
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
        st.error(f"❌ Gagal memuat model: {str(e)}")
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
    return "matang" if label == 1 else "belum matang"

# ============================
# HERO SECTION
# ============================
st.markdown("""
<div class='hero-section'>
    <div class='hero-title'>🍅 Deteksi Kematangan Tomat</div>
    <p class='hero-subtitle'>Sistem Klasifikasi Cerdas Menggunakan Hybrid CNN + KNN</p>
</div>
""", unsafe_allow_html=True)

# ============================
# MAIN LAYOUT
# ============================
col1, col2 = st.columns([1.4, 1], gap="large")

# ============================
# LEFT COLUMN - DETECTION
# ============================
with col1:
    # Upload Section
    st.markdown("""
    <div class='card'>
        <div class='card-title'>📤 Upload Gambar Tomat</div>
        <p>Unggah gambar tomat untuk mendeteksi tingkat kematangannya</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Pilih gambar (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded)
        
        # Display Image
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Processing Steps
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🔬 Proses Analisis</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='process-step'>
            <div class='step-number'>1</div>
            <div>
                <b>Preprocessing Gambar</b><br/>
                <small>Resize ke 128×128 px & konversi grayscale</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='process-step'>
            <div class='step-number'>2</div>
            <div>
                <b>Ekstraksi Fitur CNN</b><br/>
                <small>Mengekstrak fitur visual menggunakan CNN</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='process-step'>
            <div class='step-number'>3</div>
            <div>
                <b>Klasifikasi KNN</b><br/>
                <small>Prediksi tingkat kematangan tomat</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Predict Button
        if st.button("🔍 Analisis Kematangan", use_container_width=True):
            with st.spinner("Menganalisis gambar..."):
                arr = preprocess_image(img)
                result = hybrid_predict(arr)
                
                # Display Result
                if result == "matang":
                    st.markdown(f"""
                    <div class='result-success'>
                        <div class='result-title'>✅ Hasil Deteksi</div>
                        <div class='result-value'>TOMAT MATANG</div>
                        <p><b>Status:</b> Siap untuk dipanen atau dikonsumsi</p>
                        <p><b>Karakteristik:</b> Warna merah optimal, tekstur ideal</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-warning'>
                        <div class='result-title'>⚠️ Hasil Deteksi</div>
                        <div class='result-value'>TOMAT BELUM MATANG</div>
                        <p><b>Status:</b> Perlu waktu lebih lama untuk matang</p>
                        <p><b>Saran:</b> Tunggu beberapa hari sebelum panen</p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================
# RIGHT COLUMN - INFORMATION
# ============================
with col2:
    # About System
    st.markdown("""
    <div class='card'>
        <div class='card-title'>ℹ️ Tentang Sistem</div>
        <p>Sistem deteksi kematangan tomat menggunakan pendekatan <b>Hybrid Deep Learning</b> yang menggabungkan kekuatan CNN dan KNN untuk akurasi optimal.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Architecture
    st.markdown("""
    <div class='card'>
        <div class='card-title'>🏗️ Arsitektur Model</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-box'>
        <div class='feature-title'>🧠 CNN (Convolutional Neural Network)</div>
        <p style='font-size: 14px; margin: 0;'>Ekstraksi fitur visual otomatis dari gambar tomat</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-box'>
        <div class='feature-title'>🎯 KNN (K-Nearest Neighbors)</div>
        <p style='font-size: 14px; margin: 0;'>Klasifikasi tingkat kematangan berdasarkan fitur</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Specifications
    st.markdown("""
    <div class='card'>
        <div class='card-title'>⚙️ Spesifikasi Teknis</div>
    </div>
    """, unsafe_allow_html=True)
    
    specs_col1, specs_col2 = st.columns(2)
    
    with specs_col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Input Size</div>
            <div class='metric-value'>128×128</div>
        </div>
        """, unsafe_allow_html=True)
        
    with specs_col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Color Mode</div>
            <div class='metric-value'>Grayscale</div>
        </div>
        """, unsafe_allow_html=True)
    
    # How to Use
    st.markdown("""
    <div class='card'>
        <div class='card-title'>📖 Cara Penggunaan</div>
        <ol style='margin: 0; padding-left: 20px;'>
            <li style='margin: 8px 0;'>Upload gambar tomat dengan format JPG/PNG</li>
            <li style='margin: 8px 0;'>Klik tombol <b>"Analisis Kematangan"</b></li>
            <li style='margin: 8px 0;'>Lihat hasil deteksi tingkat kematangan</li>
            <li style='margin: 8px 0;'>Gunakan hasil untuk menentukan waktu panen</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips
    st.markdown("""
    <div class='card'>
        <div class='card-title'>💡 Tips Foto Terbaik</div>
        <ul style='margin: 0; padding-left: 20px;'>
            <li style='margin: 8px 0;'>Gunakan pencahayaan yang cukup</li>
            <li style='margin: 8px 0;'>Fokus pada satu buah tomat</li>
            <li style='margin: 8px 0;'>Hindari bayangan gelap</li>
            <li style='margin: 8px 0;'>Gunakan background sederhana</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
