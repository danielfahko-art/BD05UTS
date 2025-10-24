import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Styling & Theme (Revised)
# ==========================
st.markdown("""
    <style>
    /* ===== Warna dasar aplikasi ===== */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff; /* putih bersih */
        color: #1a1a1a; /* teks gelap agar mudah dibaca */
    }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background-color: #f1f5ff; /* biru muda lembut */
        border-right: 1px solid #cdd9f7;
    }

    /* ===== Judul ===== */
    h1 {
        color: #0056d6;
        text-align: center;
        font-weight: 800;
        margin-top: 0.3em;
    }

    /* ===== Subjudul ===== */
    h2, h3, h4 {
        color: #003a91;
        font-weight: 600;
    }

    /* ===== Tabs ===== */
    div[data-baseweb="tab-list"] {
        border-bottom: 2px solid #cde0ff;
    }
    div[data-baseweb="tab"] {
        background-color: #e8f0ff;
        color: #003366 !important;
        border-radius: 8px 8px 0 0;
        margin-right: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 16px;
    }
    div[data-baseweb="tab"]:hover {
        background-color: #d5e5ff;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #0056d6;
        color: white !important;
    }

    /* ===== File uploader ===== */
    [data-testid="stFileUploader"] {
        background-color: #f5f8ff;
        border: 2px dashed #80aaff;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stFileUploader"] section div {
        color: #002b5b !important;
    }

    /* ===== Gambar ===== */
    img {
        border-radius: 12px;
        border: 1px solid #d0e2ff;
    }

    /* ===== Pesan sukses/info ===== */
    .stSuccess, .stInfo {
        border-radius: 10px;
        padding: 10px;
        font-weight: 500;
    }

    /* ===== Tombol ===== */
    button {
        background-color: #0056d6 !important;
        color: white !important;
        border-radius: 6px !important;
    }

    </style>
""", unsafe_allow_html=True)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Daniel Fahlevi Bako_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Daniel Fahlevi Bako.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🌺 Iris AI Dashboard")

tab1, tab2 = st.tabs(["🧩 Deteksi Objek", "🌸 Klasifikasi Gambar"])

with tab1:
    st.subheader("🧠 Deteksi Objek (YOLO)")
    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi objek", type=["jpg", "jpeg", "png"], key="deteksi")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

with tab2:
    st.subheader("🌼 Klasifikasi Gambar")
    uploaded_file = st.file_uploader("Unggah gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="klasifikasi")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        class_names = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

        st.success(f"### 🌷 Hasil Prediksi: {class_names[class_index]}")
        st.info(f"Probabilitas: {np.max(prediction):.2f}")
