import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Styling & Theme
# ==========================
st.markdown("""
    <style>
    /* Latar belakang utama */
    [data-testid="stAppViewContainer"] {
        background-color: #f4f9ff; /* putih kebiruan lembut */
        color: #002b5b; /* teks biru gelap */
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #d0e2ff;
    }

    /* Judul utama */
    h1 {
        color: #0056b3;
        text-align: center;
        font-weight: 800;
        padding-bottom: 0.2em;
    }

    /* Subheader */
    h2, h3 {
        color: #004080;
        font-weight: 600;
    }

    /* Tabs */
    div[data-baseweb="tab"] {
        background-color: #e3f2ff;
        color: #003366 !important;
        border-radius: 8px;
        margin-right: 10px;
        padding: 10px 18px;
        font-weight: 600;
    }
    div[data-baseweb="tab"]:hover {
        background-color: #d0e7ff;
    }

    /* File uploader agar tidak hitam */
    [data-testid="stFileUploader"] {
        background-color: #f0f7ff;
        border: 2px dashed #80bfff;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stFileUploader"] section div {
        color: #004080 !important;
    }

    /* Gambar */
    img {
        border-radius: 12px;
        border: 1px solid #d0e2ff;
    }

    /* Pesan success/info */
    .stSuccess, .stInfo {
        border-radius: 10px;
        padding: 10px;
    }

    </style>
""", unsafe_allow_html=True)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Daniel Fahlevi Bako_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Daniel Fahlevi Bako.h5")  # Model klasifikasi
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
        result_img = results[0].plot()  # hasil deteksi
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

with tab2:
    st.subheader("🌼 Klasifikasi Gambar")
    uploaded_file = st.file_uploader("Unggah gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="klasifikasi")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        class_names = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

        st.success(f"### 🌷 Hasil Prediksi: {class_names[class_index]}")
        st.info(f"Probabilitas: {np.max(prediction):.2f}")
