import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

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
st.set_page_config(page_title="Iris AI 🌺", layout="wide")
st.title("🌺 Iris AI Dashboard")

tab1, tab2 = st.tabs(["🧩 Deteksi Objek", "🌸 Klasifikasi"])

# ==========================
# TAB 1 — DETEKSI OBJEK
# ==========================
with tab1:
    st.subheader("🧩 Deteksi Objek (YOLO)")
    uploaded_file = st.file_uploader("Unggah Gambar untuk Deteksi Objek", type=["jpg", "jpeg", "png"], key="deteksi")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        # Proses deteksi
        with st.spinner("🔍 Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()  # hasil deteksi (gambar dengan bounding box)
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

# ==========================
# TAB 2 — KLASIFIKASI GAMBAR
# ==========================
with tab2:
    st.subheader("🌸 Klasifikasi Gambar (CNN)")
    uploaded_file = st.file_uploader("Unggah Gambar untuk Klasifikasi", type=["jpg", "jpeg", "png"], key="klasifikasi")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        with st.spinner("🔮 Sedang memprediksi..."):
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        st.success(f"### 🌸 Hasil Prediksi: {class_index}")
        st.write(f"**Probabilitas:** {confidence:.2%}")
