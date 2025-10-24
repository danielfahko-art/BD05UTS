import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
from streamlit_option_menu import option_menu

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
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Iris AI 🌺",
    layout="wide",
    page_icon="🌸",
    initial_sidebar_state="collapsed",
)

# ==========================
# NAVIGATION BAR
# ==========================
selected = option_menu(
    menu_title=None,
    options=["Home", "Object Detection", "Classification", "About Models", "Comparison"],
    icons=["house", "bounding-box", "image", "info-circle", "bar-chart"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#EAF4FC"},
        "icon": {"color": "#1E90FF", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#B3D9FF",
        },
        "nav-link-selected": {"background-color": "#1E90FF", "color": "white"},
    },
)

# ==========================
# TAB 1 — HOME
# ==========================
if selected == "Home":
    st.title("🌺 Selamat Datang di Iris AI Dashboard")
    st.write(
        """
        **Iris AI** adalah platform sederhana yang memanfaatkan dua model kecerdasan buatan:
        - Model **YOLO** untuk *object detection*  
        - Model **CNN (Keras)** untuk *image classification*  

        Pilih menu di atas untuk mulai menggunakan fitur deteksi atau klasifikasi.  
        """
    )

# ==========================
# TAB 2 — OBJECT DETECTION
# ==========================
elif selected == "Object Detection":
    st.title("🧩 Object Detection (YOLO)")
    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi objek", type=["jpg", "jpeg", "png"], key="deteksi")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔍 Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

# ==========================
# TAB 3 — CLASSIFICATION
# ==========================
elif selected == "Classification":
    st.title("🌸 Image Classification (CNN)")
    uploaded_file = st.file_uploader("Unggah gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="klasifikasi")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        with st.spinner("🔮 Sedang memprediksi..."):
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        st.success(f"### 🌺 Hasil Prediksi: {class_index}")
        st.write(f"**Probabilitas:** {confidence:.2%}")

# ==========================
# TAB 4 — ABOUT MODELS
# ==========================
elif selected == "About Models":
    st.title("🧠 About Models")
    st.write("Berikut adalah perbandingan arsitektur antara kedua model yang digunakan:")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧩 YOLO Model (Object Detection)")
        st.markdown("""
        - Arsitektur: YOLOv8  
        - Tujuan: Deteksi objek dalam gambar  
        - Ciri:  
            - Menggunakan *anchor-free detection head*  
            - Cepat dan efisien untuk real-time inference  
            - Output berupa koordinat bounding box dan label objek
        """)

    with col2:
        st.subheader("🌸 CNN Model (Classification)")
        st.markdown("""
        - Arsitektur: Convolutional Neural Network  
        - Tujuan: Klasifikasi citra menjadi beberapa kelas  
        - Ciri:  
            - Menggunakan *convolutional*, *pooling*, dan *dense layers*  
            - Fokus pada fitur visual global  
            - Output berupa probabilitas kelas
        """)

# ==========================
# TAB 5 — COMPARISON
# ==========================
elif selected == "Comparison":
    st.title("📊 Model Comparison")
    st.write("Bandingkan performa kedua model dalam satu tampilan:")

    # Dummy metrics comparison example (ganti sesuai datamu)
    comparison_data = {
        "Model": ["YOLOv8", "CNN"],
        "Akurasi": [0.93, 0.89],
        "Waktu Prediksi (detik)": [0.15, 0.35],
        "Ukuran Model (MB)": [45, 90],
    }

    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

    st.bar_chart(df.set_index("Model")[["Akurasi", "Waktu Prediksi (detik)"]])
