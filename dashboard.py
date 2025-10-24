# dashboard.py
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# optional: jika pakai streamlit-option-menu
try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

# ==========================
# Load Models (dengan error handling)
# ==========================
@st.cache_resource
def load_models():
    # Ganti path file model sesuai lokasimu
    yolo_path = "model/Daniel Fahlevi Bako_Laporan 4.pt"
    cnn_path = "model/Daniel Fahlevi Bako.h5"

    yolo_model = None
    cnn_model = None

    # Load YOLO (ultralytics)
    try:
        yolo_model = YOLO(yolo_path)
    except Exception as e:
        # kita tidak raise agar app masih jalan meski YOLO gagal
        st.error(f"Gagal load YOLO model: {e}")

    # Load CNN (Keras)
    try:
        cnn_model = tf.keras.models.load_model(cnn_path)
    except Exception as e:
        st.error(f"Gagal load CNN model (.h5): {e}")

    return yolo_model, cnn_model

yolo_model, classifier = load_models()

# ==========================
# Helper: preprocessing robust sesuai input_shape model
# ==========================
def preprocess_for_model(pil_img: Image.Image, model):
    """
    Sesuaikan ukuran & channel sesuai model.input_shape.
    Mengembalikan numpy array shape (1, H, W, C) atau (1, C, H, W) sesuai model.
    """
    # default jika model None
    default_target = (224, 224, 3)

    if model is None:
        # fallback
        img_resized = pil_img.resize((default_target[0], default_target[1]))
        arr = image.img_to_array(img_resized).astype("float32") / 255.0
        return np.expand_dims(arr, axis=0)

    # Ambil input shape model (bisa tuple atau list)
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    # Buat list tanpa None (buang batch dim)
    shape = list(input_shape)
    if shape and shape[0] is None:
        shape = shape[1:]

    # Jika shape memiliki None, isi dengan default
    for i in range(len(shape)):
        if shape[i] is None:
            if i == 0:
                shape[i] = default_target[2]  # channel
            else:
                shape[i] = default_target[i-1]  # H/W

    # Setelah normalisasi, shape bisa jadi [H,W,C] (channels_last) atau [C,H,W] (channels_first).
    if len(shape) == 3:
        # Heuristik: jika last dim 1 atau 3 => channels_last
        if shape[-1] in (1, 3):
            target_h, target_w, channels = int(shape[0]), int(shape[1]), int(shape[2])
            channels_first = False
        else:
            # anggap channels_first: [C,H,W]
            channels_first = True
            channels = int(shape[0])
            target_h = int(shape[1])
            target_w = int(shape[2])
    else:
        # fallback
        target_h, target_w, channels = default_target
        channels_first = False

    # Resize gambar
    img_mode = "RGB" if channels == 3 else "L"
    if pil_img.mode != img_mode:
        img_proc = pil_img.convert(img_mode)
    else:
        img_proc = pil_img.copy()

    img_resized = img_proc.resize((target_w, target_h))
    arr = image.img_to_array(img_resized)  # shape (H,W,C) channels_last

    # Jika model mengharapkan channels_first, transpose
    if channels_first:
        # arr shape (H,W,C) -> (C,H,W)
        arr = np.transpose(arr, (2, 0, 1))

    # Convert dtype & scale
    arr = arr.astype("float32") / 255.0

    # Expand batch dim
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, C) atau (1, C, H, W)
    return arr

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Iris AI 🌺", layout="wide", page_icon="🌸")

# ==========================
# NAVIGATION
# ==========================
if option_menu is not None:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Object Detection", "Classification", "About Models", "Comparison"],
        icons=["house", "bounding-box", "image", "info-circle", "bar-chart"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#EAF4FC"},
            "icon": {"color": "#1E90FF", "font-size": "18px"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#B3D9FF"},
            "nav-link-selected": {"background-color": "#1E90FF", "color": "white"},
        },
    )
else:
    # fallback: st.selectbox di sidebar
    selected = st.sidebar.selectbox("Menu", ["Home", "Object Detection", "Classification", "About Models", "Comparison"])

# ==========================
# HOME
# ==========================
if selected == "Home":
    st.title("🌺 Selamat Datang di Iris AI Dashboard")
    st.write("""
    **Iris AI**: platform sederhana untuk menampilkan:
    - Deteksi objek menggunakan YOLO
    - Klasifikasi gambar menggunakan model CNN (Keras)
    """)

# ==========================
# OBJECT DETECTION
# ==========================
elif selected == "Object Detection":
    st.title("🧩 Object Detection (YOLO)")
    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi objek", type=["jpg", "jpeg", "png"], key="deteksi")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        if yolo_model is None:
            st.error("YOLO model belum dimuat.")
        else:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                try:
                    results = yolo_model(img)
                    result_img = results[0].plot()
                    st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

                    # List detected classes safe
                    boxes = results[0].boxes
                    if boxes is not None and hasattr(boxes, "cls") and len(boxes.cls) > 0:
                        detected_classes = [results[0].names[int(c)] for c in boxes.cls]
                        st.write(f"**Objek Terdeteksi:** {', '.join(set(detected_classes))}")
                    else:
                        st.write("Tidak ada objek terdeteksi.")
                except Exception as e:
                    st.error(f"Terjadi error saat deteksi YOLO: {e}")

# ==========================
# CLASSIFICATION
# ==========================
elif selected == "Classification":
    st.title("🌸 Image Classification (CNN)")
    uploaded_file = st.file_uploader("Unggah gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="klasifikasi")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        if classifier is None:
            st.error("Model CNN belum dimuat.")
        else:
            try:
                with st.spinner("🔮 Sedang memproses klasifikasi..."):
                    arr = preprocess_for_model(img, classifier)
                    start = time.time()
                    prediction = classifier.predict(arr)
                    elapsed = time.time() - start

                # Kalau output adalah vektor probabilitas
                if prediction is None:
                    st.error("Model mengembalikan None sebagai prediksi.")
                else:
                    # Jika prediksi multi-output, pilih axis yang tepat
                    pred_vec = np.asarray(prediction)
                    # asumsi: (1, num_classes)
                    if pred_vec.ndim == 2:
                        class_index = int(np.argmax(pred_vec[0]))
                        confidence = float(np.max(pred_vec[0]))
                    elif pred_vec.ndim == 1:
                        class_index = int(np.argmax(pred_vec))
                        confidence = float(np.max(pred_vec))
                    else:
                        # fallback
                        class_index = 0
                        confidence =
