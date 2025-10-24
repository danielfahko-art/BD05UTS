import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import io  # Untuk download gambar

# Optional: Jika pakai streamlit-option-menu
try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

# ==========================
# Custom CSS untuk Tema Gelap Modern (Background Gelap, Font Terang)
# ==========================
st.markdown("""
    <style>
    .main { background-color: #121212; color: #F0F0F0; }  /* Background gelap, font terang */
    .stTitle { color: #FFFFFF; font-family: 'Arial', sans-serif; font-weight: bold; }  /* Judul putih */
    .stMarkdown { font-size: 16px; color: #F0F0F0; }  /* Teks abu terang */
    .stButton>button { background: linear-gradient(45deg, #8A2BE2, #00CED1); color: white; border-radius: 10px; border: none; padding: 10px 20px; font-weight: bold; }
    .stButton>button:hover { background: linear-gradient(45deg, #00CED1, #8A2BE2); }
    .stProgress > div > div > div > div { background: linear-gradient(45deg, #8A2BE2, #00CED1); }
    .stSidebar { background-color: #1E1E1E; color: #F0F0F0; }  /* Sidebar gelap */
    .stRadio > div { color: #F0F0F0; }
    .model-box { border: 2px solid #8A2BE2; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #1E1E1E; }  /* Kotak model */
    </style>
""", unsafe_allow_html=True)

# ==========================
# Mapping Kelas CNN ke Nama Bunga (Sesuai Dataset Iris)
# ==========================
CLASS_NAMES = {
    0: "Iris-setosa",  # Jika kelas dimulai dari 0, adjust sesuai model Anda
    1: "Iris-setosa",
    2: "Iris-versicolor",
    3: "Iris-virginica"
}

# ==========================
# Load Models (dengan error handling yang lebih baik)
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/Daniel Fahlevi Bako_Laporan 4.pt"
    cnn_path = "model/Daniel Fahlevi Bako.h5"

    yolo_model = None
    cnn_model = None

    try:
        yolo_model = YOLO(yolo_path)
        st.success("✅ YOLO model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat YOLO model: {e}. Pastikan file ada di 'model/'.")

    try:
        cnn_model = tf.keras.models.load_model(cnn_path)
        st.success("✅ CNN model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat CNN model: {e}. Pastikan file ada di 'model/'.")

    return yolo_model, cnn_model

yolo_model, classifier = load_models()

# ==========================
# Helper: Preprocessing gambar
# ==========================
def preprocess_for_model(pil_img: Image.Image, model):
    default_target = (224, 224, 3)
    if model is None:
        img_resized = pil_img.resize((default_target[0], default_target[1]))
        arr = image.img_to_array(img_resized).astype("float32") / 255.0
        return np.expand_dims(arr, axis=0)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    shape = [s for s in input_shape if s is not None]
    if len(shape) < 3:
        shape = default_target

    if len(shape) == 3 and shape[-1] in (1, 3):
        target_h, target_w, channels = shape[0], shape[1], shape[2]
        channels_first = False
    else:
        channels_first = True
        channels, target_h, target_w = shape[0], shape[1], shape[2]

    img_mode = "RGB" if channels == 3 else "L"
    img_proc = pil_img.convert(img_mode) if pil_img.mode != img_mode else pil_img.copy()
    img_resized = img_proc.resize((target_w, target_h))
    arr = image.img_to_array(img_resized)

    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))
    arr = arr.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

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
            "container": {"padding": "0!important", "background-color": "#1E1E1E"},
            "icon": {"color": #8A2BE2", "font-size": "18px"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#333333"},
            "nav-link-selected": {"background-color": "#8A2BE2", "color": "white"},
        },
    )
else:
    st.sidebar.title("🌺 Navigasi Iris AI")
    selected = st.sidebar.radio("Pilih Halaman", ["Home", "Object Detection", "Classification", "About Models", "Comparison"])

# ==========================
# HOME
# ==========================
if selected == "Home":
    st.title("🌺 Selamat Datang di Iris AI Dashboard")
    st.markdown("""
    **Iris AI** adalah platform sederhana untuk eksplorasi AI:
    - 🧩 **Object Detection**: Deteksi objek menggunakan YOLO.
    - 🌸 **Image Classification**: Klasifikasi gambar dengan model CNN.
    
    Unggah gambar dan jelajahi fitur-fitur kami!
    """)
    st.image("https://via.placeholder.com/800x200/8A2BE2/FFFFFF?text=Iris+AI+Dashboard", use_container_width=True)  # Placeholder image dengan warna baru

# ==========================
# OBJECT DETECTION
# ==========================
elif selected == "Object Detection":
    st.title("🧩 Object Detection (YOLO)")
    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi objek", type=["jpg", "jpeg", "png"], key="deteksi")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar Asli")
            st.image(img, caption="Gambar yang Diupload", use_container_width=True)
        
        if yolo_model is None:
            st.error("❌ YOLO model belum dimuat. Periksa folder 'model/'.")
        else:
            with st.spinner("🔍 Mendeteksi objek..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Simulasi progress
                    progress_bar.progress(i + 1)
                try:
                    results = yolo_model(img)
                    result_img = results[0].plot()
                    with col2:
                        st.subheader("Hasil Deteksi")
                        st.image(result_img, caption="Objek Terdeteksi", use_container_width=True)
                    
                    # List objek terdeteksi
                    boxes = results[0].boxes
                    if boxes is not None and hasattr(boxes, "cls") and len(boxes.cls) > 0:
                        detected_classes = [results[0].names[int(c)] for c in boxes.cls]
                        st.success(f"✅ Objek Terdeteksi: {', '.join(set(detected_classes))}")
                        # Download hasil
                        buf = io.BytesIO()
                        Image.fromarray(result_img).save(buf, format="PNG")
                        st.download_button("📥 Download Hasil Deteksi", buf.getvalue(), "hasil_deteksi.png", "image/png")
                    else:
                        st.info("ℹ️ Tidak ada objek terdeteksi.")
                except Exception as e:
                    st.error(f"❌ Error saat deteksi: {e}")
    
    if st.button("🔄 Reset"):
        st.experimental_rerun()

# ==========================
# CLASSIFICATION
# ==========================
elif selected == "Classification":
    st.title("🌸 Image Classification (CNN)")
    uploaded_file = st.file_uploader("Unggah gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="klasifikasi")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar Asli")
            st.image(img, caption="Gambar yang Diupload", use_container_width=True)
        
        if classifier is None:
            st.error("❌ Model CNN belum dimuat. Periksa folder 'model/'.")
        else:
            with st.spinner("🔮 Mengklasifikasi gambar..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                try:
                    arr = preprocess_for_model(img, classifier)
                    start = time.time()
                    prediction = classifier.predict(arr)
                    elapsed = time.time() - start
                    
                    pred_vec = np.asarray(prediction)
                    if pred_vec.ndim == 2:
                        class_index = int(np.argmax(pred_vec[0]))
                        confidence = float(np.max(pred_vec[0]))
                    elif pred_vec.ndim == 1:
                        class_index = int(np.argmax(pred_vec))
                        confidence = float(np.max(pred_vec))
                    else:
                        class_index, confidence = 0, 0.0
                    
                    # Mapping ke nama bunga
                    flower_name = CLASS_NAMES.get(class_index, f"Kelas {class_index}")
                    
                    with col2:
                        st.subheader("Hasil Klasifikasi")
                        st.metric("Nama Bunga", flower_name, f"{confidence:.2%} Confidence")
                        st.write(f"⏱️ Waktu Proses: {elapsed:.2f} detik")
                except Exception as e:
                    st.error(f"❌ Error saat klasifikasi: {e}")
    
    if st.button("🔄 Reset"):
        st.experimental_rerun()

# ==========================
# ABOUT MODELS
# ==========================
elif selected == "About Models":
    st.title("ℹ️ Tentang Model")
    st.markdown("Halaman ini menjelaskan model yang digunakan dalam dashboard Iris AI.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="model-box">', unsafe_allow_html=True)
        st.subheader("🧩 Model yang Digunakan: YOLO")
        st.markdown("""
        **YOLO (You Only Look Once)** adalah model deteksi objek real-time dari Ultralytics. 
        - **Fungsi**: Mendeteksi dan mengklasifikasikan objek dalam gambar dengan bounding box.
        - **Keunggulan**: Cepat, akurat untuk aplikasi real-time, dan mendukung multiple objek.
        - **Penggunaan di Dashboard**: Digunakan untuk halaman Object Detection.
        
        [Pelajari lebih lanjut](https://docs.ultralytics.com/).
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="model-box">', unsafe_allow_html=True)
        st.subheader("🌸 Model yang Digunakan: CNN")
        st.markdown("""
        **CNN (Convolutional Neural Network)** adalah model klasifikasi gambar berbasis Keras/TensorFlow. 
        - **Fungsi**: Menganalisis fitur gambar untuk mengkategorikan ke dalam kelas (misalnya, nama bunga).
        - **Keunggulan**: Efisien untuk tugas klasifikasi, dapat dilatih pada dataset besar.
        - **Penggunaan di Dashboard**: Digunakan untuk halaman Classification, memprediksi nama bunga seperti Iris-setosa.
        
        [Pelajari lebih lanjut](https://www.tensorflow.org/).
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# COMPARISON
# ==========================
elif selected == "Comparison":
    st.title("📊 Perbandingan Model")
    uploaded_file = st.file_uploader("Unggah gambar untuk perbandingan", type=["jpg", "jpeg", "png"], key="comparison")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)
        
        results = {}
        if yolo_model:
            try:
                yolo_results = yolo_model(img)
                boxes = yolo_results[0].boxes
                results["YOLO"] = ", ".join(set([yolo_results[0].names[int(c)] for c in boxes.cls])) if boxes and len(boxes.cls) > 0 else "Tidak ada objek"
            except:
                results["YOLO"] = "Error"
        else:
            results["YOLO"] = "Model tidak dimuat"
        
        if classifier:
            try:
                arr = preprocess_for_model(img, classifier)
                pred = classifier.predict(arr)
                class_idx = np.argmax(pred)
                flower_name = CLASS_NAMES.get(class_idx, f"Kelas {class_idx}")
                results["CNN"] = f"{flower_name} ({np.max(pred):.2%})"
            except:
                results["CNN"] = "Error"
        else:
            results["CNN"] = "Model tidak dimuat"
        
        st.table({"Model": list(results.keys()), "Hasil": list(results.values())})

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("**Iris AI Dashboard** - Dibuat dengan ❤️ menggunakan Streamlit. Tema gelap oleh AI Assistant.")
