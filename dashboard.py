import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models (cached)
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Daniel Fahlevi Bako_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Daniel Fahlevi Bako.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Global UI Setup
# ==========================
st.set_page_config(
    page_title="AI Iris Vision",
    page_icon="🌸",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f8faff;
        }
        h1, h2, h3, h4 {
            color: #1f3b73;
        }
        .stButton>button {
            background-color: #2b6cb0;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #1a4f8b;
        }
        .footer {
            text-align: center;
            color: gray;
            font-size: 0.9em;
            margin-top: 3em;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Sidebar Menu
# ==========================
menu = st.sidebar.radio(
    "📂 Menu Navigasi",
    ["🏠 Homepage", "🧩 Object Detection", "🌺 Classification", "📊 About Models", "⚡ Live Model"],
)

# ==========================
# HOMEPAGE
# ==========================
if menu == "🏠 Homepage":
    st.title("🌸 AI Iris Vision Dashboard")
    st.markdown("#### Mendeteksi dan mengklasifikasi gambar bunga atau objek menggunakan kekuatan AI (YOLO + CNN).")

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/49/Iris_versicolor_3.jpg", use_container_width=True)
    with col2:
        st.markdown(
            """
            ### Apa yang bisa kamu lakukan?
            - 🔍 **Deteksi Objek** menggunakan YOLOv8  
            - 🌼 **Klasifikasi Gambar** bunga iris dengan CNN  
            - ⚡ **Coba Live Model** untuk melihat keduanya bekerja bersama  
            """
        )
        st.markdown("#### Siap mencoba?")
        st.page_link("self", label="⚡ Mulai Live Demo", icon="🚀")

    st.markdown("<div class='footer'>© 2025 AI Iris Vision Dashboard</div>", unsafe_allow_html=True)

# ==========================
# OBJECT DETECTION
# ==========================
elif menu == "🧩 Object Detection":
    st.header("🔍 Object Detection (YOLOv8)")
    st.write("Unggah gambar untuk mendeteksi objek di dalamnya menggunakan model YOLOv8.")

    uploaded_file = st.file_uploader("Unggah gambar (jpg/png):", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        if st.button("🚀 Jalankan Deteksi"):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

# ==========================
# CLASSIFICATION
# ==========================
elif menu == "🌺 Classification":
    st.header("🌼 Image Classification")
    st.write("Unggah gambar bunga iris untuk diklasifikasi ke dalam tiga spesies: *Setosa*, *Versicolor*, atau *Virginica*.")

    uploaded_file = st.file_uploader("Unggah gambar bunga:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        if st.button("🔮 Prediksi"):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            prob = np.max(prediction)

            class_names = ["Setosa", "Versicolor", "Virginica"]
            predicted_class = class_names[class_index]

            st.success(f"**Hasil Prediksi:** {predicted_class} ({prob:.2%})")

            # Fun facts
            facts = {
                "Setosa": "🌸 *Setosa* dikenal memiliki kelopak kecil dan warna lembut, sering tumbuh di dataran rendah.",
                "Versicolor": "🌺 *Versicolor* memiliki warna ungu kebiruan dan batang yang lebih tinggi dari Setosa.",
                "Virginica": "🌼 *Virginica* adalah spesies terbesar, berwarna ungu tua dengan kelopak lebar.",
            }
            st.info(facts[predicted_class])

# ==========================
# ABOUT MODELS
# ==========================
elif menu == "📊 About Models":
    st.header("📘 About the Models")

    st.subheader("🧠 YOLOv8 (Object Detection)")
    st.write(
        """
        YOLOv8 digunakan untuk mendeteksi objek dalam gambar dengan cepat dan efisien.
        Model ini menggunakan *anchor-free detection* dan arsitektur ringan yang cocok untuk real-time prediction.
        """
    )

    st.subheader("🌺 CNN Classifier (Iris Flower Classification)")
    st.write(
        """
        Model CNN digunakan untuk mengenali jenis bunga iris berdasarkan bentuk dan warna kelopaknya.
        Dataset yang digunakan adalah **Iris Dataset** klasik, dengan tiga kelas: Setosa, Versicolor, dan Virginica.
        """
    )

    st.divider()
    st.subheader("💡 Fun Facts tentang Spesies Iris")

    with st.expander("🌸 Iris Setosa"):
        st.write("Kelopak kecil, warna lembut, habitat di dataran rendah dan lembap.")

    with st.expander("🌺 Iris Versicolor"):
        st.write("Campuran warna ungu dan biru, batang tinggi, sering tumbuh di rawa-rawa.")

    with st.expander("🌼 Iris Virginica"):
        st.write("Paling besar di antara ketiganya, warna ungu tua pekat, habitat di tanah asam.")

# ==========================
# LIVE MODEL
# ==========================
elif menu == "⚡ Live Model":
    st.header("⚡ Live Model (Detection + Classification)")
    st.write("Coba keduanya sekaligus — deteksi objek dan klasifikasi bunga dalam satu langkah!")

    uploaded_file = st.file_uploader("Unggah gambar untuk Live Mode:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔍 Hasil Deteksi Objek")
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, use_container_width=True)

        with col2:
            st.subheader("🌸 Hasil Klasifikasi Bunga")
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            prob = np.max(prediction)
            class_names = ["Setosa", "Versicolor", "Virginica"]
            st.success(f"{class_names[class_index]} ({prob:.2%})")
