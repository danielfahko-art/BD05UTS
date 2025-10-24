import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Konfigurasi Tema
# ==========================
st.markdown("""
    <style>
    /* Background dan teks utama */
    [data-testid="stAppViewContainer"] {
        background-color: #f8fbff;
        color: #0d1b2a;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #e6f0ff;
    }

    /* Header judul */
    h1 {
        color: #0056b3;
        text-align: center;
        font-weight: 700;
    }

    /* Tab styling */
    div[data-baseweb="tab"] {
        background-color: #cfe2ff;
        color: #003366;
        border-radius: 10px;
        margin-right: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    div[data-baseweb="tab"]:hover {
        background-color: #b6d4fe;
    }

    /* Tombol unggah */
    [data-testid="stFileUploader"] {
        background-color: #e6f0ff;
        border-radius: 10px;
        padding: 10px;
    }

    /* Gambar */
    img {
