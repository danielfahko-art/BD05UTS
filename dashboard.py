import streamlit as st

st.set_page_config(page_title="IrisAI Dashboard", layout="wide")

# ===============================
# CSS UNTUK GAYA NAVBAR
# ===============================
st.markdown("""
<style>
div[data-testid="column"] {
    display: flex;
    align-items: center;
}
.navbar {
    background-color: #ffffff;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    padding: 1rem 2rem;
    margin-bottom: 1.5rem;
    position: sticky;
    top: 0;
    z-index: 999;
}
.nav-button {
    background: none;
    border: none;
    color: #1f3b73;
    font-weight: 500;
    font-size: 1.05rem;
    cursor: pointer;
    transition: color 0.2s ease, border-bottom 0.2s ease;
}
.nav-button:hover {
    color: #2563eb;
}
.active {
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# STATE HALAMAN
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# Fungsi untuk ganti halaman
def set_page(page_name):
    st.session_state.page = page_name

# ===============================
# NAVBAR STREAMLIT
# ===============================
st.markdown('<div class="navbar">', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1, 1.5, 1.8, 1.5, 1.5])

with col1:
    st.markdown("<h3 style='color:#2563eb; margin:0;'>IrisAI</h3>", unsafe_allow_html=True)
with col2:
    if st.button("🏠 Home", key="home", use_container_width=True):
        set_page("home")
with col3:
    if st.button("🧩 Object Detection", key="object", use_container_width=True):
        set_page("object")
with col4:
    if st.button("🌸 Classification", key="classify", use_container_width=True):
        set_page("classify")
with col5:
    if st.button("📊 About Models", key="about", use_container_width=True):
        set_page("about")
with col6:
    if st.button("⚡ Live Demo", key="live", use_container_width=True):
        set_page("live")

st.markdown('</div>', unsafe_allow_html=True)

# Tambahkan efek aktif secara visual
st.markdown(
    f"""
    <style>
    button[kind="secondary"] {{
        color: #1f3b73 !important;
        border: none !important;
        background: none !important;
    }}
    div.stButton > button:first-child {{
        border-radius: 0px !important;
    }}
    button[key="{st.session_state.page}"] {{
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# HALAMAN UTAMA
# ===============================
page = st.session_state.page

if page == "home":
    st.title("🏠 Home Page")
    st.write("Selamat datang di dashboard IrisAI.")

elif page == "object":
    st.title("🧩 Object Detection")
    st.write("Unggah gambar dan deteksi objek di dalamnya.")

elif page == "classify":
    st.title("🌸 Classification")
    st.write("Klasifikasikan gambar bunga iris ke dalam tiga spesies.")

elif page == "about":
    st.title("📊 About Models")
    st.write("Pelajari model dan fakta menarik tentang bunga iris.")

elif page == "live":
    st.title("⚡ Live Demo")
    st.write("Coba deteksi dan klasifikasi secara bersamaan.")
