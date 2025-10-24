import streamlit as st

st.set_page_config(page_title="IrisAI Dashboard", layout="wide")

# ===============================
# CSS UNTUK GAYA NAVBAR
# ===============================
st.markdown("""
<style>
/* Hapus ruang kosong default di atas halaman */
section[data-testid="stHeader"] {
    display: none;
}
div.block-container {
    padding-top: 0rem;
}

/* Navbar elegan dengan gradient */
.navbar {
    background: linear-gradient(90deg, #1e3a8a, #7c3aed);
    color: white;
    padding: 1rem 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15);
}

/* Nama aplikasi di kiri */
.nav-title {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* Tombol menu */
.nav-menu {
    display: flex;
    gap: 1.5rem;
}
.nav-btn {
    background: none;
    border: none;
    color: #e0e7ff;
    font-weight: 500;
    font-size: 1.05rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.nav-btn:hover {
    color: #ffffff;
    transform: translateY(-1px);
}
.active {
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff;
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
# NAVBAR HTML
# ===============================
pages = ["home", "object", "classify", "about", "live"]
icons = {
    "home": "🏠 Home",
    "object": "🧩 Object Detection",
    "classify": "🌸 Classification",
    "about": "📊 About Models",
    "live": "⚡ Live Demo"
}

nav_html = f"""
<div class="navbar">
    <div class="nav-title">🌺 IrisAI</div>
    <div class="nav-menu">
"""
for p in pages:
    active_class = "active" if st.session_state.page == p else ""
    nav_html += f"""
        <form action="" method="get" style="display:inline;">
            <button class="nav-btn {active_class}" name="page" value="{p}" type="submit">{icons[p]}</button>
        </form>
    """
nav_html += "</div></div>"

st.markdown(nav_html, unsafe_allow_html=True)

# ===============================
# LOGIKA NAVIGASI
# ===============================
query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"]

page = st.session_state.page

# ===============================
# HALAMAN UTAMA
# ===============================
if page == "home":
    st.title("🏠 Home Page")
    st.write("Selamat datang di dashboard **IrisAI**.")

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
