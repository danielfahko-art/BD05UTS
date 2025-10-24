import streamlit as st

st.set_page_config(page_title="IrisAI Dashboard", layout="wide")

# ===============================
# CSS UNTUK NAVBAR
# ===============================
st.markdown("""
<style>
/* Navbar Container */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #ffffff;
    padding: 1rem 2rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    position: sticky;
    top: 0;
    z-index: 999;
}

/* Logo kiri */
.navbar-left {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2563eb;
    font-family: 'Poppins', sans-serif;
}

/* Menu kanan */
.navbar-right {
    display: flex;
    gap: 1.5rem;
}

.nav-item {
    text-decoration: none;
    color: #1f3b73;
    font-size: 1.05rem;
    font-weight: 500;
    cursor: pointer;
    transition: color 0.2s ease, border-bottom 0.2s ease;
    padding-bottom: 2px;
}

.nav-item:hover {
    color: #2563eb;
}

.active {
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# NAVIGATION LOGIC
# ===============================
# Simpan halaman aktif di session_state
if "page" not in st.session_state:
    st.session_state.page = "home"

# Fungsi ganti halaman
def switch_page(page_name):
    st.session_state.page = page_name

# ===============================
# RENDER NAVBAR
# ===============================
navbar_html = f"""
<div class="navbar">
    <div class="navbar-left">IrisAI</div>
    <div class="navbar-right">
        <span class="nav-item {'active' if st.session_state.page=='home' else ''}" onClick="window.location.href='?page=home'">Home</span>
        <span class="nav-item {'active' if st.session_state.page=='object' else ''}" onClick="window.location.href='?page=object'">Object Detection</span>
        <span class="nav-item {'active' if st.session_state.page=='classify' else ''}" onClick="window.location.href='?page=classify'">Classification</span>
        <span class="nav-item {'active' if st.session_state.page=='about' else ''}" onClick="window.location.href='?page=about'">About Models</span>
        <span class="nav-item {'active' if st.session_state.page=='live' else ''}" onClick="window.location.href='?page=live'">Live Demo</span>
    </div>
</div>
"""
st.markdown(navbar_html, unsafe_allow_html=True)

# ===============================
# HALAMAN UTAMA (CONTENT)
# ===============================
query_params = st.query_params
if "page" in query_params:
    page = query_params["page"][0]
    st.session_state.page = page

page = st.session_state.page

if page == "home":
    st.title("🏠 Welcome to IrisAI Dashboard")
    st.write("This is your Home Page.")

elif page == "object":
    st.title("🧩 Object Detection")
    st.write("Here you can upload an image and detect objects.")

elif page == "classify":
    st.title("🌸 Image Classification")
    st.write("Classify iris flower images into their correct species.")

elif page == "about":
    st.title("ℹ️ About Models")
    st.write("Learn about the technology and architecture behind each model.")

elif page == "live":
    st.title("🚀 Live Demo")
    st.write("Try the combined detection and classification system live here.")
