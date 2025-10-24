import streamlit as st

st.set_page_config(page_title="IrisAI Dashboard", layout="wide")

# ===============================
# Helper: ambil halaman aktif
# ===============================
def get_current_page():
    params = st.experimental_get_query_params()
    page = params.get("page", ["home"])[0]
    if page not in ["home", "object", "classify", "about", "live"]:
        return "home"
    return page

current_page = get_current_page()

# ===============================
# CSS STYLING — versi aman (tertutup sempurna)
# ===============================
st.markdown("""
<style>
/* Hilangkan header Streamlit dan padding default */
section[data-testid="stHeader"] {display: none;}
div.block-container {padding-top: 0rem;}

/* Navbar */
.navbar {
    width: 100%;
    padding: 14px 36px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    p
