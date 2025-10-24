import streamlit as st

st.set_page_config(page_title="IrisAI Dashboard", layout="wide")

# ===============================
# Helper: read current page from URL query param (safe)
# ===============================
def get_current_page():
    params = st.experimental_get_query_params()
    page = params.get("page", ["home"])[0]
    if page not in ["home", "object", "classify", "about", "live"]:
        return "home"
    return page

current_page = get_current_page()

# ===============================
# STYLES: remove top white gap + navbar gradient + link styling
# ===============================
st.markdown(
    """
    <style>
    /* remove Streamlit top header and top padding */
    section[data-testid="stHeader"] {display: none;}
    div.block-container {padding-top: 0rem;}

    /* navbar container */
    .navbar {
        width: 100%;
        box-sizing: border-box;
        padding: 14px 36px;
        displa
