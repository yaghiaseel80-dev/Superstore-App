import streamlit as st
from src.style import apply_style, footer

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Superstore KDD Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_style()

# ── HIDE ALL STREAMLIT SYSTEM UI ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebarNav"]   { display: none !important; }
[data-testid="stToolbar"]      { display: none !important; }
#MainMenu                      { display: none !important; }
header[data-testid="stHeader"] { visibility: hidden !important; height: 0 !important; }
footer                         { visibility: hidden !important; }
.main .block-container         { padding-top: 1.2rem !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Data Collection"
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

dark = st.session_state["dark_mode"]

# ── DARK MODE CSS ─────────────────────────────────────────────────────────────
if dark:
    st.markdown("""
    <style>
    /* Base page */
    .stApp, .main, .block-container,
    html, body { background-color: #0f1b2d !important; }

    /* ALL text */
    * { color: #e0e7ef !important; }

    /* Keep branded colors */
    .page-title  { color: #17a589 !important; }
    .page-subtitle { color: #a8d8d0 !important; }
    .section-header { color: #a8d8d0 !important; border-left-color: #17a589 !important; }
    i.bi { color: #17a589 !important; }

    /* ALL divs with inline backgrounds — make transparent so page bg shows */
    div { background-color: transparent !important; }

    /* Then re-apply dark bg to specific containers */
    .stApp, .main, .block-container { background-color: #0f1b2d !important; }
    .kpi-card { background-color: #1e2d40 !important; border-top-color: #17a589 !important; }
    [data-testid="stExpander"] { background-color: #1e2d40 !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #1e2d40 !important; border-color: #17a589 !important; }
    [data-testid="stAlert"] { background-color: #1e2d40 !important; }

    /* Streamlit widgets */
    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    [data-baseweb="datepicker"] { background-color: #1e2d40 !important; }

    /* Calendar popup */
    [data-baseweb="calendar"],
    [data-baseweb="calendar"] * { background-color: #1e2d40 !important; }

    /* Buttons */
    button { color: #ffffff !important; }
    div.nav-container button[data-testid="baseButton-secondary"] {
        background: #1e2d40 !important;
    }

    /* Dropdown popups */
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [role="listbox"],
    [role="listbox"] *,
    [role="option"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #1e2d40 !important;
        color: #e0e7ef !important;
    }
    li[role="option"]:hover {
        background-color: #17a589 !important;
        color: #ffffff !important;
    }

    /* Multiselect dropdown */
    [data-baseweb="menu"],
    [data-baseweb="menu"] * {
        background-color: #1e2d40 !important;
        color: #e0e7ef !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] { color: #a8d8d0 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #17a589 !important; }

    /* Footer & HR */
    .footer-bar { border-top-color: #2c3e50 !important; }
    hr { border-color: #2c3e50 !important; }
    </style>
    """, unsafe_allow_html=True)

# ── BRANDED HEADER ────────────────────────────────────────────────────────────
col_logo, col_title, col_toggle = st.columns([1, 7, 1])
with col_logo:
    st.image("assets/lau-logo.jpg", width=100)
with col_title:
    title_color = "#17a589"
    sub_color   = "#a8d8d0" if dark else "#7f8c8d"
    st.markdown(f"""
    <div style="padding-top:10px;">
        <span style="font-size:26px; font-weight:800; color:{title_color}; letter-spacing:-0.5px;">
            Superstore KDD Analytics Dashboard
        </span><br>
        <span style="font-size:13px; color:{sub_color}; font-weight:400;">
            DAN614 – Advanced Data Visualization &nbsp;|&nbsp;
            MSDA &nbsp;|&nbsp;
            LAU Adnan Kassar School of Business &nbsp;|&nbsp;
            Aseel Yaghi
        </span>
    </div>
    """, unsafe_allow_html=True)
with col_toggle:
    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    new_dark = st.toggle("🌙", value=dark, key="dark_mode_toggle", help="Toggle Dark / Light mode")
    if new_dark != dark:
        st.session_state["dark_mode"] = new_dark
        st.rerun()

st.markdown(
    "<hr style='border:none; border-top:2px solid #17a589; margin:12px 0 10px 0;'>",
    unsafe_allow_html=True
)

# ── PAGE DEFINITIONS ──────────────────────────────────────────────────────────
pages = [
    ("☁",  "Data Collection"),
    ("▼",  "Data Processing & Cleaning"),
    ("▦",  "Descriptive Analysis"),
    ("⬡",  "Machine Learning"),
    ("◎",  "Predict"),
    ("ℹ",  "About"),
]

current = st.session_state["current_page"]

# ── NAV BUTTON STYLES ─────────────────────────────────────────────────────────
st.markdown("""
<style>
div.nav-container > div[data-testid="stHorizontalBlock"] {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    padding: 6px 8px !important;
    gap: 6px !important;
    margin-bottom: 20px;
}
div.nav-container button {
    border-radius: 8px !important;
    border: none !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    height: 46px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
div.nav-container button[data-testid="baseButton-secondary"] {
    background: #ffffff !important;
    color: #2c3e50 !important;
    box-shadow: none !important;
}
div.nav-container button[data-testid="baseButton-secondary"]:hover {
    background: #e8f8f5 !important;
    color: #17a589 !important;
}
div.nav-container button[data-testid="baseButton-primary"] {
    background: #17a589 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(23,165,137,0.3) !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="nav-container">', unsafe_allow_html=True)
cols = st.columns(len(pages))
for col, (icon, name) in zip(cols, pages):
    with col:
        btn_type = "primary" if name == current else "secondary"
        if st.button(f"{icon}  {name}", key=f"nav_{name}",
                     use_container_width=True, type=btn_type):
            st.session_state["current_page"] = name
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ── ROUTE TO PAGE ─────────────────────────────────────────────────────────────
page = st.session_state["current_page"]

if page == "Data Collection":
    import pages.data_collection as p
elif page == "Data Processing & Cleaning":
    import pages.data_processing_cleaning as p
elif page == "Descriptive Analysis":
    import pages.descriptive_data as p
elif page == "Machine Learning":
    import pages.machine_learning as p
elif page == "Predict":
    import pages.predict as p
elif page == "About":
    import pages.about as p

p.show()

footer()