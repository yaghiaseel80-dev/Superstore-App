import streamlit as st

def apply_style():
    st.markdown("""
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f0f4f8;
        color: #2c3e50;
    }

    header[data-testid="stHeader"] {
        background-color: #f0f4f8;
    }

    [data-testid="stNavLink"] {
        font-size: 14px;
        font-weight: 500;
        color: #2c3e50 !important;
        border-radius: 8px;
        padding: 6px 14px;
        transition: background 0.2s;
    }
    [data-testid="stNavLink"]:hover {
        background-color: #d0eae8;
        color: #17a589 !important;
    }
    [data-testid="stNavLink"][aria-selected="true"] {
        background-color: #17a589 !important;
        color: white !important;
        font-weight: 600;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }

    .page-title {
        font-size: 26px;
        font-weight: 700;
        color: #17a589;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 14px;
        color: #7f8c8d;
        margin-bottom: 24px;
    }

    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #1a5276;
        border-left: 4px solid #17a589;
        padding-left: 10px;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    .kpi-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-top: 4px solid #17a589;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .kpi-icon {
        font-size: 28px;
        color: #17a589;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 500;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a5276;
    }
    .kpi-delta {
        font-size: 12px;
        color: #17a589;
        margin-top: 4px;
    }

    .content-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }

    .info-banner {
        background-color: #d6eaf8;
        border-left: 5px solid #2980b9;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 14px;
        color: #1a5276;
        margin-bottom: 16px;
    }
    .success-banner {
        background-color: #d5f5e3;
        border-left: 5px solid #17a589;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 14px;
        color: #1e8449;
        margin-bottom: 16px;
    }
    .warning-banner {
        background-color: #fdebd0;
        border-left: 5px solid #e87c5a;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 14px;
        color: #a04000;
        margin-bottom: 16px;
    }

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e8ecef;
    }

    [data-testid="stTab"] {
        font-size: 14px;
        font-weight: 500;
        color: #2c3e50;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #17a589 !important;
        border-bottom: 3px solid #17a589 !important;
        font-weight: 600 !important;
    }

    [data-testid="stButton"] > button {
        background-color: #17a589;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        padding: 8px 20px;
        transition: background 0.2s;
    }
    [data-testid="stButton"] > button:hover {
        background-color: #148f77;
    }

    [data-testid="stSidebar"] {
        background-color: #1a5276;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stRadio"] label,
    [data-testid="stMultiSelect"] label {
        font-weight: 500;
        font-size: 13px;
        color: #2c3e50;
    }

    [data-testid="stExpander"] {
        border: 1px solid #e0e7ef;
        border-radius: 10px;
        background-color: #ffffff;
    }

    .footer-bar {
        text-align: center;
        font-size: 12px;
        color: #aab4be;
        padding: 20px 0 10px 0;
        border-top: 1px solid #e0e7ef;
        margin-top: 40px;
    }

    </style>
    """, unsafe_allow_html=True)


def kpi_card(icon_class, label, value, delta=None):
    """
    Returns HTML string for a KPI card.
    Use with st.markdown(..., unsafe_allow_html=True)
    or just call it directly — it renders itself.
    """
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-icon"><i class="bi {icon_class}"></i></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def page_header(title, subtitle=""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def footer():
    st.markdown("""
    <div class="footer-bar">
        DAN614 – Advanced Data Visualization &nbsp;|&nbsp;
        MSDA &nbsp;|&nbsp; LAU Adnan Kassar School of Business &nbsp;|&nbsp;
        Aseel Yaghi &nbsp;|&nbsp; Superstore KDD Analytics Dashboard
    </div>
    """, unsafe_allow_html=True)