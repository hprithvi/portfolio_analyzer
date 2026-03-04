import streamlit as st


def apply_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');

    /* Conifer cone SVG as base64 watermark pattern */
    .stApp {
        background-color: #080d14;
        background-image:
            radial-gradient(ellipse at 20% 50%, #0d2137 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, #071a10 0%, transparent 40%),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Cg opacity='0.04' fill='%2334a853'%3E%3Cellipse cx='60' cy='30' rx='8' ry='12'/%3E%3Cellipse cx='60' cy='45' rx='13' ry='8'/%3E%3Cellipse cx='60' cy='58' rx='17' ry='8'/%3E%3Cellipse cx='60' cy='71' rx='20' ry='8'/%3E%3Cellipse cx='60' cy='84' rx='16' ry='7'/%3E%3Crect x='57' y='88' width='6' height='14' rx='2'/%3E%3C/g%3E%3C/svg%3E");
        background-size: auto, auto, 120px 120px;
        font-family: 'Syne', sans-serif;
    }

    /* Main content area */
    .main .block-container {
        background: rgba(10, 16, 28, 0.85);
        border: 1px solid rgba(52, 168, 83, 0.1);
        border-radius: 12px;
        padding: 2rem 3rem;
        backdrop-filter: blur(12px);
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        color: #e8f0fe;
        letter-spacing: -0.02em;
    }

    h1 {
        border-bottom: 1px solid rgba(52, 168, 83, 0.3);
        padding-bottom: 0.5rem;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: rgba(52, 168, 83, 0.05);
        border: 1px solid rgba(52, 168, 83, 0.15);
        border-radius: 8px;
        padding: 1rem;
    }

    [data-testid="stMetricValue"] {
        font-family: 'DM Mono', monospace;
        color: #34a853;
        font-size: 1.8rem !important;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'DM Mono', monospace;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(6, 10, 18, 0.95);
        border-right: 1px solid rgba(52, 168, 83, 0.1);
    }

    /* Dataframes & tables */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(52, 168, 83, 0.15);
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: rgba(52, 168, 83, 0.1);
        color: #34a853;
        border: 1px solid rgba(52, 168, 83, 0.3);
        border-radius: 6px;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.05em;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: rgba(52, 168, 83, 0.2);
        border-color: #34a853;
        box-shadow: 0 0 12px rgba(52, 168, 83, 0.2);
    }

    /* Selectboxes & inputs */
    .stSelectbox > div > div,
    .stTextInput > div > div {
        background: rgba(10, 16, 28, 0.9);
        border-color: rgba(52, 168, 83, 0.2) !important;
        color: #e8f0fe;
    }

    /* Number inputs */
    [data-testid="stNumberInput"] input {
        font-family: 'DM Mono', monospace;
        color: #34a853;
    }

    /* General text */
    p, li, label {
        color: #9aa5b4;
        font-family: 'Syne', sans-serif;
    }

    /* Positive/negative colors for finance */
    .positive { color: #34a853; }
    .negative { color: #ea4335; }
    </style>
    """, unsafe_allow_html=True)
