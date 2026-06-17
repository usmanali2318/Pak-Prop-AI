"""
Pakistan AI House Price Prediction & Market Analytics Platform
Streamlit Application — all pages in one file
Run: streamlit run app.py

PAYLOAD CONTRACT (keys read from best_model.pkl):
  model                    — trained best model object
  model_name               — name of best model (str)
  label_encoder_city       — LabelEncoder for city
  label_encoder_ptype      — LabelEncoder for property_type
  location_median_price_map — dict: location -> median price (PKR)
  global_median_price      — float: fallback when location unknown
  city_ptype_median_map    — dict: (city, ptype) -> median price (PKR)
  feature_cols / feature_names / feature_importances
  results                  — dict: model_name -> {R2, MAE, RMSE}
  cv_scores                — 5-fold CV R² scores
  y_test_log / y_pred_log / y_test_pkr / y_pred_pkr / residuals
  df_clean                 — cleaned market DataFrame
  city_stats / area_cat_stats / area_cat_labels
  total_records / cities_supported / property_types
"""

import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from utils import (pkr_to_display, get_area_category, AREA_CAT_ORDER,
                   convert_to_marla, normalize_location, AREA_UNITS, GREEN_PALETTE,
                   GREEN_SINGLE, GREEN_LIGHT, GREEN_ACCENT,
                   PLOTLY_TEMPLATE, MARLA_TO_SQFT)

MODEL_PATH = Path("models/best_model.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def hex_to_rgba(hex_color: str, alpha: float = 0.35) -> str:
    """Convert a hex color string to rgba() for use in Plotly fillcolor."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def make_layout(**overrides) -> dict:
    """Merge PLOTLY_TEMPLATE with per-chart overrides, avoiding duplicate keyword errors.
    Handles xaxis/yaxis dict merging so callers can partially override axes."""
    layout = dict(PLOTLY_TEMPLATE)
    for k, v in overrides.items():
        if k in ('xaxis', 'yaxis') and k in layout and isinstance(v, dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PakProp AI | House Price Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: #f9fafb !important;
    color: #111827 !important;
}

.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ─── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #064e3b 0%, #065f46 50%, #059669 100%) !important;
    border-right: none !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.10) !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio > label {
    color: #a7f3d0 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    padding: 0.55rem 1rem !important;
    margin: 0.15rem 0 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] [aria-checked="true"] {
    background: rgba(255,255,255,0.20) !important;
    border-color: rgba(255,255,255,0.40) !important;
}

/* ─── Cards ───────────────────────────────────────────────────────── */
.stat-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    border: 1px solid #e5e7eb;
    text-align: center;
    transition: box-shadow 0.15s;
}
.stat-card:hover { box-shadow: 0 4px 16px rgba(5,150,105,0.10); }
.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #059669;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}
.stat-card .stat-label {
    font-size: 0.78rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
.stat-card .stat-icon {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
}

/* ─── Page headers ────────────────────────────────────────────────── */
.page-header h2 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.25rem;
}
.page-header p {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 0;
}
.page-header hr {
    margin: 1rem 0 1.5rem 0;
    border: none;
    border-top: 1px solid #e5e7eb;
}

/* ─── Section Headers ─────────────────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111827;
    border-left: 3px solid #059669;
    padding-left: 0.75rem;
    margin: 1.75rem 0 1rem 0;
}

/* ─── Hero Banner ─────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #059669 100%);
    border-radius: 12px;
    padding: 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -5%;
    width: 360px;
    height: 360px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-banner h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: white !important;
    margin-bottom: 0.4rem !important;
}
.hero-banner p {
    font-size: 1rem !important;
    color: #a7f3d0 !important;
    max-width: 560px !important;
    margin-bottom: 0 !important;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

/* ─── Form Card ───────────────────────────────────────────────────── */
.form-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 2rem;
    border: 1px solid #e5e7eb;
}

/* ─── Prediction Result ───────────────────────────────────────────── */
.prediction-result {
    background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    text-align: center;
    color: white;
    margin: 1.25rem 0;
}
.prediction-result .pred-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #a7f3d0;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.prediction-result .pred-amount {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.25rem;
    line-height: 1.2;
}
.prediction-result .pred-amount-alt {
    font-size: 1rem;
    color: #d1fae5;
    margin-bottom: 0.75rem;
}
.pred-range {
    font-size: 0.88rem;
    color: #a7f3d0;
    margin-top: 0.4rem;
}
.pred-note {
    font-size: 0.78rem;
    color: #6ee7b7;
    margin-top: 0.3rem;
}

/* ─── Info Box ────────────────────────────────────────────────────── */
.info-box {
    background: #ecfdf5;
    border-left: 3px solid #059669;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #064e3b;
}

/* ─── Metric Tags ─────────────────────────────────────────────────── */
.metric-tag {
    display: inline-block;
    background: #ecfdf5;
    color: #065f46;
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem;
}

/* ─── Tabs ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #f3f4f6 !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 3px !important;
    border: 1px solid #e5e7eb !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    color: #374151 !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    border: none !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: #059669 !important;
    color: white !important;
}

/* ─── Inputs ──────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}

/* ─── Button ──────────────────────────────────────────────────────── */
.stButton > button {
    background: #059669 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 2rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #047857 !important; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL (CACHED)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

payload = load_model()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.25rem 0 0.75rem;'>
        <div style='font-size:2rem;'>🏠</div>
        <div style='font-size:1.15rem; font-weight:700; color:white; margin-top:0.3rem;'>PakProp AI</div>
        <div style='font-size:0.7rem; color:#a7f3d0; letter-spacing:1.5px; text-transform:uppercase;'>
            Price Intelligence
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.12); margin: 0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        options=["Home", "Predict Price", "Market Analysis", "ML Models"],
        index=0,
        label_visibility="visible"
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12); margin:1.25rem 0 0.75rem;'>",
                unsafe_allow_html=True)

    if payload:
        cv = payload.get('cv_scores', None)
        cv_arr = np.array(cv) if cv is not None else None
        cv_str = f"{cv_arr.mean():.3f}" if cv_arr is not None else "—"
        st.markdown(f"""
        <div style='font-size:0.78rem; color:#d1fae5; line-height:2;'>
            Model: {payload['model_name']}<br>
            R² Score: {payload['results'][payload['model_name']]['R2']:.3f}<br>
            CV R² (5-fold): {cv_str}<br>
            Records: {payload['total_records']:,}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(220,38,38,0.15); border-radius:8px; padding:0.8rem;
                    font-size:0.8rem; border:1px solid rgba(220,38,38,0.3);'>
            <div style='color:#fca5a5; font-weight:600; margin-bottom:0.25rem;'>Model not trained</div>
            <div style='color:#fecaca;'>Run: python model_training.py</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 2rem; padding-top: 0.75rem;
                border-top: 1px solid rgba(255,255,255,0.12);
                font-size: 0.7rem; color: rgba(167,243,208,0.7);
                text-align: center; line-height: 1.9;'>
        Developed by<br>
        <span style='color: #d1fae5; font-weight: 600;'>Syed Usman Ali</span><br>
        <span style='color: #d1fae5; font-weight: 600;'>Daniyal Khalid</span>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>Pakistan Real Estate Intelligence</div>
        <h1>AI-Powered Property Price Prediction</h1>
        <p>Machine learning trained on real Zameen.com listings to give you accurate
        property valuations across Lahore, Karachi, and Islamabad.</p>
        <div style='margin-top: 1.1rem; font-size: 0.78rem; color: rgba(167,243,208,0.8);
                    letter-spacing: 0.3px;'>
            Developed by &nbsp;<strong style='color:#d1fae5;'>Syed Usman Ali</strong>
            &nbsp;&amp;&nbsp;
            <strong style='color:#d1fae5;'>Daniyal Khalid</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if payload:
        df_stats = payload['df_clean']
        total    = payload['total_records']
        cities   = len(payload['cities_supported'])
        med_p    = df_stats['price'].median()
        best_r2  = payload['results'][payload['model_name']]['R2']

        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, "🏘️", f"{total:,}", "Properties Analysed"),
            (c2, "🏙️", str(cities), "Major Cities"),
            (c3, "💰", pkr_to_display(med_p), "Median Property Price"),
            (c4, "🎯", f"{best_r2:.1%}", "Model Accuracy (R²)"),
        ]
        for col, icon, val, label in cards:
            with col:
                st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-icon'>{icon}</div>
                    <div class='stat-value'>{val}</div>
                    <div class='stat-label'>{label}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.error("Model file not found. Please run: python model_training.py")

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    if payload:
        st.markdown("<div class='section-header'>City-wise Market Overview</div>",
                    unsafe_allow_html=True)
        df_stats = payload['df_clean']
        city_med = df_stats.groupby('city')['price'].median().reset_index()
        city_med['price_lakh'] = city_med['price'] / 1e5
        city_med = city_med.sort_values('price_lakh', ascending=False)

        fig = go.Figure(go.Bar(
            x=city_med['city'],
            y=city_med['price_lakh'],
            marker=dict(color=GREEN_PALETTE[:len(city_med)], line=dict(color='#064e3b', width=1)),
            text=[f"PKR {v:.0f}L" for v in city_med['price_lakh']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Median Price: PKR %{y:.0f} Lakh<extra></extra>'
        ))
        fig.update_layout(**make_layout(
            xaxis_title='City',
            yaxis_title='Median Price (PKR Lakh)',
            height=360,
            showlegend=False
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>How It Works</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    steps = [
        ("Data Collection",
         "Real property listings sourced from Zameen.com, focused on Lahore, Karachi, and Islamabad — Pakistan's three largest and most data-rich real estate markets."),
        ("ML Training",
         "Multiple regression models compared — Random Forest, XGBoost, Ridge. Best model selected automatically by R² score."),
        ("Instant Prediction",
         "Enter property details and get a market-calibrated price estimate in PKR Lakh/Crore format with a confidence range."),
    ]
    for col, (title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
            <div class='stat-card' style='text-align:left; padding:1.2rem 1.4rem;'>
                <div style='font-size:0.9rem; font-weight:600; color:#059669;
                            margin-bottom:0.5rem; text-transform:uppercase;
                            letter-spacing:0.5px;'>{title}</div>
                <div style='font-size:0.85rem; color:#6b7280; line-height:1.6;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <b>About this project</b><br>
        Developed as a semester project by <b>Syed Usman Ali</b> and <b>Daniyal Khalid</b>
        using Python, Streamlit, and Scikit-learn. Trained on real Zameen.com listings
        for Lahore, Karachi, and Islamabad. Supports Pakistan-native area units
        (Marla, Kanal, Square Yards) and displays prices in PKR Lakh and Crore format.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT PRICE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Price":

    st.markdown("""
    <div class='page-header'>
        <h2>Property Price Predictor</h2>
        <p>Enter property details below to get an AI-powered price estimate.</p>
        <hr>
    </div>""", unsafe_allow_html=True)

    if not payload:
        st.error("Model file not found. Please run: python model_training.py")
        st.stop()

    le_city             = payload['label_encoder_city']
    le_ptype            = payload['label_encoder_ptype']
    loc_median_map      = payload.get('location_median_price_map', {})
    global_median       = payload.get('global_median_price', payload['df_clean']['price'].median())
    city_ptype_map      = payload.get('city_ptype_median_map', {})
    city_locations_map  = payload.get('city_locations_map', {})
    model               = payload['model']
    df_clean            = payload['df_clean']

    available_cities = sorted(payload['cities_supported'])
    available_types  = sorted(payload['property_types'])

    st.markdown("<div class='form-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", options=available_cities, index=0)
    with col2:
        property_type = st.selectbox("Property Type", options=available_types, index=0)

    col3, col4 = st.columns(2)
    with col3:
        city_locs = city_locations_map.get(city, [])
        location_options = ['-- Select Location --'] + city_locs
        selected_location = st.selectbox(
            "Location / Neighbourhood",
            options=location_options,
            index=0,
            help="Locations filtered to your selected city. Selecting a known location improves accuracy."
        )
        location_input = selected_location if selected_location != '-- Select Location --' else ''
    with col4:
        area_unit = st.selectbox("Area Unit", options=AREA_UNITS, index=0)

    col5, col6 = st.columns(2)
    with col5:
        if area_unit == 'Marla':
            area_val = st.number_input("Area (Marla)", min_value=0.5, max_value=500.0, value=10.0, step=0.5)
        elif area_unit == 'Kanal':
            area_val = st.number_input("Area (Kanal)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
        elif area_unit == 'Sq. Yd.':
            area_val = st.number_input("Area (Sq. Yards)", min_value=50.0, max_value=50000.0, value=500.0, step=50.0)
        else:
            area_val = st.number_input("Area (Sq. Ft.)", min_value=200.0, max_value=200000.0, value=1000.0, step=100.0)
    with col6:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)

    col7, col8 = st.columns(2)
    with col7:
        baths = st.number_input("Bathrooms", min_value=0, max_value=20, value=2, step=1)
    # col8 intentionally left empty — renders invisibly in Streamlit

    st.markdown("</div>", unsafe_allow_html=True)

    area_marla = convert_to_marla(area_val, area_unit)
    area_sqft  = area_marla * MARLA_TO_SQFT
    area_sqyd  = area_sqft / 9.0

    st.markdown(f"""
    <div class='metric-tag'>{area_marla:.2f} Marla</div>
    <div class='metric-tag'>= {area_sqft:.0f} Sq. Ft.</div>
    <div class='metric-tag'>= {area_sqyd:.0f} Sq. Yd.</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Price", type="primary"):
        try:
            if city in le_city.classes_:
                city_enc = le_city.transform([city])[0]
            else:
                city_enc = le_city.transform([le_city.classes_[0]])[0]

            if property_type in le_ptype.classes_:
                ptype_enc = le_ptype.transform([property_type])[0]
            else:
                ptype_enc = le_ptype.transform([le_ptype.classes_[0]])[0]

            if location_input and location_input.strip():
                loc_key = normalize_location(location_input.strip())
            else:
                loc_key = ''
            location_price_level = loc_median_map.get(loc_key, global_median)

            city_ptype_key = (city, property_type)
            city_ptype_median = city_ptype_map.get(city_ptype_key, global_median)

            area_cat = (0 if area_sqyd < 200 else
                        1 if area_sqyd < 500 else
                        2 if area_sqyd < 1000 else
                        3 if area_sqyd < 2000 else 4)

            features = np.array([[
                np.log1p(area_marla),
                float(bedrooms),
                float(baths),
                float(city_enc),
                float(ptype_enc),
                np.log1p(location_price_level),
                np.log1p(city_ptype_median),
                float(area_cat),
            ]])

            log_pred   = model.predict(features)[0]
            price_pred = np.expm1(log_pred)

            low  = price_pred * 0.70
            high = price_pred * 1.30

            comparable = df_clean[
                (df_clean['city'] == city) &
                (df_clean['property_type'] == property_type) &
                (df_clean['area_marla'].between(area_marla * 0.7, area_marla * 1.3))
            ]
            n_comparable = len(comparable)

            st.markdown(f"""
            <div class='prediction-result'>
                <div class='pred-label'>Estimated Market Value</div>
                <div class='pred-amount'>{pkr_to_display(price_pred)}</div>
                <div class='pred-amount-alt'>PKR {price_pred:,.0f}</div>
                <div class='pred-range'>
                    Estimated range: {pkr_to_display(low)} – {pkr_to_display(high)}
                </div>
                <div class='pred-note'>
                    Based on {n_comparable:,} comparable listings in {city}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-header'>Property Summary</div>", unsafe_allow_html=True)
            sc1, sc2, sc3, sc4 = st.columns(4)
            summaries = [
                ("City", city), ("Type", property_type),
                ("Area", f"{area_val} {area_unit}"), ("Beds / Baths", f"{int(bedrooms)} / {int(baths)}")
            ]
            for col, (label, value) in zip([sc1, sc2, sc3, sc4], summaries):
                with col:
                    st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-label'>{label}</div>
                        <div style='font-size:1rem; font-weight:600; color:#059669; margin-top:0.35rem;'>
                            {value}
                        </div>
                    </div>""", unsafe_allow_html=True)

            ppm = price_pred / area_marla
            st.markdown(f"""
            <div class='info-box'>
                Price per Marla: <b>{pkr_to_display(ppm)}</b> &nbsp;|&nbsp;
                Area category: <b>{get_area_category(area_sqyd)}</b>
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Market Analysis":

    st.markdown("""
    <div class='page-header'>
        <h2>Market Analysis</h2>
        <p>Explore real estate trends, city comparisons, and price distributions.</p>
        <hr>
    </div>""", unsafe_allow_html=True)

    if not payload:
        st.error("Model file not found. Please run: python model_training.py")
        st.stop()

    df_mkt = payload['df_clean'].copy()
    df_mkt['price_lakh'] = df_mkt['price'] / 1e5
    df_mkt['area_cat']   = df_mkt['area_sqyd'].apply(get_area_category)

    tab_house, tab_flat = st.tabs(["Houses", "Flats"])

    def render_market_tab(prop_types: list, tab_label: str):
        df_t = df_mkt[df_mkt['property_type'].isin(prop_types)].copy()

        if len(df_t) == 0:
            st.warning(f"No data available for {tab_label}.")
            return

        st.markdown("<div class='section-header'>City-wise Median Price</div>", unsafe_allow_html=True)
        city_avg = df_t.groupby('city')['price_lakh'].median().reset_index()
        city_avg = city_avg.sort_values('price_lakh', ascending=False)

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=city_avg['city'],
            y=city_avg['price_lakh'],
            marker=dict(color=GREEN_PALETTE[:len(city_avg)], line=dict(color='#064e3b', width=0.8)),
            text=[f"PKR {v:.0f}L" for v in city_avg['price_lakh']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Median: PKR %{y:.1f} Lakh<extra></extra>',
        ))
        fig1.update_layout(**make_layout(
            xaxis_title='City',
            yaxis_title='Median Price (PKR Lakh)',
            height=360,
            showlegend=False
        ))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("<div class='section-header'>Price Distribution by City</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        for i, city_name in enumerate(city_avg['city']):
            d = df_t[df_t['city'] == city_name]['price_lakh']
            fig2.add_trace(go.Box(
                y=d,
                name=city_name,
                marker_color=GREEN_PALETTE[i % len(GREEN_PALETTE)],
                line=dict(color='#064e3b', width=1.5),
                fillcolor=hex_to_rgba(GREEN_PALETTE[i % len(GREEN_PALETTE)]),
                hovertemplate=f'<b>{city_name}</b><br>PKR %{{y:.1f}} Lakh<extra></extra>',
                boxmean=True
            ))
        fig2.update_layout(**make_layout(
            yaxis_title='Price (PKR Lakh)',
            height=380,
            showlegend=False
        ))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-header'>Price by Property Size Category</div>",
                    unsafe_allow_html=True)
        area_avg = (df_t.groupby(['city', 'area_cat'], observed=True)['price_lakh']
                    .median().reset_index())
        area_avg['area_order'] = area_avg['area_cat'].map(
            {v: i for i, v in enumerate(AREA_CAT_ORDER)}
        )
        area_avg = area_avg.dropna(subset=['area_order'])
        area_avg = area_avg.sort_values(['area_order', 'price_lakh'], ascending=[True, False])

        fig3 = go.Figure()
        for i, city_name in enumerate(sorted(df_t['city'].unique())):
            subset = area_avg[area_avg['city'] == city_name].sort_values('area_order')
            fig3.add_trace(go.Bar(
                name=city_name,
                x=subset['area_cat'],
                y=subset['price_lakh'],
                marker_color=GREEN_PALETTE[i * 2 % len(GREEN_PALETTE)],
                hovertemplate=f'<b>{city_name}</b><br>%{{x}}<br>PKR %{{y:.1f}} Lakh<extra></extra>',
            ))
        fig3.update_layout(**make_layout(
            barmode='group',
            xaxis=dict(categoryorder='array', categoryarray=AREA_CAT_ORDER,
                       title='Area Category (Square Yards)'),
            yaxis_title='Median Price (PKR Lakh)',
            height=380,
            legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=11))
        ))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("<div class='section-header'>Area vs Price Relationship</div>",
                    unsafe_allow_html=True)
        df_scatter = df_t.sample(min(5000, len(df_t)), random_state=42)
        fig4 = px.scatter(
            df_scatter,
            x='area_marla',
            y='price_lakh',
            color='city',
            color_discrete_sequence=GREEN_PALETTE,
            opacity=0.45,
            labels={'area_marla': 'Area (Marla)', 'price_lakh': 'Price (PKR Lakh)', 'city': 'City'},
            hover_data={'city': True, 'area_marla': ':.1f', 'price_lakh': ':.1f'}
        )
        fig4.update_layout(**make_layout(height=380))
        fig4.update_traces(
            hovertemplate='<b>%{customdata[0]}</b><br>Area: %{x:.1f} Marla<br>PKR %{y:.1f} Lakh<extra></extra>'
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("<div class='section-header'>Market Summary by City</div>", unsafe_allow_html=True)
        summary = df_t.groupby('city').agg(
            Listings=('price', 'count'),
            Min_Price=('price_lakh', lambda x: f"{x.min():.0f}L"),
            Median_Price=('price_lakh', lambda x: f"{x.median():.0f}L"),
            Max_Price=('price_lakh', lambda x: f"{x.max():.0f}L"),
            Avg_Area_Marla=('area_marla', lambda x: f"{x.median():.1f}")
        ).reset_index()
        summary.columns = ['City', 'Listings', 'Min Price', 'Median Price', 'Max Price', 'Avg Area (Marla)']
        summary = summary.sort_values('Listings', ascending=False)
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with tab_house:
        render_market_tab(['House', 'Upper Portion', 'Lower Portion', 'Farm House'], 'House')
    with tab_flat:
        render_market_tab(['Flat', 'Penthouse', 'Room'], 'Flat')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ML Models":

    st.markdown("""
    <div class='page-header'>
        <h2>ML Model Analytics</h2>
        <p>Model performance metrics, feature importance, and prediction diagnostics.</p>
        <hr>
    </div>""", unsafe_allow_html=True)

    if not payload:
        st.error("Model file not found. Please run: python model_training.py")
        st.stop()

    results    = payload['results']
    best_name  = payload['model_name']
    feat_imp   = payload['feature_importances']
    feat_names = payload['feature_names']
    y_test_pkr = payload['y_test_pkr']
    y_pred_pkr = payload['y_pred_pkr']
    residuals  = payload['residuals']
    cv_scores  = payload.get('cv_scores', None)

    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)

    model_names = list(results.keys())
    r2_vals     = [results[m]['R2']        for m in model_names]
    mae_vals    = [results[m]['MAE'] / 1e5 for m in model_names]
    rmse_vals   = [results[m]['RMSE'] / 1e5 for m in model_names]

    mc1, mc2, mc3 = st.columns(3)

    with mc1:
        fig_r2 = go.Figure(go.Bar(
            x=model_names, y=r2_vals,
            marker_color=[GREEN_SINGLE if n == best_name else GREEN_LIGHT for n in model_names],
            text=[f"{v:.3f}" for v in r2_vals], textposition='outside',
            hovertemplate='<b>%{x}</b><br>R² Score: %{y:.4f}<extra></extra>'
        ))
        fig_r2.update_layout(**make_layout(
            yaxis=dict(range=[0, 1.1], title='R²'),
            height=300, showlegend=False,
            title=dict(text='R² Score (higher = better)', font=dict(size=12, color='#374151'))
        ))
        st.plotly_chart(fig_r2, use_container_width=True)

    with mc2:
        fig_mae = go.Figure(go.Bar(
            x=model_names, y=mae_vals,
            marker_color=[GREEN_SINGLE if n == best_name else GREEN_LIGHT for n in model_names],
            text=[f"{v:.0f}L" for v in mae_vals], textposition='outside',
            hovertemplate='<b>%{x}</b><br>MAE: PKR %{y:.1f} Lakh<extra></extra>'
        ))
        fig_mae.update_layout(**make_layout(
            yaxis_title='PKR Lakh', height=300, showlegend=False,
            title=dict(text='MAE (lower = better)', font=dict(size=12, color='#374151'))
        ))
        st.plotly_chart(fig_mae, use_container_width=True)

    with mc3:
        fig_rmse = go.Figure(go.Bar(
            x=model_names, y=rmse_vals,
            marker_color=[GREEN_SINGLE if n == best_name else GREEN_LIGHT for n in model_names],
            text=[f"{v:.0f}L" for v in rmse_vals], textposition='outside',
            hovertemplate='<b>%{x}</b><br>RMSE: PKR %{y:.1f} Lakh<extra></extra>'
        ))
        fig_rmse.update_layout(**make_layout(
            yaxis_title='PKR Lakh', height=300, showlegend=False,
            title=dict(text='RMSE (lower = better)', font=dict(size=12, color='#374151'))
        ))
        st.plotly_chart(fig_rmse, use_container_width=True)

    best_r2   = results[best_name]['R2']
    best_mae  = results[best_name]['MAE'] / 1e5
    best_rmse = results[best_name]['RMSE'] / 1e5
    cv_arr2   = np.array(cv_scores) if cv_scores is not None else None
    cv_line   = f" &nbsp;|&nbsp; CV R² (5-fold): {cv_arr2.mean():.4f} ± {cv_arr2.std():.4f}" if cv_arr2 is not None else ""

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#064e3b,#059669); border-radius:10px;
                padding:1.25rem 1.75rem; color:white; margin:0.75rem 0;'>
        <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;
                    color:#a7f3d0; margin-bottom:0.4rem;'>Best Performing Model</div>
        <div style='font-size:1.3rem; font-weight:700; margin-bottom:0.6rem;'>{best_name}</div>
        <span class='metric-tag' style='background:rgba(255,255,255,0.18); color:white;'>
            R² {best_r2:.4f}</span>
        <span class='metric-tag' style='background:rgba(255,255,255,0.18); color:white;'>
            MAE PKR {best_mae:.0f}L</span>
        <span class='metric-tag' style='background:rgba(255,255,255,0.18); color:white;'>
            RMSE PKR {best_rmse:.0f}L</span>{cv_line}
    </div>
    """, unsafe_allow_html=True)

    if feat_imp is not None:
        st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
        fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': feat_imp})
        fi_df = fi_df.sort_values('Importance', ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importance'],
            y=fi_df['Feature'],
            orientation='h',
            marker=dict(
                color=fi_df['Importance'],
                colorscale=[[0, '#d1fae5'], [0.5, '#10b981'], [1, '#064e3b']],
                showscale=False,
                line=dict(color='#064e3b', width=0.5)
            ),
            text=[f"{v:.3f}" for v in fi_df['Importance']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        fig_fi.update_layout(**make_layout(
            xaxis_title='Importance Score',
            yaxis=dict(automargin=True, tickfont=dict(size=12)),
            height=max(300, len(feat_names) * 50),
            margin=dict(l=200, r=60, t=40, b=40),
            showlegend=False
        ))
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("<div class='section-header'>Actual vs Predicted Prices</div>",
                unsafe_allow_html=True)
    n_diag  = min(2000, len(y_test_pkr))
    idx     = np.random.choice(len(y_test_pkr), n_diag, replace=False)
    act_l   = y_test_pkr[idx] / 1e5
    pred_l  = y_pred_pkr[idx] / 1e5
    max_val = max(act_l.max(), pred_l.max())

    fig_av = go.Figure()
    fig_av.add_trace(go.Scatter(
        x=act_l, y=pred_l,
        mode='markers',
        marker=dict(color=GREEN_ACCENT, opacity=0.4, size=5,
                    line=dict(color='#064e3b', width=0.3)),
        hovertemplate='Actual: PKR %{x:.1f}L<br>Predicted: PKR %{y:.1f}L<extra></extra>',
        name='Properties'
    ))
    fig_av.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines',
        line=dict(color='#064e3b', dash='dash', width=1.5),
        name='Perfect Prediction'
    ))
    fig_av.update_layout(**make_layout(
        xaxis_title='Actual Price (PKR Lakh)',
        yaxis_title='Predicted Price (PKR Lakh)',
        height=440,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11))
    ))
    st.plotly_chart(fig_av, use_container_width=True)

    st.markdown("<div class='section-header'>Residual Analysis</div>", unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        fig_hist = go.Figure(go.Histogram(
            x=residuals, nbinsx=50,
            marker_color=GREEN_LIGHT,
            marker_line=dict(color='#064e3b', width=0.5),
            opacity=0.85,
            hovertemplate='Residual: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        fig_hist.add_vline(x=0, line=dict(color='#064e3b', dash='dash', width=2))
        fig_hist.update_layout(**make_layout(
            xaxis_title='Residual (Log Price)', yaxis_title='Count',
            height=340, showlegend=False
        ))
        st.plotly_chart(fig_hist, use_container_width=True)

    with rc2:
        fig_res = go.Figure(go.Scatter(
            x=payload['y_pred_log'][:n_diag],
            y=residuals[:n_diag],
            mode='markers',
            marker=dict(color=GREEN_LIGHT, opacity=0.4, size=4,
                        line=dict(color='#064e3b', width=0.3)),
            hovertemplate='Predicted (log): %{x:.2f}<br>Residual: %{y:.3f}<extra></extra>'
        ))
        fig_res.add_hline(y=0, line=dict(color='#064e3b', dash='dash', width=1.5))
        fig_res.update_layout(**make_layout(
            xaxis_title='Predicted Log Price', yaxis_title='Residual',
            height=340, showlegend=False
        ))
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("""
    <div class='info-box'>
        <b>How to read these charts</b><br>
        <b>R²:</b> Proportion of variance explained — closer to 1.0 is better. Above 0.80 is excellent for real estate.<br>
        <b>MAE:</b> Average prediction error in PKR. Lower is better.<br>
        <b>RMSE:</b> Penalises large errors more than MAE. Lower is better.<br>
        <b>Actual vs Predicted:</b> Points near the dashed diagonal indicate accurate predictions.<br>
        <b>Residuals:</b> Should be centred around zero with no systematic pattern.
    </div>""", unsafe_allow_html=True)
