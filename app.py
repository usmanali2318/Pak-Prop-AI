"""
PakProp AI - Pakistan house price prediction and market analytics.
Streamlit app. Run: streamlit run app.py

Layout: mobile-first, bottom dock navigation (Home / Predict / Market / Models).
Styling lives in theme.py; the green palette is unchanged.

PAYLOAD CONTRACT (keys read from models/best_model.pkl):
  model, model_name, label_encoder_city, label_encoder_ptype,
  location_median_price_map, global_median_price, city_ptype_median_map,
  city_locations_map, feature_names, feature_importances, results, cv_scores,
  y_test_pkr, y_pred_pkr, y_pred_log, residuals, df_clean,
  total_records, cities_supported, property_types
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import theme
from utils import (AREA_CAT_ORDER, AREA_UNITS, GREEN_ACCENT, GREEN_LIGHT, GREEN_PALETTE,
                   GREEN_SINGLE, MARLA_TO_SQFT, PLOTLY_TEMPLATE, convert_to_marla,
                   get_area_category, normalize_location, pkr_to_display)

MODEL_PATH = Path("models/best_model.pkl")
HOUSE_TYPES = ["House", "Upper Portion", "Lower Portion", "Farm House"]
FLAT_TYPES = ["Flat", "Penthouse", "Room"]
DEV_A, DEV_B = "Syed Usman Ali", "Daniyal Khalid"

st.set_page_config(page_title="PakProp AI | House Price Analytics", page_icon=":material/home:",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(theme.CSS, unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 0.35) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def make_layout(**overrides) -> dict:
    """Merge PLOTLY_TEMPLATE with per-chart overrides (axes are merged, not replaced)."""
    layout = dict(PLOTLY_TEMPLATE)
    for k, v in overrides.items():
        if k in ("xaxis", "yaxis") and k in layout and isinstance(v, dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


def bottom_legend() -> dict:
    return dict(orientation="h", y=-0.28, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)", font=dict(size=10))


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


payload = load_model()


def missing_model() -> None:
    st.markdown(theme.note("<b>Model file not found.</b> Run <code>python model_training.py</code> first."),
                unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f"<div class='foot'>Built by <b>{DEV_A}</b> and <b>{DEV_B}</b><br>"
                "Estimates are indicative, not a formal valuation</div>", unsafe_allow_html=True)


# ── app bar + navigation ──────────────────────────────────────────────────────

status = f"{payload['model_name']} ready" if payload else "No model"
st.markdown(theme.appbar(status), unsafe_allow_html=True)

tab_home, tab_pred, tab_mkt, tab_mod = st.tabs([
    ":material/home: Home",
    ":material/calculate: Predict",
    ":material/bar_chart: Market",
    ":material/model_training: Models",
])


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown(
        "<div class='hero'><span class='hero-badge'>Pakistan real estate</span>"
        "<h1>Know what a property is worth before you make an offer</h1>"
        "<p>A model trained on real Zameen.com listings for Lahore, Karachi and Islamabad. "
        "Prices in Lakh and Crore, areas in Marla and Kanal.</p>"
        f"<div class='hero-by'>BY <b>{DEV_A.upper()}</b> / <b>{DEV_B.upper()}</b></div></div>",
        unsafe_allow_html=True)

    if not payload:
        missing_model()
    else:
        df_stats = payload["df_clean"]
        best_r2 = payload["results"][payload["model_name"]]["R2"]
        st.markdown(theme.grid([
            theme.mcard("Properties", f"{payload['total_records']:,}", "listings analysed"),
            theme.mcard("Cities", str(len(payload["cities_supported"])), ", ".join(sorted(payload["cities_supported"]))),
            theme.mcard("Median price", pkr_to_display(df_stats["price"].median()), "across all listings"),
            theme.mcard("Accuracy (R²)", f"{best_r2:.1%}", payload["model_name"]),
        ]), unsafe_allow_html=True)

        st.markdown(theme.sec("Median price by city"), unsafe_allow_html=True)
        city_med = df_stats.groupby("city")["price"].median().reset_index()
        city_med["lakh"] = city_med["price"] / 1e5
        city_med = city_med.sort_values("lakh", ascending=False)
        fig = go.Figure(go.Bar(
            x=city_med["city"], y=city_med["lakh"],
            marker=dict(color=GREEN_PALETTE[:len(city_med)], line=dict(color="#064e3b", width=1)),
            text=[f"{v:.0f}L" for v in city_med["lakh"]], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Median: PKR %{y:.0f} Lakh<extra></extra>"))
        fig.update_layout(**make_layout(yaxis_title="PKR Lakh", height=300, showlegend=False))
        plot(fig)

    st.markdown(theme.sec("How it works"), unsafe_allow_html=True)
    st.markdown(
        theme.step(1, "Collect", "Real listings from Zameen.com for Pakistan's three largest property markets.")
        + theme.step(2, "Train", "Random Forest, XGBoost and Ridge are compared. The best one by R² is kept.")
        + theme.step(3, "Predict", "Enter the property details and get an estimate with a likely range."),
        unsafe_allow_html=True)
    st.markdown(theme.note(
        f"Semester project by <b>{DEV_A}</b> and <b>{DEV_B}</b>, built with Python, Streamlit and scikit-learn. "
        "Supports Marla, Kanal, Square Yards and Square Feet."), unsafe_allow_html=True)
    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown(theme.sec("Property details"), unsafe_allow_html=True)
    if not payload:
        missing_model()
    else:
        le_city, le_ptype = payload["label_encoder_city"], payload["label_encoder_ptype"]
        loc_median_map = payload.get("location_median_price_map", {})
        df_clean = payload["df_clean"]
        global_median = payload.get("global_median_price", df_clean["price"].median())
        city_ptype_map = payload.get("city_ptype_median_map", {})
        city_locations_map = payload.get("city_locations_map", {})
        model = payload["model"]

        c1, c2 = st.columns(2)
        with c1:
            city = st.selectbox("City", sorted(payload["cities_supported"]))
        with c2:
            property_type = st.selectbox("Property type", sorted(payload["property_types"]))

        c3, c4 = st.columns(2)
        with c3:
            loc_pick = st.selectbox("Location", ["Any location"] + city_locations_map.get(city, []),
                                    help="Picking a known neighbourhood improves accuracy.")
            location_input = "" if loc_pick == "Any location" else loc_pick
        with c4:
            area_unit = st.selectbox("Area unit", AREA_UNITS)

        c5, c6 = st.columns(2)
        with c5:
            if area_unit == "Marla":
                area_val = st.number_input("Area (Marla)", 0.5, 500.0, 10.0, 0.5)
            elif area_unit == "Kanal":
                area_val = st.number_input("Area (Kanal)", 0.1, 100.0, 1.0, 0.1)
            elif area_unit == "Sq. Yd.":
                area_val = st.number_input("Area (Sq. Yards)", 50.0, 50000.0, 500.0, 50.0)
            else:
                area_val = st.number_input("Area (Sq. Ft.)", 200.0, 200000.0, 1000.0, 100.0)
        with c6:
            bedrooms = st.number_input("Bedrooms", 0, 20, 3, 1)
        c7, _ = st.columns(2)
        with c7:
            baths = st.number_input("Bathrooms", 0, 20, 2, 1)

        area_marla = convert_to_marla(area_val, area_unit)
        area_sqft = area_marla * MARLA_TO_SQFT
        area_sqyd = area_sqft / 9.0
        st.markdown(theme.chips([f"{area_marla:.2f} Marla", f"{area_sqft:,.0f} Sq. Ft.", f"{area_sqyd:,.0f} Sq. Yd."]),
                    unsafe_allow_html=True)

        if st.button("Estimate price", type="primary"):
            try:
                city_enc = le_city.transform([city if city in le_city.classes_ else le_city.classes_[0]])[0]
                ptype_enc = le_ptype.transform([property_type if property_type in le_ptype.classes_ else le_ptype.classes_[0]])[0]
                loc_key = normalize_location(location_input.strip()) if location_input.strip() else ""
                loc_level = loc_median_map.get(loc_key, global_median)
                cp_median = city_ptype_map.get((city, property_type), global_median)
                area_cat = (0 if area_sqyd < 200 else 1 if area_sqyd < 500 else
                            2 if area_sqyd < 1000 else 3 if area_sqyd < 2000 else 4)
                features = np.array([[np.log1p(area_marla), float(bedrooms), float(baths), float(city_enc),
                                      float(ptype_enc), np.log1p(loc_level), np.log1p(cp_median), float(area_cat)]])
                price = float(np.expm1(model.predict(features)[0]))
                comparable = df_clean[(df_clean["city"] == city) & (df_clean["property_type"] == property_type)
                                      & (df_clean["area_marla"].between(area_marla * 0.7, area_marla * 1.3))]
                st.session_state["pred"] = dict(
                    price=price, low=price * 0.70, high=price * 1.30, n=len(comparable), city=city,
                    ptype=property_type, area=f"{area_val:g} {area_unit}", beds=int(bedrooms), baths=int(baths),
                    ppm=price / area_marla, cat=get_area_category(area_sqyd), loc=location_input)
            except Exception as e:  # keep the UI alive on bad input
                st.session_state.pop("pred", None)
                st.error(f"Prediction error: {e}")

        r = st.session_state.get("pred")
        if r:
            st.markdown(
                "<div class='hero'><div class='hero-label'>Estimated market value</div>"
                f"<div class='hero-price'>{pkr_to_display(r['price'])}</div>"
                f"<div class='hero-alt'>PKR {r['price']:,.0f}</div>"
                + theme.range_bar(r["low"], r["price"], r["high"], pkr_to_display(r["low"]), pkr_to_display(r["high"]))
                + f"<div class='hero-note'>Based on {r['n']:,} comparable listings in {r['city']}</div></div>",
                unsafe_allow_html=True)
            st.markdown(theme.grid([
                theme.mcard("City", r["city"], r["loc"] or "any location"),
                theme.mcard("Type", r["ptype"]),
                theme.mcard("Area", r["area"], r["cat"]),
                theme.mcard("Beds / baths", f"{r['beds']} / {r['baths']}", f"{pkr_to_display(r['ppm'])} per Marla"),
            ]), unsafe_allow_html=True)
    footer()


# ══════════════════════════════════════════════════════════════════════════════
# MARKET
# ══════════════════════════════════════════════════════════════════════════════
with tab_mkt:
    if not payload:
        missing_model()
    else:
        kind = st.segmented_control("Property group", ["Houses", "Flats"], default="Houses", key="mkt_kind",
                                    label_visibility="collapsed") or "Houses"
        df_mkt = payload["df_clean"].copy()
        df_mkt["price_lakh"] = df_mkt["price"] / 1e5
        df_mkt["area_cat"] = df_mkt["area_sqyd"].apply(get_area_category)
        df_t = df_mkt[df_mkt["property_type"].isin(HOUSE_TYPES if kind == "Houses" else FLAT_TYPES)].copy()

        if df_t.empty:
            st.markdown(theme.note(f"No data available for {kind.lower()}."), unsafe_allow_html=True)
        else:
            st.markdown(theme.grid([
                theme.mcard("Listings", f"{len(df_t):,}", kind.lower()),
                theme.mcard("Median price", pkr_to_display(df_t["price"].median())),
            ]), unsafe_allow_html=True)

            st.markdown(theme.sec("Median price by city"), unsafe_allow_html=True)
            city_avg = df_t.groupby("city")["price_lakh"].median().reset_index().sort_values("price_lakh", ascending=False)
            fig1 = go.Figure(go.Bar(
                x=city_avg["city"], y=city_avg["price_lakh"],
                marker=dict(color=GREEN_PALETTE[:len(city_avg)], line=dict(color="#064e3b", width=0.8)),
                text=[f"{v:.0f}L" for v in city_avg["price_lakh"]], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Median: PKR %{y:.1f} Lakh<extra></extra>"))
            fig1.update_layout(**make_layout(yaxis_title="PKR Lakh", height=300, showlegend=False))
            plot(fig1)

            st.markdown(theme.sec("Price spread by city"), unsafe_allow_html=True)
            fig2 = go.Figure()
            for i, cn in enumerate(city_avg["city"]):
                col = GREEN_PALETTE[i % len(GREEN_PALETTE)]
                fig2.add_trace(go.Box(y=df_t[df_t["city"] == cn]["price_lakh"], name=cn, marker_color=col,
                                      line=dict(color="#064e3b", width=1.4), fillcolor=hex_to_rgba(col), boxmean=True,
                                      hovertemplate=f"<b>{cn}</b><br>PKR %{{y:.1f}} Lakh<extra></extra>"))
            fig2.update_layout(**make_layout(yaxis_title="PKR Lakh", height=320, showlegend=False))
            plot(fig2)

            st.markdown(theme.sec("Price by size category"), unsafe_allow_html=True)
            area_avg = df_t.groupby(["city", "area_cat"], observed=True)["price_lakh"].median().reset_index()
            fig3 = go.Figure()
            for i, cn in enumerate(sorted(df_t["city"].unique())):
                sub = area_avg[area_avg["city"] == cn]
                fig3.add_trace(go.Bar(name=cn, x=sub["area_cat"], y=sub["price_lakh"],
                                      marker_color=GREEN_PALETTE[i * 2 % len(GREEN_PALETTE)],
                                      hovertemplate=f"<b>{cn}</b><br>%{{x}}<br>PKR %{{y:.1f}} Lakh<extra></extra>"))
            fig3.update_layout(**make_layout(
                barmode="group", height=340, yaxis_title="PKR Lakh", legend=bottom_legend(),
                xaxis=dict(categoryorder="array", categoryarray=AREA_CAT_ORDER, title="Square yards", tickangle=-30)))
            plot(fig3)

            st.markdown(theme.sec("Area vs price"), unsafe_allow_html=True)
            fig4 = px.scatter(df_t.sample(min(4000, len(df_t)), random_state=42), x="area_marla", y="price_lakh",
                              color="city", color_discrete_sequence=GREEN_PALETTE, opacity=0.45,
                              labels={"area_marla": "Area (Marla)", "price_lakh": "PKR Lakh", "city": "City"})
            fig4.update_layout(**make_layout(height=340, legend=bottom_legend()))
            fig4.update_traces(hovertemplate="Area: %{x:.1f} Marla<br>PKR %{y:.1f} Lakh<extra></extra>")
            plot(fig4)

            st.markdown(theme.sec("Summary by city"), unsafe_allow_html=True)
            rows = []
            for cn, g in sorted(df_t.groupby("city"), key=lambda kv: -len(kv[1])):
                rows.append(theme.row(cn, f"{len(g):,} listings", [
                    ("Min", f"{g['price_lakh'].min():.0f}L"), ("Median", f"{g['price_lakh'].median():.0f}L"),
                    ("Max", f"{g['price_lakh'].max():.0f}L"), ("Area", f"{g['area_marla'].median():.1f} Marla")]))
            st.markdown("".join(rows), unsafe_allow_html=True)
    footer()


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab_mod:
    if not payload:
        missing_model()
    else:
        results, best = payload["results"], payload["model_name"]
        feat_imp, feat_names = payload["feature_importances"], payload["feature_names"]
        y_test_pkr, y_pred_pkr = payload["y_test_pkr"], payload["y_pred_pkr"]
        residuals = payload["residuals"]
        cv = payload.get("cv_scores")
        cv_arr = np.array(cv) if cv is not None else None

        b = results[best]
        st.markdown(
            "<div class='hero'><div class='hero-label'>Best performing model</div>"
            f"<div class='hero-price' style='font-size:clamp(1.8rem,8vw,2.6rem)'>{best}</div>"
            f"<div class='hero-alt'>R² {b['R2']:.4f}  /  MAE {b['MAE']/1e5:.0f}L  /  RMSE {b['RMSE']/1e5:.0f}L"
            + (f"  /  CV {cv_arr.mean():.3f} ± {cv_arr.std():.3f}" if cv_arr is not None else "")
            + "</div></div>", unsafe_allow_html=True)

        st.markdown(theme.sec("Model comparison"), unsafe_allow_html=True)
        metric = st.segmented_control("Metric", ["R²", "MAE", "RMSE"], default="R²", key="mod_metric",
                                      label_visibility="collapsed") or "R²"
        names = list(results.keys())
        vals = {"R²": [results[m]["R2"] for m in names],
                "MAE": [results[m]["MAE"] / 1e5 for m in names],
                "RMSE": [results[m]["RMSE"] / 1e5 for m in names]}[metric]
        fmt = (lambda v: f"{v:.3f}") if metric == "R²" else (lambda v: f"{v:.0f}L")
        note_txt = "higher is better" if metric == "R²" else "lower is better, PKR Lakh"
        fig_m = go.Figure(go.Bar(
            x=names, y=vals, marker_color=[GREEN_SINGLE if n == best else GREEN_LIGHT for n in names],
            text=[fmt(v) for v in vals], textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"))
        fig_m.update_layout(**make_layout(height=300, showlegend=False,
                                          title=dict(text=f"{metric} ({note_txt})", font=dict(size=12, color="#065f46")),
                                          yaxis=dict(range=[0, 1.1]) if metric == "R²" else {}))
        plot(fig_m)

        if feat_imp is not None:
            st.markdown(theme.sec("Feature importance"), unsafe_allow_html=True)
            fi = pd.DataFrame({"Feature": feat_names, "Importance": feat_imp}).sort_values("Importance")
            fig_fi = go.Figure(go.Bar(
                x=fi["Importance"], y=fi["Feature"], orientation="h",
                marker=dict(color=fi["Importance"], colorscale=[[0, "#d1fae5"], [0.5, "#10b981"], [1, "#064e3b"]], showscale=False),
                text=[f"{v:.3f}" for v in fi["Importance"]], textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>"))
            fig_fi.update_layout(**make_layout(height=max(280, len(feat_names) * 40), showlegend=False,
                                               yaxis=dict(automargin=True), margin=dict(l=8, r=48, t=20, b=36)))
            plot(fig_fi)

        st.markdown(theme.sec("Actual vs predicted"), unsafe_allow_html=True)
        n_diag = min(2000, len(y_test_pkr))
        idx = np.random.RandomState(42).choice(len(y_test_pkr), n_diag, replace=False)
        act, pred = y_test_pkr[idx] / 1e5, y_pred_pkr[idx] / 1e5
        top = float(max(act.max(), pred.max()))
        fig_av = go.Figure()
        fig_av.add_trace(go.Scatter(x=act, y=pred, mode="markers", name="Properties",
                                    marker=dict(color=GREEN_ACCENT, opacity=0.4, size=5),
                                    hovertemplate="Actual: %{x:.1f}L<br>Predicted: %{y:.1f}L<extra></extra>"))
        fig_av.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect",
                                    line=dict(color="#064e3b", dash="dash", width=1.5)))
        fig_av.update_layout(**make_layout(xaxis_title="Actual (PKR Lakh)", yaxis_title="Predicted (PKR Lakh)",
                                           height=360, legend=bottom_legend()))
        plot(fig_av)

        st.markdown(theme.sec("Residuals"), unsafe_allow_html=True)
        fig_h = go.Figure(go.Histogram(x=residuals, nbinsx=50, marker_color=GREEN_LIGHT,
                                       marker_line=dict(color="#064e3b", width=0.5), opacity=0.85,
                                       hovertemplate="Residual: %{x:.3f}<br>Count: %{y}<extra></extra>"))
        fig_h.add_vline(x=0, line=dict(color="#064e3b", dash="dash", width=2))
        fig_h.update_layout(**make_layout(xaxis_title="Residual (log price)", yaxis_title="Count", height=300, showlegend=False))
        plot(fig_h)
        fig_r = go.Figure(go.Scatter(x=payload["y_pred_log"][:n_diag], y=residuals[:n_diag], mode="markers",
                                     marker=dict(color=GREEN_LIGHT, opacity=0.4, size=4),
                                     hovertemplate="Predicted (log): %{x:.2f}<br>Residual: %{y:.3f}<extra></extra>"))
        fig_r.add_hline(y=0, line=dict(color="#064e3b", dash="dash", width=1.5))
        fig_r.update_layout(**make_layout(xaxis_title="Predicted log price", yaxis_title="Residual", height=300, showlegend=False))
        plot(fig_r)

        st.markdown(theme.note(
            "<b>R²</b> is the share of price variance explained, closer to 1 is better. "
            "<b>MAE</b> is the average error and <b>RMSE</b> punishes big misses harder. "
            "In the actual vs predicted chart, points near the dashed line are accurate. "
            "Residuals should centre on zero with no pattern."), unsafe_allow_html=True)
    footer()
