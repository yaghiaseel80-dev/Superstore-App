import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.style import apply_style, page_header, section_header, kpi_card
from src.modeling import prepare_features, train_models, get_feature_importance, interpret_results

TEAL  = "#17a589"
CORAL = "#e07b54"
NAVY  = "#1a5276"
BLUE  = "#2e86c1"
TEAL3 = "#a8d8d0"

def _fmt(v):
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

def _chart_layout(title, height=400):
    return dict(
        title=dict(text=f"<b>{title}</b>", x=0, font=dict(size=15, color=NAVY, family="Inter")),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter", size=12, color="#2c3e50"),
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
    )

def _section(text):
    st.markdown(
        f"<div style='font-size:17px; font-weight:700; color:#1a5276; "
        f"border-left:4px solid #17a589; padding-left:10px; "
        f"margin-top:24px; margin-bottom:14px;'>{text}</div>",
        unsafe_allow_html=True
    )


def show():
    apply_style()

    st.markdown("""
    <style>
    div[data-testid="stSelectbox"] label,
    div[data-testid="stMultiSelect"] label {
        font-weight: 700 !important; font-size: 13px !important; color: #1a5276 !important;
    }
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        color: #7f8c8d !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #17a589 !important;
        border-bottom: 3px solid #17a589 !important;
        font-weight: 700 !important;
    }
    [data-testid="stTabs"] {
        margin-top: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    page_header(
        "Machine Learning",
        "Train predictive models on your dataset and uncover which factors drive sales and profit."
    )

    if "raw_df" not in st.session_state:
        st.warning("⚠️ No dataset loaded. Please upload your data in the Data Collection page first.")
        return

    df = st.session_state.get("cleaned_df", st.session_state["raw_df"]).copy()
    for c in ["Sales", "Profit", "Quantity", "Discount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # ── Configuration panel ───────────────────────────────────────────────────
    with st.expander("⚙  Model Configuration", expanded=True):

        available_features = [
            c for c in ["Quantity", "Discount", "Category", "Region",
                         "Segment", "Ship Mode", "Sub-Category"]
            if c in df.columns
        ]

        col1, col2 = st.columns(2)

        with col1:
            # Input features FIRST, empty by default
            selected_features = st.multiselect(
                "Input Features",
                options=available_features,
                default=[],          # empty by default
                key="ml_features",
                placeholder="Select features the model will learn from..."
            )

        with col2:
            target = st.selectbox(
                "Target Variable",
                options=["Sales", "Profit"],
                key="ml_target_widget"
            )

        st.markdown(
            "<p style='font-size:12px; color:#888; margin-top:6px;'>"
            "Select at least 2 input features, then choose what to predict and run the models.</p>",
            unsafe_allow_html=True
        )

        run_btn = st.button("▶  Run Models", key="run_ml", type="primary")

    # ── Run models ────────────────────────────────────────────────────────────
    if run_btn:
        if len(selected_features) < 2:
            st.error("Please select at least 2 input features before running.")
            st.stop()

        with st.spinner("Training models — this may take a moment..."):
            try:
                X, y, feature_names = prepare_features(df, target=target,
                                                        feature_cols=selected_features)
                if len(X) < 50:
                    st.error("Not enough data rows to train. Please upload a larger dataset.")
                    st.stop()

                results = train_models(X, y)
                st.session_state["ml_results"]       = results
                st.session_state["ml_feature_names"] = feature_names
                st.session_state["ml_target_used"]   = target
                st.session_state["ml_ran"]           = True

            except Exception as e:
                st.error(f"Model training failed: {e}")
                st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    if not st.session_state.get("ml_ran"):
        st.markdown(
            "<div style='background:#f0faf8; border:1.5px solid #17a589; border-radius:10px; "
            "padding:20px 24px; margin-top:20px; text-align:center; color:#1a5276;'>"
            "<i class='bi bi-cpu-fill' style='font-size:28px; color:#17a589;'></i><br><br>"
            "<b style='font-size:15px;'>Select your features and run the models to see results here.</b>"
            "</div>",
            unsafe_allow_html=True
        )
        return

    results       = st.session_state["ml_results"]
    feature_names = st.session_state["ml_feature_names"]
    target_used   = st.session_state.get("ml_target_used", "Sales")
    lr            = results["Linear Regression"]
    rf            = results["Random Forest"]
    better_r2     = "Random Forest" if rf["r2"]  > lr["r2"]  else "Linear Regression"
    better_mae    = "Random Forest" if rf["mae"] < lr["mae"] else "Linear Regression"
    best_r2       = max(lr["r2"],  rf["r2"])
    best_mae      = min(lr["mae"], rf["mae"])
    best_cv       = max(lr["cv_r2_mean"], rf["cv_r2_mean"])

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    _section("Model Performance Summary")

    better = "Random Forest" if rf["r2"] > lr["r2"] else "Linear Regression"

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("bi-trophy-fill",  "Best Model",        better)
    with c2: kpi_card("bi-bullseye",     "LR R²  ·  RF R²",  f"{lr['r2']:.3f}  ·  {rf['r2']:.3f}")
    with c3: kpi_card("bi-calculator",   "LR MAE  ·  RF MAE",f"{_fmt(lr['mae'])}  ·  {_fmt(rf['mae'])}")
    with c4: kpi_card("bi-shield-check", "Target Variable",   target_used)

    st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "  Linear Regression",
        "  Random Forest",
        "  Comparison & Insights"
    ])

    # helper: actual vs predicted chart
    def avp_chart(res, model_name):
        y_test = res["y_test"]
        y_pred = res["y_pred"]
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color=TEAL, size=5, opacity=0.5,
                        line=dict(color="white", width=0.4)),
            name="Predictions",
            hovertemplate=f"Actual: $%{{x:,.0f}}<br>Predicted: $%{{y:,.0f}}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color=CORAL, width=2, dash="dash"),
            name="Perfect Prediction"
        ))
        fig.update_layout(
            **_chart_layout(f"Actual vs Predicted — {model_name}", height=420),
            margin=dict(t=60, b=80, l=60, r=30),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            xaxis=dict(title=f"Actual {target_used} ($)",
                       showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
            yaxis=dict(title=f"Predicted {target_used} ($)",
                       showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
        )
        return fig

    # helper: feature importance chart
    def fi_chart(res, model_name):
        imp = get_feature_importance(res["model"], feature_names, top_n=12)
        if imp.empty:
            return None
        fig = go.Figure(go.Bar(
            x=imp["Importance %"], y=imp["Feature"],
            orientation="h",
            text=imp["Importance %"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            marker=dict(color=imp["Importance %"],
                        colorscale=[[0, TEAL3],[1, NAVY]],
                        showscale=False, line_width=0),
            cliponaxis=False
        ))
        fig.update_layout(
            **_chart_layout(f"Feature Importance — {model_name}", height=440),
            margin=dict(t=60, b=40, l=20, r=90),
            xaxis=dict(title="Importance %", showgrid=True,
                       gridcolor="#f0f4f8", range=[0, imp["Importance %"].max()*1.25]),
            yaxis=dict(showgrid=False, autorange="reversed"),
            showlegend=False,
        )
        return fig

    # helper: metrics card for one model
    def metrics_card(res, model_name, color):
        st.markdown(f"""
        <div style='background:#ffffff; border-radius:12px; padding:20px 24px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.07); border-top:4px solid {color};
                    margin-bottom:16px;'>
            <div style='font-size:15px; font-weight:700; color:{NAVY}; margin-bottom:14px;'>
                {model_name} — Metrics
            </div>
            <table style='width:100%; border-collapse:collapse; font-size:13px;'>
                <tr style='background:#f8fafb;'>
                    <td style='padding:8px 12px; font-weight:600; color:#555;'>MAE</td>
                    <td style='padding:8px 12px; font-weight:700; color:{NAVY};'>${res['mae']:,.2f}</td>
                    <td style='padding:8px 12px; font-size:11px; color:#888;'>Average prediction error</td>
                </tr>
                <tr>
                    <td style='padding:8px 12px; font-weight:600; color:#555;'>RMSE</td>
                    <td style='padding:8px 12px; font-weight:700; color:{NAVY};'>${res['rmse']:,.2f}</td>
                    <td style='padding:8px 12px; font-size:11px; color:#888;'>Penalises large errors more</td>
                </tr>
                <tr style='background:#f8fafb;'>
                    <td style='padding:8px 12px; font-weight:600; color:#555;'>R²</td>
                    <td style='padding:8px 12px; font-weight:700; color:{NAVY};'>{res['r2']:.4f}</td>
                    <td style='padding:8px 12px; font-size:11px; color:#888;'>Variance explained (1.0 = perfect)</td>
                </tr>
                <tr>
                    <td style='padding:8px 12px; font-weight:600; color:#555;'>CV R² Mean</td>
                    <td style='padding:8px 12px; font-weight:700; color:{NAVY};'>{res['cv_r2_mean']:.4f}</td>
                    <td style='padding:8px 12px; font-size:11px; color:#888;'>Cross-validated score</td>
                </tr>
                <tr style='background:#f8fafb;'>
                    <td style='padding:8px 12px; font-weight:600; color:#555;'>CV R² Std</td>
                    <td style='padding:8px 12px; font-weight:700; color:{NAVY};'>{res['cv_r2_std']:.4f}</td>
                    <td style='padding:8px 12px; font-size:11px; color:#888;'>Consistency across folds</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — LINEAR REGRESSION
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        metrics_card(lr, "Linear Regression", TEAL)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(avp_chart(lr, "Linear Regression"), use_container_width=True)
        with col2:
            fig = fi_chart(lr, "Linear Regression")
            if fig: st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<p style='font-size:12px; color:#888; margin-top:4px;'>"
            "Left: points close to the dashed line = accurate predictions. "
            "Right: taller bars = stronger influence on the prediction.</p>",
            unsafe_allow_html=True
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — RANDOM FOREST
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        metrics_card(rf, "Random Forest", BLUE)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(avp_chart(rf, "Random Forest"), use_container_width=True)
        with col2:
            fig = fi_chart(rf, "Random Forest")
            if fig: st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<p style='font-size:12px; color:#888; margin-top:4px;'>"
            "Left: points close to the dashed line = accurate predictions. "
            "Right: taller bars = stronger influence on the prediction.</p>",
            unsafe_allow_html=True
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — COMPARISON & INSIGHTS
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        # Side by side metrics
        _section("Metrics Comparison")
        col1, col2 = st.columns(2)
        with col1: metrics_card(lr, "Linear Regression", TEAL)
        with col2: metrics_card(rf, "Random Forest", BLUE)

        # Comparison bar chart
        _section("Visual Comparison")
        metrics_names = ["MAE", "RMSE", "R²", "CV R²"]
        lr_vals = [lr["mae"], lr["rmse"], lr["r2"]*1000, lr["cv_r2_mean"]*1000]
        rf_vals = [rf["mae"], rf["rmse"], rf["r2"]*1000, rf["cv_r2_mean"]*1000]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Linear Regression", x=metrics_names, y=[lr["mae"], lr["rmse"], lr["r2"], lr["cv_r2_mean"]],
                             marker_color=TEAL, text=[f"${lr['mae']:,.0f}", f"${lr['rmse']:,.0f}", f"{lr['r2']:.4f}", f"{lr['cv_r2_mean']:.4f}"],
                             textposition="outside"))
        fig.add_trace(go.Bar(name="Random Forest",     x=metrics_names, y=[rf["mae"], rf["rmse"], rf["r2"], rf["cv_r2_mean"]],
                             marker_color=BLUE, text=[f"${rf['mae']:,.0f}", f"${rf['rmse']:,.0f}", f"{rf['r2']:.4f}", f"{rf['cv_r2_mean']:.4f}"],
                             textposition="outside"))
        fig.update_layout(
            **_chart_layout("Model Metrics Side by Side", height=380),
            margin=dict(t=60, b=60, l=50, r=30),
            barmode="group",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#f0f4f8", title="Value"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Business interpretation
        _section("Business Interpretation")

        lr_res = results["Linear Regression"]
        rf_res = results["Random Forest"]
        best_name = "Random Forest" if rf_res["r2"] > lr_res["r2"] else "Linear Regression"
        best_res  = results[best_name]
        other_name = "Linear Regression" if best_name == "Random Forest" else "Random Forest"
        other_res  = results[other_name]

        # Get top feature from Random Forest
        rf_imp = get_feature_importance(rf_res["model"], feature_names, top_n=1)
        top_feature = rf_imp["Feature"].iloc[0].replace("_", " ").title() if not rf_imp.empty else "Quantity"
        top_pct     = rf_imp["Importance %"].iloc[0] if not rf_imp.empty else 0

        r2_pct    = round(best_res["r2"] * 100, 1)
        mae_val   = best_res["mae"]
        mae_fmt   = _fmt(mae_val)

        if best_res["r2"] >= 0.75:
            model_quality = "strong"
            quality_color = "#17a589"
            quality_icon  = "bi-check-circle-fill"
            quality_text  = "The model explains a strong portion of variation — reliable for forecasting."
        elif best_res["r2"] >= 0.5:
            model_quality = "moderate"
            quality_color = "#d4ac0d"
            quality_icon  = "bi-exclamation-circle-fill"
            quality_text  = "The model explains a moderate portion of variation — useful for trend analysis."
        else:
            model_quality = "limited"
            quality_color = "#e07b54"
            quality_icon  = "bi-info-circle-fill"
            quality_text  = "The model has limited predictive power. More features or cleaner data could improve results."

        # Pre-compute all values to avoid quote conflicts in f-string
        best_r2      = best_res["r2"]
        other_r2     = other_res["r2"]
        best_cv      = best_res["cv_r2_mean"]

        card1 = f"""
        <div style="background:#ffffff; border-radius:14px; padding:22px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.07); border-top:4px solid #17a589;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <i class="bi bi-trophy-fill" style="font-size:22px; color:#17a589;"></i>
                <span style="font-size:14px; font-weight:700; color:#1a5276;">Best Model</span>
            </div>
            <p style="font-size:13px; color:#2c3e50; line-height:1.6; margin:0;">
                <b>{best_name}</b> outperformed {other_name} with an R² of <b>{best_r2:.4f}</b>
                versus <b>{other_r2:.4f}</b>. It explains <b>{r2_pct}%</b> of the
                variation in {target_used} — making it the more reliable choice.
            </p>
        </div>"""

        card2 = f"""
        <div style="background:#ffffff; border-radius:14px; padding:22px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.07); border-top:4px solid #2e86c1;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <i class="bi bi-bullseye" style="font-size:22px; color:#2e86c1;"></i>
                <span style="font-size:14px; font-weight:700; color:#1a5276;">Prediction Accuracy</span>
            </div>
            <p style="font-size:13px; color:#2c3e50; line-height:1.6; margin:0;">
                On average, predictions are off by <b>{mae_fmt}</b> per record (MAE).
                The cross-validation score of <b>{best_cv:.4f}</b> confirms this
                performance holds on unseen data — not just the training set.
            </p>
        </div>"""

        card3 = f"""
        <div style="background:#ffffff; border-radius:14px; padding:22px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.07); border-top:4px solid {quality_color};">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <i class="bi {quality_icon}" style="font-size:22px; color:{quality_color};"></i>
                <span style="font-size:14px; font-weight:700; color:#1a5276;">Model Quality</span>
            </div>
            <p style="font-size:13px; color:#2c3e50; line-height:1.6; margin:0;">
                {quality_text}
            </p>
        </div>"""

        card4 = f"""
        <div style="background:#ffffff; border-radius:14px; padding:22px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.07); border-top:4px solid #7d3c98;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <i class="bi bi-graph-up-arrow" style="font-size:22px; color:#7d3c98;"></i>
                <span style="font-size:14px; font-weight:700; color:#1a5276;">Key Business Driver</span>
            </div>
            <p style="font-size:13px; color:#2c3e50; line-height:1.6; margin:0;">
                The most influential factor driving <b>{target_used}</b> is <b>{top_feature}</b>,
                accounting for <b>{top_pct:.1f}%</b> of the model's decisions.
                Business strategy should prioritise this variable when planning promotions,
                pricing adjustments, or inventory decisions.
            </p>
        </div>"""

        card5 = f"""
        <div style="background:#ffffff; border-radius:14px; padding:22px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.07); border-top:4px solid #e07b54;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <i class="bi bi-lightbulb-fill" style="font-size:22px; color:#e07b54;"></i>
                <span style="font-size:14px; font-weight:700; color:#1a5276;">Recommendation</span>
            </div>
            <p style="font-size:13px; color:#2c3e50; line-height:1.6; margin:0;">
                Use the <b>{best_name}</b> model for {target_used} forecasting.
                Apply all data cleaning steps before running models for best accuracy.
                Focus on the top features in the importance chart to understand
                what levers the business can pull to influence {target_used}.
            </p>
        </div>"""

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:16px;">
            {card1}{card2}{card3}
        </div>
        <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-top:16px;">
            {card4}{card5}
        </div>
        """, unsafe_allow_html=True)