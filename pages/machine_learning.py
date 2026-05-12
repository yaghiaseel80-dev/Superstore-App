import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.style import apply_style, page_header, section_header, kpi_card
from src.modeling import (
    prepare_features, train_models, get_feature_importance, interpret_results,
    REGRESSION_CONFIG, CLASSIFICATION_CONFIG,
    prepare_cluster_data, find_optimal_clusters, run_clustering
)

TEAL   = "#17a589"
NAVY   = "#1a2940"
CORAL  = "#e74c3c"
AMBER  = "#f39c12"
GREEN  = "#27ae60"
PURPLE = "#8e44ad"
BLUE   = "#2e86c1"

CLUSTER_COLORS = [TEAL, PURPLE, AMBER, BLUE, CORAL, GREEN]


def _chart_layout(title="", legend_below=False):
    layout = dict(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=NAVY)),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=NAVY),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#e0e0e0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#e0e0e0"),
    )
    if legend_below:
        layout["legend"] = dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
    else:
        layout["showlegend"] = False
    return layout


def _sub_nav(options, state_key):
    if state_key not in st.session_state:
        st.session_state[state_key] = options[0]

    cols = st.columns(len(options))
    for i, opt in enumerate(options):
        with cols[i]:
            selected = st.session_state[state_key] == opt
            label    = f"● {opt}" if selected else opt
            if st.button(label, key=f"subnav_{state_key}_{i}", use_container_width=True):
                st.session_state[state_key] = opt
                st.rerun()

    return st.session_state[state_key]


def show():
    apply_style()
    page_header("Machine Learning",
                "Train supervised and unsupervised models on your Superstore dataset.")

    if "raw_df" not in st.session_state:
        st.warning("Please upload your dataset on the Data Collection page first.")
        return

    df = st.session_state.get("cleaned_df", st.session_state["raw_df"]).copy()

    st.markdown("""<style>
    button[data-baseweb="tab"] {
        font-size: 15px !important; font-weight: 700 !important; padding: 10px 28px !important;
    }
    </style>""", unsafe_allow_html=True)

    main_sup, main_unsup = st.tabs([
        "  Supervised Learning  ",
        "  Unsupervised Learning  "
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUPERVISED
    # ══════════════════════════════════════════════════════════════════════════
    with main_sup:
        st.markdown("<br>", unsafe_allow_html=True)

        sup_mode = _sub_nav(
            ["Predicting Numbers  (Regression)", "Predicting Categories  (Classification)"],
            "sup_mode"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if "Regression" in sup_mode:
            _render_supervised_section(
                df,
                config      = REGRESSION_CONFIG,
                section_key = "reg",
                model_type  = "regression",
                color       = TEAL,
                subtitle    = "Estimate exact numerical values — like how many days an order will take to ship."
            )
        else:
            _render_supervised_section(
                df,
                config      = CLASSIFICATION_CONFIG,
                section_key = "clf",
                model_type  = "classification",
                color       = PURPLE,
                subtitle    = "Classify orders into meaningful categories — like Fast, Normal, or Slow shipping."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # UNSUPERVISED
    # ══════════════════════════════════════════════════════════════════════════
    with main_unsup:
        st.markdown("<br>", unsafe_allow_html=True)
        _render_clustering(df)


# ── Supervised section ────────────────────────────────────────────────────────

def _render_supervised_section(df, config, section_key, model_type, color, subtitle):
    st.markdown(
        f"<div style='background:#f8fffe;border-left:4px solid {color};border-radius:8px;"
        f"padding:1.2rem 1.5rem;margin-bottom:1.5rem;'>"
        f"<i class='bi bi-sliders2-vertical' style='color:{color};margin-right:8px;'></i>"
        f"<strong style='color:{NAVY};font-size:15px;'>Model Configuration</strong>"
        f"<p style='color:#555;margin:0.4rem 0 0;font-size:13px;'>{subtitle}</p>"
        f"</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        # Use a different key from session state storage key
        target = st.selectbox("**Prediction Target**",
                               options=list(config.keys()),
                               key=f"{section_key}_target_widget")
        st.caption(config[target]["description"])
    with c2:
        features = st.multiselect("**Input Features**",
                                   options=config[target]["features"],
                                   default=[],
                                   key=f"{section_key}_features_widget")

    if st.button("Run Models", type="primary", key=f"run_{section_key}"):
        if not features:
            st.error("Please select at least one input feature.")
        else:
            with st.spinner("Training models..."):
                try:
                    X, y, fnames = prepare_features(df, target=target, feature_cols=features)
                    results      = train_models(X, y, target)
                    # Store with _trained_ prefix to avoid widget key conflicts
                    st.session_state[f"{section_key}_results"]         = results
                    st.session_state[f"{section_key}_fnames"]          = fnames
                    st.session_state[f"{section_key}_trained_target"]  = target
                    st.session_state[f"{section_key}_trained_features"]= features
                    st.session_state[f"{section_key}_interp"]          = interpret_results(results, target)
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if f"{section_key}_results" not in st.session_state:
        st.info("Configure your model above and click **Run Models** to see results.")
        return

    results      = st.session_state[f"{section_key}_results"]
    fnames       = st.session_state[f"{section_key}_fnames"]
    trained_tgt  = st.session_state[f"{section_key}_trained_target"]

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Model Performance Summary")

    if model_type == "regression":
        _regression_kpis(results, config[trained_tgt].get("unit", ""))
        m1_name = "Linear Regression"
    else:
        _classification_kpis(results)
        m1_name = "Logistic Regression"

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([
        f"  {m1_name}  ", "  Random Forest  ", "  Comparison & Insights  "
    ])

    with tab1:
        if model_type == "regression":
            _regression_tab(results[m1_name], m1_name, fnames,
                            config[trained_tgt].get("unit", ""), color)
        else:
            _classification_tab(results[m1_name], m1_name, fnames, color)

    with tab2:
        if model_type == "regression":
            _regression_tab(results["Random Forest"], "Random Forest", fnames,
                            config[trained_tgt].get("unit", ""), PURPLE)
        else:
            _classification_tab(results["Random Forest"], "Random Forest", fnames, PURPLE)

    with tab3:
        if model_type == "regression":
            _regression_comparison(results, trained_tgt, config[trained_tgt].get("unit", ""))
        else:
            _classification_comparison(results, trained_tgt)


# ── Regression KPIs ───────────────────────────────────────────────────────────

def _regression_kpis(results, unit):
    lr = results["Linear Regression"]
    rf = results["Random Forest"]
    best_name = "Random Forest" if rf["r2"] >= lr["r2"] else "Linear Regression"
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("bi-trophy-fill", "Best Model", best_name)
    with k2: kpi_card("bi-bullseye",    "Best R²",    str(max(lr["r2"], rf["r2"])))
    with k3: kpi_card("bi-rulers",      "LR MAE",     f"{lr['mae']} {unit}")
    with k4: kpi_card("bi-rulers",      "RF MAE",     f"{rf['mae']} {unit}")


# ── Classification KPIs ───────────────────────────────────────────────────────

def _classification_kpis(results):
    m1  = list(results.keys())[0]
    m2  = list(results.keys())[1]
    r1  = results[m1]
    r2  = results[m2]
    best_name = m1 if r1["acc"] >= r2["acc"] else m2
    best_acc  = max(r1["acc"], r2["acc"])
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("bi-trophy-fill",  "Best Model",     best_name)
    with k2: kpi_card("bi-bullseye",     "Best Accuracy",  f"{round(best_acc*100,1)}%")
    with k3: kpi_card("bi-intersect",    "Precision",      f"{round(results[best_name]['precision']*100,1)}%")
    with k4: kpi_card("bi-arrow-repeat", "Recall",         f"{round(results[best_name]['recall']*100,1)}%")


# ── Regression tab ────────────────────────────────────────────────────────────

def _regression_tab(res, model_name, feature_names, unit, color):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("bi-calculator",       "MAE",    f"{res['mae']} {unit}")
    with c2: kpi_card("bi-calculator",       "RMSE",   f"{res['rmse']} {unit}")
    with c3: kpi_card("bi-graph-up",         "R²",     str(res["r2"]))
    with c4: kpi_card("bi-arrow-repeat",     "CV R²",  str(res["cv_r2_mean"]))
    with c5: kpi_card("bi-plus-slash-minus", "CV Std", f"±{res['cv_r2_std']}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        section_header("Actual vs Predicted")
        y_test, y_pred = res["y_test"], res["y_pred"]
        mn = float(min(y_test.min(), y_pred.min()))
        mx = float(max(y_test.max(), y_pred.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                  marker=dict(color=color, opacity=0.6, size=6), name="Predictions"))
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                  line=dict(color=CORAL, dash="dash", width=2), name="Perfect"))
        layout = _chart_layout(f"{model_name} — Actual vs Predicted", legend_below=True)
        layout["xaxis"]["title"] = f"Actual ({unit})"
        layout["yaxis"]["title"] = f"Predicted ({unit})"
        layout["margin"] = dict(l=50, r=30, t=60, b=100)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        _feature_importance_chart(res, model_name, feature_names, color)


# ── Classification tab ────────────────────────────────────────────────────────

def _classification_tab(res, model_name, feature_names, color):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("bi-bullseye",         "Accuracy",    f"{round(res['acc']*100,1)}%")
    with c2: kpi_card("bi-intersect",        "Precision",   f"{round(res['precision']*100,1)}%")
    with c3: kpi_card("bi-arrow-repeat",     "Recall",      f"{round(res['recall']*100,1)}%")
    with c4: kpi_card("bi-shield-check",     "Specificity", f"{round(res['specificity']*100,1)}%")
    with c5: kpi_card("bi-stars",            "F1 Score",    f"{round(res['f1']*100,1)}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        section_header("Confusion Matrix")
        cm      = res["cm"]
        classes = res["classes"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=list(classes), y=list(classes),
            colorscale=[[0, "#f0faf8"], [1, color]],
            text=cm, texttemplate="%{text}", showscale=False
        ))
        layout = _chart_layout(f"{model_name} — Confusion Matrix")
        layout["xaxis"]["title"] = "Predicted"
        layout["yaxis"]["title"] = "Actual"
        layout["margin"] = dict(l=80, r=40, t=60, b=60)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        _feature_importance_chart(res, model_name, feature_names, color)


# ── Feature importance ────────────────────────────────────────────────────────

def _feature_importance_chart(res, model_name, feature_names, color):
    section_header("Feature Importance")
    fi_df = get_feature_importance(res["model"], feature_names)
    if not fi_df.empty:
        fig = go.Figure(go.Bar(
            x=fi_df["Importance %"], y=fi_df["Feature"], orientation="h",
            marker=dict(color=fi_df["Importance %"],
                        colorscale=[[0, "#b2dfdb"], [1, color]], showscale=False),
            text=[f"{v:.1f}%" for v in fi_df["Importance %"]], textposition="outside"
        ))
        layout = _chart_layout(f"{model_name} — Feature Importance")
        layout["margin"] = dict(l=160, r=60, t=60, b=40)
        layout["xaxis"]["title"] = "Importance (%)"
        layout["yaxis"]["autorange"] = "reversed"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available.")


# ── Regression comparison ─────────────────────────────────────────────────────

def _regression_comparison(results, target, unit):
    section_header("Side-by-Side Metrics")
    lr = results["Linear Regression"]
    rf = results["Random Forest"]

    metrics = ["MAE", "RMSE", "R²", "CV R²"]
    lr_vals = [lr["mae"], lr["rmse"], lr["r2"], lr["cv_r2_mean"]]
    rf_vals = [rf["mae"], rf["rmse"], rf["r2"], rf["cv_r2_mean"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Linear Regression", x=metrics, y=lr_vals,
                          marker_color=TEAL, text=[str(v) for v in lr_vals], textposition="outside"))
    fig.add_trace(go.Bar(name="Random Forest", x=metrics, y=rf_vals,
                          marker_color=PURPLE, text=[str(v) for v in rf_vals], textposition="outside"))
    layout = _chart_layout("Linear Regression vs Random Forest", legend_below=True)
    layout["barmode"] = "group"
    layout["margin"]  = dict(l=40, r=40, t=60, b=100)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
    _insight_cards(results, target, unit, mode="regression")


# ── Classification comparison ─────────────────────────────────────────────────

def _classification_comparison(results, target):
    section_header("Side-by-Side Metrics")
    m1_name = list(results.keys())[0]
    m2_name = list(results.keys())[1]
    r1 = results[m1_name]
    r2 = results[m2_name]

    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
    r1_vals = [round(r1["acc"]*100,1), round(r1["precision"]*100,1),
               round(r1["recall"]*100,1), round(r1["specificity"]*100,1), round(r1["f1"]*100,1)]
    r2_vals = [round(r2["acc"]*100,1), round(r2["precision"]*100,1),
               round(r2["recall"]*100,1), round(r2["specificity"]*100,1), round(r2["f1"]*100,1)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=m1_name, x=metrics, y=r1_vals, marker_color=PURPLE,
                          text=[f"{v}%" for v in r1_vals], textposition="outside"))
    fig.add_trace(go.Bar(name=m2_name, x=metrics, y=r2_vals, marker_color=BLUE,
                          text=[f"{v}%" for v in r2_vals], textposition="outside"))
    layout = _chart_layout("Logistic Regression vs Random Forest", legend_below=True)
    layout["barmode"] = "group"
    layout["yaxis"]["title"] = "Score (%)"
    layout["margin"] = dict(l=40, r=40, t=60, b=100)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
    _insight_cards(results, target, "", mode="classification")


# ── Insight cards ─────────────────────────────────────────────────────────────

def _insight_cards(results, target, unit, mode):
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Business Insights")

    if mode == "regression":
        lr = results["Linear Regression"]
        rf = results["Random Forest"]
        best_name = "Random Forest" if rf["r2"] >= lr["r2"] else "Linear Regression"
        best      = rf if rf["r2"] >= lr["r2"] else lr
        score     = best["r2"]
        color_q   = GREEN if score >= 0.75 else (AMBER if score >= 0.5 else CORAL)
        quality   = "Strong" if score >= 0.75 else ("Moderate" if score >= 0.5 else "Limited")
        rec       = "Reliable for planning." if score >= 0.75 else ("Validate before acting." if score >= 0.5 else "Add more features.")
        metric_line = f"R² = {best['r2']} | MAE = {best['mae']} {unit}"
    else:
        m1 = list(results.keys())[0]
        m2 = list(results.keys())[1]
        best_name = m1 if results[m1]["acc"] >= results[m2]["acc"] else m2
        best      = results[best_name]
        score     = best["acc"]
        color_q   = GREEN if score >= 0.80 else (AMBER if score >= 0.65 else CORAL)
        quality   = "Strong" if score >= 0.80 else ("Moderate" if score >= 0.65 else "Limited")
        rec       = "Reliable for business decisions." if score >= 0.80 else ("Use with caution." if score >= 0.65 else "Consider feature engineering.")
        metric_line = f"Accuracy = {round(score*100,1)}% | F1 = {round(best['f1']*100,1)}%"

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;'>
      <div style='background:#fff;border-top:4px solid {TEAL};border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
        <i class='bi bi-trophy-fill' style='color:{TEAL};font-size:20px;'></i>
        <div style='font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;'>Best Model</div>
        <div style='color:#555;font-size:13px;'><b>{best_name}</b> — {metric_line}</div>
      </div>
      <div style='background:#fff;border-top:4px solid {color_q};border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
        <i class='bi bi-graph-up-arrow' style='color:{color_q};font-size:20px;'></i>
        <div style='font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;'>Model Quality</div>
        <div style='color:#555;font-size:13px;'>{quality} performance — {rec}</div>
      </div>
      <div style='background:#fff;border-top:4px solid {GREEN};border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
        <i class='bi bi-lightbulb-fill' style='color:{GREEN};font-size:20px;'></i>
        <div style='font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;'>Recommendation</div>
        <div style='color:#555;font-size:13px;'>Use <b>{best_name}</b> for predicting <b>{target}</b>. {rec}</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── Clustering ────────────────────────────────────────────────────────────────

def _render_clustering(df):
    st.markdown(f"""
    <div style='background:#f8fffe;border-left:4px solid {TEAL};border-radius:8px;
                padding:1.2rem 1.5rem;margin-bottom:1.5rem;'>
        <i class='bi bi-diagram-3-fill' style='color:{TEAL};margin-right:8px;'></i>
        <strong style='color:{NAVY};font-size:15px;'>Customer Segmentation — K-Means Clustering</strong>
        <p style='color:#555;margin:0.4rem 0 0;font-size:13px;'>
            Customers are grouped by Sales, Profit, Discount, Quantity, and Order frequency.
            The optimal number of clusters is determined automatically using the Elbow and Silhouette methods.
        </p>
    </div>""", unsafe_allow_html=True)

    if st.button("Run Clustering", type="primary", key="run_clustering"):
        with st.spinner("Finding optimal clusters and fitting model..."):
            try:
                X_sc, scaler, cdf, fcols = prepare_cluster_data(df)
                opt_k, analysis          = find_optimal_clusters(X_sc)
                cdf, profile, km         = run_clustering(X_sc, cdf, fcols, opt_k)
                st.session_state["cluster_model"]    = km
                st.session_state["cluster_scaler"]   = scaler
                st.session_state["cluster_cdf"]      = cdf
                st.session_state["cluster_profile"]  = profile
                st.session_state["cluster_fcols"]    = fcols
                st.session_state["cluster_opt_k"]    = opt_k
                st.session_state["cluster_analysis"] = analysis
            except Exception as e:
                st.error(f"Clustering failed: {e}")

    if "cluster_model" not in st.session_state:
        st.info("Click **Run Clustering** to segment your customers automatically.")
        return

    opt_k    = st.session_state["cluster_opt_k"]
    analysis = st.session_state["cluster_analysis"]
    profile  = st.session_state["cluster_profile"]
    cdf      = st.session_state["cluster_cdf"]
    fcols    = st.session_state["cluster_fcols"]

    st.markdown(
        f"<div style='background:#f0faf8;border-left:4px solid {TEAL};border-radius:8px;"
        f"padding:14px 20px;margin-bottom:20px;font-size:14px;color:{NAVY};'>"
        f"<i class='bi bi-search' style='color:{TEAL};margin-right:8px;'></i>"
        f"Based on {analysis['method']}, the optimal number of customer segments is "
        f"<b>{opt_k} clusters</b>.</div>", unsafe_allow_html=True)

    section_header("Customer Segment Overview")
    seg_cols = st.columns(opt_k)
    for i, (idx, row) in enumerate(profile.iterrows()):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        with seg_cols[i]:
            count = int(row.get("Customer Count", 0))
            label = row.get("Label", f"Segment {i+1}")
            sales = f"${row['Sales']:,.0f}" if "Sales" in row else "N/A"
            st.markdown(
                f"<div style='background:#fff;border-top:4px solid {color};"
                f"border-radius:10px;padding:16px;text-align:center;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.07);'>"
                f"<div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;'>Cluster {idx}</div>"
                f"<div style='font-weight:700;color:{NAVY};font-size:13px;margin:6px 0;'>{label}</div>"
                f"<div style='font-size:24px;font-weight:800;color:{color};'>{count}</div>"
                f"<div style='font-size:11px;color:#aaa;'>customers</div>"
                f"<div style='font-size:12px;color:#555;margin-top:6px;'>Avg Sales: {sales}</div>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Customer Segments — Sales vs Profit")
    if "Sales" in cdf.columns and "Profit" in cdf.columns:
        fig = go.Figure()
        for i in range(opt_k):
            seg   = cdf[cdf["Cluster"] == i]
            label = profile.loc[i, "Label"] if i in profile.index else f"Cluster {i}"
            fig.add_trace(go.Scatter(
                x=seg["Sales"], y=seg["Profit"], mode="markers",
                marker=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], size=7, opacity=0.7),
                name=f"Cluster {i}: {label}"))
        layout = _chart_layout("Sales vs Profit by Segment", legend_below=True)
        layout["xaxis"]["title"] = "Total Sales ($)"
        layout["yaxis"]["title"] = "Total Profit ($)"
        layout["margin"] = dict(l=60, r=40, t=60, b=100)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Cluster Profile Summary")
    st.dataframe(profile.reset_index(), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Business Interpretation")
    cards = ""
    for i, (idx, row) in enumerate(profile.iterrows()):
        color  = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        label  = row.get("Label", f"Segment {i+1}")
        sales  = f"${row['Sales']:,.0f}"       if "Sales"    in row else "N/A"
        profit = f"${row['Profit']:,.0f}"      if "Profit"   in row else "N/A"
        disc   = f"{row['Discount']*100:.1f}%" if "Discount" in row else "N/A"
        count  = int(row.get("Customer Count", 0))
        action = {
            "High-Value Customers":       "Prioritise retention — offer loyalty rewards and premium service.",
            "Growth Potential Customers": "Invest in upselling — targeted promotions can convert them to high-value.",
            "Low-Value Customers":        "Reduce acquisition cost — focus on selective reactivation campaigns.",
            "Mid-Tier Customers":         "Monitor and engage — consistent outreach can grow this segment.",
        }.get(label, "Analyse behaviour and tailor strategy accordingly.")

        cards += f"""
        <div style='background:#fff;border-top:4px solid {color};border-radius:10px;
                    padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
            <div style='font-weight:700;color:{NAVY};font-size:14px;margin-bottom:6px;'>
                Cluster {idx} — {label}</div>
            <div style='font-size:12px;color:#555;line-height:1.8;'>
                <b>{count}</b> customers &nbsp;|&nbsp;
                Avg Sales: <b>{sales}</b> &nbsp;|&nbsp;
                Avg Profit: <b>{profit}</b> &nbsp;|&nbsp;
                Avg Discount: <b>{disc}</b><br>
                <i class='bi bi-arrow-right-circle-fill' style='color:{color};'></i>
                &nbsp;{action}
            </div>
        </div>"""

    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin-top:0.5rem;'>{cards}</div>",
        unsafe_allow_html=True)