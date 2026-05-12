import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.style import apply_style, page_header, section_header, kpi_card
from src.modeling import (
    prepare_features, train_models, get_feature_importance,
    interpret_results, TARGET_CONFIG,
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
        layout["legend"] = dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    else:
        layout["showlegend"] = False
    return layout


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
        st.markdown(f"""
        <div style='background:#f8fffe;border-left:4px solid {TEAL};border-radius:8px;
                    padding:1.2rem 1.5rem;margin-bottom:1.5rem;'>
            <i class='bi bi-sliders2-vertical' style='color:{TEAL};margin-right:8px;'></i>
            <strong style='color:{NAVY};font-size:15px;'>Model Configuration</strong>
            <p style='color:#555;margin:0.4rem 0 0;font-size:13px;'>
                Select what to predict and which features to use, then run the models.
            </p>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            target_label = st.selectbox("**Prediction Target**",
                                        options=list(TARGET_CONFIG.keys()),
                                        key="ml_target_sel")
            cfg = TARGET_CONFIG[target_label]
            st.caption(cfg["description"])
        with c2:
            selected_features = st.multiselect("**Input Features**",
                                               options=cfg["features"],
                                               default=[],
                                               key="ml_features_sel")

        if st.button("Run Models", type="primary", key="run_supervised"):
            if not selected_features:
                st.error("Please select at least one input feature.")
            else:
                with st.spinner("Training models..."):
                    try:
                        X, y, fnames = prepare_features(df, target=target_label,
                                                        feature_cols=selected_features)
                        results = train_models(X, y)
                        st.session_state["ml_results"]       = results
                        st.session_state["ml_feature_names"] = fnames
                        st.session_state["ml_target"]        = target_label
                        st.session_state["ml_unit"]          = cfg["unit"]
                        st.session_state["ml_features_used"] = selected_features
                        st.session_state["ml_interp"]        = interpret_results(results, target_label)
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        if "ml_results" not in st.session_state:
            st.info("Configure your model above and click **Run Models** to see results.")
        else:
            results  = st.session_state["ml_results"]
            fnames   = st.session_state["ml_feature_names"]
            tgt      = st.session_state["ml_target"]
            unit     = st.session_state["ml_unit"]
            lr       = results["Linear Regression"]
            rf       = results["Random Forest"]
            best_name= "Random Forest" if rf["r2"] >= lr["r2"] else "Linear Regression"

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Model Performance Summary")
            k1, k2, k3, k4 = st.columns(4)
            with k1: kpi_card("bi-trophy-fill",     "Best Model", best_name)
            with k2: kpi_card("bi-bullseye",         "Best R²",   str(max(lr["r2"], rf["r2"])))
            with k3: kpi_card("bi-rulers",           "LR MAE",    f"{lr['mae']} {unit}")
            with k4: kpi_card("bi-rulers",           "RF MAE",    f"{rf['mae']} {unit}")

            st.markdown("<br>", unsafe_allow_html=True)
            sub_lr, sub_rf, sub_cmp = st.tabs([
                "  Linear Regression  ", "  Random Forest  ", "  Comparison & Insights  "
            ])
            with sub_lr:  _render_model_tab(lr, "Linear Regression", fnames, unit, TEAL)
            with sub_rf:  _render_model_tab(rf, "Random Forest",     fnames, unit, PURPLE)
            with sub_cmp: _render_comparison(lr, rf, tgt, unit)

    # ══════════════════════════════════════════════════════════════════════════
    # UNSUPERVISED
    # ══════════════════════════════════════════════════════════════════════════
    with main_unsup:
        st.markdown("<br>", unsafe_allow_html=True)
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
        else:
            opt_k    = st.session_state["cluster_opt_k"]
            analysis = st.session_state["cluster_analysis"]
            profile  = st.session_state["cluster_profile"]
            cdf      = st.session_state["cluster_cdf"]
            fcols    = st.session_state["cluster_fcols"]

            # Optimal K card
            st.markdown(
                f"<div style='background:#f0faf8;border-left:4px solid {TEAL};border-radius:8px;"
                f"padding:14px 20px;margin-bottom:20px;font-size:14px;color:{NAVY};'>"
                f"<i class='bi bi-search' style='color:{TEAL};margin-right:8px;'></i>"
                f"Based on {analysis['method']}, the optimal number of customer segments is "
                f"<b>{opt_k} clusters</b>."
                f"</div>",
                unsafe_allow_html=True
            )

            # Segment overview cards
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
                        f"<div style='font-size:11px;color:#888;text-transform:uppercase;"
                        f"letter-spacing:1px;'>Cluster {idx}</div>"
                        f"<div style='font-weight:700;color:{NAVY};font-size:13px;"
                        f"margin:6px 0;'>{label}</div>"
                        f"<div style='font-size:24px;font-weight:800;color:{color};'>{count}</div>"
                        f"<div style='font-size:11px;color:#aaa;'>customers</div>"
                        f"<div style='font-size:12px;color:#555;margin-top:6px;'>Avg Sales: {sales}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Scatter
            section_header("Customer Segments — Sales vs Profit")
            if "Sales" in cdf.columns and "Profit" in cdf.columns:
                fig = go.Figure()
                for i in range(opt_k):
                    seg   = cdf[cdf["Cluster"] == i]
                    label = profile.loc[i, "Label"] if i in profile.index else f"Cluster {i}"
                    fig.add_trace(go.Scatter(
                        x=seg["Sales"], y=seg["Profit"], mode="markers",
                        marker=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], size=7, opacity=0.7),
                        name=f"Cluster {i}: {label}"
                    ))
                layout = _chart_layout("Sales vs Profit by Segment", legend_below=True)
                layout["xaxis"]["title"] = "Total Sales ($)"
                layout["yaxis"]["title"] = "Total Profit ($)"
                layout["margin"] = dict(l=60, r=40, t=60, b=100)
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            # Profile table
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Cluster Profile Summary")
            st.dataframe(profile.reset_index(), use_container_width=True)

            # Business interpretation
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Business Interpretation")

            cards = ""
            for i, (idx, row) in enumerate(profile.iterrows()):
                color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                label = row.get("Label", f"Segment {i+1}")
                sales = f"${row['Sales']:,.0f}" if "Sales" in row else "N/A"
                profit= f"${row['Profit']:,.0f}" if "Profit" in row else "N/A"
                disc  = f"{row['Discount']*100:.1f}%" if "Discount" in row else "N/A"
                count = int(row.get("Customer Count", 0))

                action = {
                    "High-Value Customers":      "Prioritise retention — offer loyalty rewards and premium service.",
                    "Growth Potential Customers": "Invest in upselling — targeted promotions can convert them to high-value.",
                    "Low-Value Customers":        "Reduce acquisition cost — focus on selective reactivation campaigns.",
                    "Mid-Tier Customers":         "Monitor and engage — consistent outreach can grow this segment.",
                }.get(label, "Analyse behaviour and tailor strategy accordingly.")

                cards += f"""
                <div style='background:#fff;border-top:4px solid {color};border-radius:10px;
                            padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                    <div style='font-weight:700;color:{NAVY};font-size:14px;margin-bottom:6px;'>
                        Cluster {idx} — {label}
                    </div>
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
                f"<div style='display:grid;grid-template-columns:repeat(2,1fr);"
                f"gap:1rem;margin-top:0.5rem;'>{cards}</div>",
                unsafe_allow_html=True
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_model_tab(res, model_name, feature_names, unit, color):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("bi-calculator",      "MAE",    f"{res['mae']} {unit}")
    with c2: kpi_card("bi-calculator",      "RMSE",   f"{res['rmse']} {unit}")
    with c3: kpi_card("bi-graph-up",        "R²",     str(res["r2"]))
    with c4: kpi_card("bi-arrow-repeat",    "CV R²",  str(res["cv_r2_mean"]))
    with c5: kpi_card("bi-plus-slash-minus","CV Std", f"±{res['cv_r2_std']}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        section_header("Actual vs Predicted")
        y_test, y_pred = res["y_test"], res["y_pred"]
        mn = float(min(y_test.min(), y_pred.min()))
        mx = float(max(y_test.max(), y_pred.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                  marker=dict(color=color, opacity=0.6, size=6),
                                  name="Predictions"))
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                  line=dict(color=CORAL, dash="dash", width=2),
                                  name="Perfect Prediction"))
        layout = _chart_layout(f"{model_name} — Actual vs Predicted", legend_below=True)
        layout["xaxis"]["title"] = f"Actual ({unit})"
        layout["yaxis"]["title"] = f"Predicted ({unit})"
        layout["margin"] = dict(l=50, r=30, t=60, b=80)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section_header("Feature Importance")
        fi_df = get_feature_importance(res["model"], feature_names)
        if not fi_df.empty:
            fig2 = go.Figure(go.Bar(
                x=fi_df["Importance %"], y=fi_df["Feature"], orientation="h",
                marker=dict(color=fi_df["Importance %"],
                            colorscale=[[0, "#b2dfdb"], [1, color]], showscale=False),
                text=[f"{v:.1f}%" for v in fi_df["Importance %"]], textposition="outside"
            ))
            layout2 = _chart_layout(f"{model_name} — Feature Importance")
            layout2["margin"] = dict(l=160, r=60, t=60, b=40)
            layout2["xaxis"]["title"] = "Importance (%)"
            layout2["yaxis"]["autorange"] = "reversed"
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Feature importance not available.")


def _render_comparison(lr, rf, target_label, unit):
    section_header("Side-by-Side Metrics")
    metrics = ["MAE", "RMSE", "R²", "CV R²", "CV Std"]
    lr_vals = [lr["mae"], lr["rmse"], lr["r2"], lr["cv_r2_mean"], lr["cv_r2_std"]]
    rf_vals = [rf["mae"], rf["rmse"], rf["r2"], rf["cv_r2_mean"], rf["cv_r2_std"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Linear Regression", x=metrics, y=lr_vals,
                          marker_color=TEAL, text=[str(v) for v in lr_vals], textposition="outside"))
    fig.add_trace(go.Bar(name="Random Forest", x=metrics, y=rf_vals,
                          marker_color=PURPLE, text=[str(v) for v in rf_vals], textposition="outside"))
    layout = _chart_layout("Linear Regression vs Random Forest", legend_below=True)
    layout["barmode"] = "group"
    layout["margin"]  = dict(l=40, r=40, t=60, b=80)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Business Insights")

    best_model = "Random Forest" if rf["r2"] >= lr["r2"] else "Linear Regression"
    best       = rf if rf["r2"] >= lr["r2"] else lr
    color_q    = GREEN if best["r2"] >= 0.75 else (AMBER if best["r2"] >= 0.5 else CORAL)
    quality    = ("Strong generalisation." if best["r2"] >= 0.75 else
                  "Moderate — treat as estimates." if best["r2"] >= 0.5 else
                  "Limited — consider more features.")
    rec        = ("Reliable for planning." if best["r2"] >= 0.75 else
                  "Validate before acting." if best["r2"] >= 0.5 else
                  "Collect more data to improve accuracy.")

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;'>
      <div style='background:#fff;border-top:4px solid {TEAL};border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
        <i class='bi bi-trophy-fill' style='color:{TEAL};font-size:20px;'></i>
        <div style='font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;'>Best Model</div>
        <div style='color:#555;font-size:13px;'><b>{best_model}</b> — R² = {best['r2']} ({round(best['r2']*100,1)}% variance explained).</div>
      </div>
      <div style='background:#fff;border-top:4px solid {AMBER};border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
        <i class='bi bi-bullseye' style='color:{AMBER};font-size:20px;'></i>
        <div style='font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;'>Prediction Accuracy</div>
        <div style='color:#555;font-size:13px;'>Avg error: <b>{best['mae']} {unit}</b>. RMSE: {best['rmse']} {unit}.</div>
      </div>
      <div style='background:#fff;border-top:4px solid {color_q};border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
        <i class='bi bi-graph-up-arrow' style='color:{color_q};font-size:20px;'></i>
        <div style='font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;'>Model Quality</div>
        <div style='color:#555;font-size:13px;'>CV R² {best['cv_r2_mean']} — {quality}</div>
      </div>
    </div>
    <div style='display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin-top:1rem;'>
      <div style='background:#fff;border-top:4px solid {PURPLE};border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
        <i class='bi bi-diagram-3-fill' style='color:{PURPLE};font-size:20px;'></i>
        <div style='font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;'>Key Driver</div>
        <div style='color:#555;font-size:13px;'>Focus improvements on top-ranked features for <b>{target_label}</b>.</div>
      </div>
      <div style='background:#fff;border-top:4px solid {GREEN};border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
        <i class='bi bi-lightbulb-fill' style='color:{GREEN};font-size:20px;'></i>
        <div style='font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;'>Recommendation</div>
        <div style='color:#555;font-size:13px;'>Use <b>{best_model}</b>. {rec}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)