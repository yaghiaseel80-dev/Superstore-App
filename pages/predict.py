import streamlit as st
import pandas as pd
from src.style import apply_style, page_header, section_header
from src.modeling import predict_single, predict_cluster, TARGET_CONFIG

TEAL   = "#17a589"
NAVY   = "#1a5276"
CORAL  = "#e74c3c"
AMBER  = "#d4ac0d"
PURPLE = "#8e44ad"
GREEN  = "#27ae60"
BLUE   = "#2e86c1"

CLUSTER_COLORS = [TEAL, PURPLE, AMBER, BLUE, CORAL, GREEN]


def _fmt_result(value, unit):
    if unit == "days": return f"{max(0, round(value, 1))} days"
    if unit == "%":    return f"{max(0.0, min(1.0, value)):.3f}  ({max(0.0, min(1.0, value))*100:.1f}%)"
    return f"{value:,.2f}"


def show():
    apply_style()
    page_header("Predict",
                "Apply trained models to generate live predictions for new orders or customers.")

    has_supervised = "ml_results"    in st.session_state
    has_clustering = "cluster_model" in st.session_state

    if not has_supervised and not has_clustering:
        st.markdown(
            f"<div style='background:#fff8f0;border:1.5px solid {CORAL};border-radius:10px;"
            f"padding:20px 24px;margin-top:20px;text-align:center;'>"
            f"<i class='bi bi-exclamation-triangle-fill' style='font-size:28px;color:{CORAL};'></i><br><br>"
            f"<b style='font-size:15px;color:{NAVY};'>No trained models found.</b><br>"
            f"<span style='font-size:13px;color:#555;'>Please go to the Machine Learning page, "
            f"train your models first, then come back here.</span></div>",
            unsafe_allow_html=True)
        return

    # ── What to predict ───────────────────────────────────────────────────────
    section_header("What would you like to predict?")
    options = list(TARGET_CONFIG.keys()) if has_supervised else []
    if has_clustering:
        options.append("Customer Segment")

    predict_type = st.radio("", options=options, horizontal=True,
                             label_visibility="collapsed", key="predict_type_radio")
    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SUPERVISED
    # ══════════════════════════════════════════════════════════════════════════
    if predict_type in TARGET_CONFIG:
        if not has_supervised:
            st.warning("Please run the Supervised Learning models first.")
            return

        results       = st.session_state["ml_results"]
        feature_names = st.session_state["ml_feature_names"]
        target_used   = st.session_state["ml_target"]
        unit          = st.session_state.get("ml_unit", "")
        cfg           = TARGET_CONFIG.get(target_used, {})
        features_used = cfg.get("features", [])
        df            = st.session_state.get("cleaned_df", st.session_state.get("raw_df"))

        if predict_type != target_used:
            st.warning(
                f"Your trained model predicts **{target_used}**, not **{predict_type}**. "
                f"Go to Machine Learning and re-run for **{predict_type}**.")
            return

        st.markdown(
            f"<div style='background:#f0faf8;border-left:4px solid {TEAL};"
            f"border-radius:8px;padding:12px 18px;margin-bottom:20px;font-size:14px;'>"
            f"<i class='bi bi-cpu-fill' style='color:{TEAL};'></i>&nbsp;"
            f"<b>Predicting:</b> {target_used} &nbsp;|&nbsp;"
            f"<b>Features:</b> {', '.join(features_used)}</div>",
            unsafe_allow_html=True)

        lr_r2     = results["Linear Regression"]["r2"]
        rf_r2     = results["Random Forest"]["r2"]
        best_name = "Random Forest" if rf_r2 >= lr_r2 else "Linear Regression"
        best_r2   = max(lr_r2, rf_r2)
        color_q   = GREEN if best_r2 >= 0.75 else (AMBER if best_r2 >= 0.5 else CORAL)
        quality   = "strong" if best_r2 >= 0.75 else ("moderate" if best_r2 >= 0.5 else "limited")

        section_header("Model Recommendation")
        st.markdown(
            f"<div style='background:#fffbf0;border:1.5px solid {color_q};border-radius:10px;"
            f"padding:12px 18px;margin-bottom:16px;font-size:14px;'>"
            f"<i class='bi bi-lightbulb-fill' style='color:{color_q};margin-right:6px;'></i>"
            f"<b style='color:{NAVY};'>Recommended:</b> Use <b>{best_name}</b> — "
            f"R² = {best_r2} ({quality} predictive power). "
            f"LR R² = {lr_r2} &nbsp;|&nbsp; RF R² = {rf_r2}</div>",
            unsafe_allow_html=True)

        section_header("Select Model")
        model_choice = st.radio("", ["Linear Regression", "Random Forest", "Both (compare)"],
                                 index=["Linear Regression", "Random Forest"].index(best_name),
                                 horizontal=True, label_visibility="collapsed",
                                 key="predict_model_choice")
        st.markdown("<br>", unsafe_allow_html=True)

        section_header("Enter Order Details")

        def get_options(col):
            if df is not None and col in df.columns:
                return sorted(df[col].dropna().unique().tolist())
            return []

        CAT_OPTIONS = {
            "Category":     get_options("Category")     or ["Furniture", "Office Supplies", "Technology"],
            "Sub-Category": get_options("Sub-Category") or ["Bookcases", "Chairs", "Labels"],
            "Region":       get_options("Region")       or ["East", "West", "Central", "South"],
            "Segment":      get_options("Segment")      or ["Consumer", "Corporate", "Home Office"],
            "Ship Mode":    get_options("Ship Mode")    or ["First Class", "Second Class", "Standard Class", "Same Day"],
        }

        input_dict = {}
        inp_cols   = st.columns(3)
        col_idx    = 0

        for feature in features_used:
            if feature == target_used:
                continue
            with inp_cols[col_idx % 3]:
                if feature in CAT_OPTIONS:
                    input_dict[feature] = st.selectbox(f"**{feature}**",
                                                        options=CAT_OPTIONS[feature],
                                                        key=f"pred_input_{feature}")
                elif feature == "Discount":
                    input_dict[feature] = st.number_input(f"**{feature}** (0.0–1.0)",
                                                           min_value=0.0, max_value=1.0,
                                                           value=0.2, step=0.05,
                                                           key=f"pred_input_{feature}")
                elif feature == "Quantity":
                    input_dict[feature] = st.number_input(f"**{feature}** (units)",
                                                           min_value=1, max_value=100,
                                                           value=3, step=1,
                                                           key=f"pred_input_{feature}")
                else:
                    input_dict[feature] = st.number_input(f"**{feature}**",
                                                           value=0.0, step=1.0,
                                                           key=f"pred_input_{feature}")
            col_idx += 1

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Generate Prediction", type="primary", key="predict_btn"):
            try:
                models_to_run = (["Linear Regression", "Random Forest"]
                                 if model_choice == "Both (compare)" else [model_choice])
                predictions = {m: predict_single(results[m]["model"], feature_names, input_dict)
                               for m in models_to_run}

                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Prediction Result")

                if len(predictions) == 1:
                    mname = list(predictions.keys())[0]
                    fmt   = _fmt_result(predictions[mname], unit)
                    r2    = results[mname]["r2"]
                    mae   = results[mname]["mae"]
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#f0faf8 0%,#ffffff 100%);
                                border:2px solid {TEAL};border-radius:16px;padding:32px;
                                text-align:center;box-shadow:0 4px 16px rgba(23,165,137,0.15);">
                        <i class="bi bi-bullseye" style="font-size:36px;color:{TEAL};"></i>
                        <div style="font-size:13px;color:#888;margin-top:12px;">{mname} predicts</div>
                        <div style="font-size:44px;font-weight:800;color:{NAVY};margin:8px 0;">{fmt}</div>
                        <div style="font-size:13px;color:#555;">for <b>{target_used}</b></div>
                        <div style="font-size:12px;color:#aaa;margin-top:10px;">R² = {r2} &nbsp;|&nbsp; MAE = {mae} {unit}</div>
                    </div>""", unsafe_allow_html=True)
                    _supervised_interpretation(target_used, predictions[mname], unit, color_q)

                else:
                    lr_val  = predictions["Linear Regression"]
                    rf_val  = predictions["Random Forest"]
                    lr_fmt  = _fmt_result(lr_val, unit)
                    rf_fmt  = _fmt_result(rf_val, unit)
                    rec_fmt = lr_fmt if best_name == "Linear Regression" else rf_fmt
                    lr_b    = f"3px solid {TEAL}"   if best_name == "Linear Regression" else "1.5px solid #ccc"
                    rf_b    = f"3px solid {PURPLE}" if best_name == "Random Forest"     else "1.5px solid #ccc"
                    lr_bg   = "#f0faf8" if best_name == "Linear Regression" else "#fafafa"
                    rf_bg   = "#f5f0ff" if best_name == "Random Forest"     else "#fafafa"
                    lr_rec  = f"<div style='font-size:11px;color:{TEAL};font-weight:600;margin-top:4px;'>★ Recommended</div>" if best_name == "Linear Regression" else ""
                    rf_rec  = f"<div style='font-size:11px;color:{PURPLE};font-weight:600;margin-top:4px;'>★ Recommended</div>" if best_name == "Random Forest" else ""

                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                        <div style="background:{lr_bg};border:{lr_b};border-radius:14px;
                                    padding:24px;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.07);">
                            <i class="bi bi-graph-up" style="font-size:28px;color:{TEAL};"></i>
                            <div style="font-size:12px;color:#888;margin-top:8px;">Linear Regression</div>
                            <div style="font-size:34px;font-weight:800;color:{NAVY};margin:6px 0;">{lr_fmt}</div>
                            <div style="font-size:11px;color:#aaa;">R² = {lr_r2}</div>{lr_rec}
                        </div>
                        <div style="background:{rf_bg};border:{rf_b};border-radius:14px;
                                    padding:24px;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.07);">
                            <i class="bi bi-diagram-3-fill" style="font-size:28px;color:{PURPLE};"></i>
                            <div style="font-size:12px;color:#888;margin-top:8px;">Random Forest</div>
                            <div style="font-size:34px;font-weight:800;color:{NAVY};margin:6px 0;">{rf_fmt}</div>
                            <div style="font-size:11px;color:#aaa;">R² = {rf_r2}</div>{rf_rec}
                        </div>
                    </div>
                    <div style="background:#fffbf0;border:1.5px solid {color_q};border-radius:10px;
                                padding:14px 20px;margin-top:14px;text-align:center;">
                        <i class="bi bi-lightbulb-fill" style="color:{color_q};"></i>
                        <b style="color:{NAVY};"> Best result to trust: {rec_fmt}</b>
                        <span style="font-size:12px;color:#888;"> — from {best_name} (R² = {best_r2})</span>
                    </div>""", unsafe_allow_html=True)

                    best_pred = lr_val if best_name == "Linear Regression" else rf_val
                    _supervised_interpretation(target_used, best_pred, unit, color_q)

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("View input summary"):
                    st.dataframe(pd.DataFrame([input_dict]), use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # CLUSTERING
    # ══════════════════════════════════════════════════════════════════════════
    elif predict_type == "Customer Segment":
        if not has_clustering:
            st.warning("Please run the Customer Segmentation model first.")
            return

        km      = st.session_state["cluster_model"]
        scaler  = st.session_state["cluster_scaler"]
        profile = st.session_state["cluster_profile"]
        fcols   = st.session_state["cluster_fcols"]
        opt_k   = st.session_state["cluster_opt_k"]

        st.markdown(
            f"<div style='background:#f0faf8;border-left:4px solid {TEAL};"
            f"border-radius:8px;padding:12px 18px;margin-bottom:20px;font-size:14px;'>"
            f"<i class='bi bi-diagram-3-fill' style='color:{TEAL};'></i>&nbsp;"
            f"Enter customer details to find which of the <b>{opt_k} segments</b> they belong to."
            f"</div>",
            unsafe_allow_html=True)

        section_header("Enter Customer Details")
        c_cols  = st.columns(len(fcols))
        c_input = {}
        labels_map = {
            "Sales":    ("Total Sales ($)",         0.0,  50000.0, 1000.0, 500.0),
            "Profit":   ("Total Profit ($)",      -5000.0, 20000.0,  200.0, 100.0),
            "Discount": ("Avg Discount (0–1)",       0.0,     1.0,   0.15,  0.05),
            "Quantity": ("Total Units Ordered",      1.0,   500.0,   20.0,   5.0),
            "Orders":   ("Number of Orders",         1.0,   200.0,   10.0,   1.0),
        }

        for i, col in enumerate(fcols):
            with c_cols[i]:
                info = labels_map.get(col, (col, 0.0, 10000.0, 100.0, 10.0))
                c_input[col] = st.number_input(
                    f"**{info[0]}**",
                    min_value=float(info[1]), max_value=float(info[2]),
                    value=float(info[3]),     step=float(info[4]),
                    key=f"cluster_input_{col}")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Identify Customer Segment", type="primary", key="cluster_predict_btn"):
            try:
                cid   = predict_cluster(km, scaler, fcols, c_input)
                label = profile.loc[cid, "Label"] if cid in profile.index else f"Segment {cid}"
                color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                row   = profile.loc[cid] if cid in profile.index else {}

                avg_sales  = f"${row.get('Sales',0):,.0f}"       if "Sales"    in row else "N/A"
                avg_profit = f"${row.get('Profit',0):,.0f}"      if "Profit"   in row else "N/A"
                avg_disc   = f"{row.get('Discount',0)*100:.1f}%" if "Discount" in row else "N/A"
                count      = int(row.get("Customer Count", 0))

                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Segment Result")

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#f8fffe 0%,#ffffff 100%);
                            border:2px solid {color};border-radius:16px;padding:32px;
                            text-align:center;box-shadow:0 4px 16px rgba(0,0,0,0.1);">
                    <i class="bi bi-person-badge-fill" style="font-size:36px;color:{color};"></i>
                    <div style="font-size:13px;color:#888;margin-top:12px;">This customer belongs to</div>
                    <div style="font-size:36px;font-weight:800;color:{NAVY};margin:8px 0;">{label}</div>
                    <div style="font-size:13px;color:#555;">Cluster {cid} out of {opt_k} segments</div>
                </div>""", unsafe_allow_html=True)

                action_map = {
                    "High-Value Customers":       ("bi-star-fill",             GREEN,  "Prioritise retention — offer loyalty rewards and premium service."),
                    "Growth Potential Customers":  ("bi-arrow-up-circle-fill",  BLUE,  "Invest in upselling — targeted promotions can convert them to high-value."),
                    "Low-Value Customers":         ("bi-exclamation-circle-fill", CORAL,"Reduce acquisition cost — focus on selective reactivation campaigns."),
                    "Mid-Tier Customers":          ("bi-bar-chart-fill",        AMBER, "Monitor and engage — consistent outreach can grow this segment."),
                }
                icon, ac_color, advice = action_map.get(label, ("bi-info-circle-fill", TEAL, "Analyse behaviour and tailor strategy accordingly."))

                st.markdown(f"""
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin-top:1.2rem;">
                    <div style="background:#fff;border-top:4px solid {color};border-radius:10px;
                                padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                        <i class="bi bi-bar-chart-line-fill" style="color:{color};font-size:20px;"></i>
                        <div style="font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;">Segment Profile</div>
                        <div style="font-size:13px;color:#555;line-height:1.8;">
                            <b>{count}</b> similar customers in this segment<br>
                            Avg Sales: <b>{avg_sales}</b><br>
                            Avg Profit: <b>{avg_profit}</b><br>
                            Avg Discount: <b>{avg_disc}</b>
                        </div>
                    </div>
                    <div style="background:#fff;border-top:4px solid {ac_color};border-radius:10px;
                                padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                        <i class="bi {icon}" style="color:{ac_color};font-size:20px;"></i>
                        <div style="font-weight:700;color:{NAVY};margin:0.5rem 0 0.3rem;">Recommended Action</div>
                        <div style="font-size:13px;color:#555;">{advice}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Segment prediction failed: {e}")


def _supervised_interpretation(target, value, unit, color):
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Business Interpretation")

    if target == "Shipping Duration":
        days = max(0, round(value, 1))
        if days <= 2:
            insight = "This order is expected to ship very quickly — ideal for high-priority customers."
            action  = "Consider this Ship Mode for time-sensitive orders to improve customer satisfaction."
            icon, ic = "bi-lightning-charge-fill", GREEN
        elif days <= 4:
            insight = "Standard delivery time — most customers find this acceptable."
            action  = "No immediate action needed. Monitor for delays during peak seasons."
            icon, ic = "bi-clock-fill", AMBER
        else:
            insight = "This order is expected to take longer than average to ship."
            action  = "Consider upgrading the Ship Mode or notifying the customer proactively."
            icon, ic = "bi-exclamation-triangle-fill", CORAL

    elif target == "Discount":
        pct = max(0.0, min(1.0, value)) * 100
        if pct < 15:
            insight = "A low discount is predicted — this order type preserves profit margins well."
            action  = "Maintain pricing strategy. This is a margin-friendly order."
            icon, ic = "bi-cash-stack", GREEN
        elif pct < 35:
            insight = "A moderate discount is predicted. Profit impact is manageable but worth monitoring."
            action  = "Review whether the discount is necessary to avoid margin erosion."
            icon, ic = "bi-tag-fill", AMBER
        else:
            insight = "A high discount is predicted — this may result in reduced or negative profit."
            action  = "Reconsider the discount strategy for this order type to protect profitability."
            icon, ic = "bi-exclamation-triangle-fill", CORAL
    else:
        insight = f"Predicted {target}: {value:.2f} {unit}."
        action  = "Use this prediction to inform your business decisions."
        icon, ic = "bi-info-circle-fill", color

    st.markdown(f"""
    <div style="background:#fff;border-top:4px solid {ic};border-radius:12px;
                padding:1.4rem 1.6rem;box-shadow:0 2px 10px rgba(0,0,0,0.07);margin-top:0.5rem;">
        <i class="bi {icon}" style="color:{ic};font-size:22px;"></i>
        <div style="font-weight:700;color:#1a2940;margin:0.6rem 0 0.3rem;font-size:15px;">What this means</div>
        <div style="font-size:13px;color:#555;margin-bottom:0.8rem;">{insight}</div>
        <div style="font-weight:600;color:#1a2940;font-size:13px;">
            <i class="bi bi-arrow-right-circle-fill" style="color:{ic};"></i>&nbsp;Recommended Action
        </div>
        <div style="font-size:13px;color:#555;">{action}</div>
    </div>""", unsafe_allow_html=True)