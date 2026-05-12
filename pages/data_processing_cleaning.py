import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.style import apply_style, page_header, section_header, kpi_card
from src.cleaning import (
    get_missing_summary, impute_missing,
    detect_outliers, handle_outliers, get_outlier_stats,
    format_validate,
    remove_duplicates, add_shipping_duration, standardise_text
)


def _row_count_control(key, total_rows):
    """Renders a sleek row count selector. Returns selected number of rows."""
    st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:space-between;
                background:#f8fffe; border:1.5px solid #17a589; border-radius:10px;
                padding:10px 18px; margin-bottom:12px;'>
        <div style='display:flex; align-items:center; gap:8px;'>
            <i class='bi bi-table' style='color:#17a589; font-size:16px;'></i>
            <span style='font-weight:600; color:#1a2940; font-size:14px;'>Cleaned Data Preview</span>
            <span style='font-size:12px; color:#888;'>— {total_rows:,} rows total</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_minus, col_num, col_plus, col_spacer = st.columns([0.08, 0.18, 0.08, 0.66])

    if f"preview_rows_{key}" not in st.session_state:
        st.session_state[f"preview_rows_{key}"] = 10

    with col_minus:
        if st.button("−", key=f"minus_{key}", help="Show fewer rows"):
            st.session_state[f"preview_rows_{key}"] = max(
                5, st.session_state[f"preview_rows_{key}"] - 5)

    with col_num:
        new_val = st.number_input(
            "", min_value=5, max_value=total_rows,
            value=st.session_state[f"preview_rows_{key}"],
            step=5, key=f"num_input_{key}",
            label_visibility="collapsed"
        )
        st.session_state[f"preview_rows_{key}"] = new_val

    with col_plus:
        if st.button("+", key=f"plus_{key}", help="Show more rows"):
            st.session_state[f"preview_rows_{key}"] = min(
                total_rows, st.session_state[f"preview_rows_{key}"] + 5)

    return int(st.session_state[f"preview_rows_{key}"])


def show():
    apply_style()

    st.markdown("""
    <style>
    [data-testid="stExpander"] summary svg { display: none !important; }
    [data-testid="stExpanderToggleIcon"] { display: none !important; }
    details summary span[data-testid="stExpanderToggleIcon"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    page_header(
        "Data Processing & Cleaning",
        "Review and resolve data quality issues before analysis. Apply the measures below in sequence for best results."
    )

    if "raw_df" not in st.session_state:
        st.warning("⚠️ No dataset loaded yet. Please go to **Data Collection** and upload your file first.")
        return

    if "cleaned_df" not in st.session_state:
        st.session_state["cleaned_df"] = st.session_state["raw_df"].copy()

    if "clean_messages" not in st.session_state:
        st.session_state["clean_messages"] = {}

    df = st.session_state["cleaned_df"]

    # ── Status cards ──────────────────────────────────────────────────────────
    section_header("Current Dataset Status")
    nulls = int(df.isnull().sum().sum())
    dups  = int(df.duplicated().sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("bi-table",                "Total Rows",     f"{len(df):,}")
    with c2: kpi_card("bi-exclamation-triangle", "Missing Values", f"{nulls:,}")
    with c3: kpi_card("bi-copy",                 "Duplicate Rows", f"{dups:,}")
    with c4: kpi_card("bi-check2-circle",        "Columns",        f"{len(df.columns)}")

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    def show_message(key):
        if key in st.session_state["clean_messages"]:
            msg = st.session_state["clean_messages"][key]
            if msg["type"] == "success":  st.success(msg["text"])
            elif msg["type"] == "info":   st.info(msg["text"])
            elif msg["type"] == "warning":st.warning(msg["text"])

    def store_message(key, msg_type, text):
        st.session_state["clean_messages"][key] = {"type": msg_type, "text": text}

    # ── Measure 1 ─────────────────────────────────────────────────────────────
    with st.expander("Measure 1 — Missing Data Handling", expanded=True):
        st.markdown(
            '<p style="font-size:15px; font-weight:600; color:#1a5276; margin-bottom:12px;">'
            '<i class="bi bi-database-x" style="color:#17a589; margin-right:8px;"></i>'
            'Missing Data Handling</p>', unsafe_allow_html=True)

        show_message("missing")
        missing_summary = get_missing_summary(df)

        if missing_summary.empty:
            st.success("✅ No missing values found in the current dataset.")
        else:
            st.markdown(
                f"<p style='font-size:13px; color:#555;'>"
                f"Found <b>{missing_summary['Missing Count'].sum():,}</b> missing values "
                f"across <b>{len(missing_summary)}</b> columns.</p>", unsafe_allow_html=True)
            st.dataframe(missing_summary, use_container_width=True, hide_index=True)
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            section_header("Choose Imputation Strategy")

            strategy = st.selectbox(
                "Strategy",
                options=["drop", "mean", "median", "mode", "customer", "ffill", "constant"],
                format_func=lambda x: {
                    "drop":     "Drop rows with missing values",
                    "mean":     "Fill with column mean (numeric columns)",
                    "median":   "Fill with column median (numeric columns)",
                    "mode":     "Fill with most frequent value",
                    "customer": "Fill from Customer ID history (avg per customer)",
                    "ffill":    "Forward fill (use value from previous row)",
                    "constant": "Fill with 0 (numbers) / 'Unknown' (text)",
                }[x], key="missing_strategy")

            if st.button("Apply Missing Data Fix", key="apply_missing"):
                before_nulls = int(st.session_state["cleaned_df"].isnull().sum().sum())
                st.session_state["cleaned_df"] = impute_missing(
                    st.session_state["cleaned_df"], strategy=strategy)
                after_nulls = int(st.session_state["cleaned_df"].isnull().sum().sum())
                store_message("missing", "success",
                    f"✅ Strategy '{strategy}' applied. Fixed {before_nulls - after_nulls:,} missing values. Remaining: {after_nulls:,}")
                st.rerun()

    # ── Measure 2 ─────────────────────────────────────────────────────────────
    with st.expander("Measure 2 — Outlier Detection & Handling", expanded=False):
        st.markdown(
            '<p style="font-size:15px; font-weight:600; color:#1a5276; margin-bottom:12px;">'
            '<i class="bi bi-graph-up-arrow" style="color:#17a589; margin-right:8px;"></i>'
            'Outlier Detection & Handling</p>', unsafe_allow_html=True)

        show_message("outlier")
        section_header("Detection Method")
        method = st.radio(
            "Detection method",
            ["iqr", "zscore"],
            format_func=lambda x: {
                "iqr":    "IQR — values beyond Q1−1.5×IQR or Q3+1.5×IQR",
                "zscore": "Z-Score — values more than 3 standard deviations from the mean",
            }[x], horizontal=True, key="outlier_method", label_visibility="collapsed")

        outlier_counts = detect_outliers(df, method=method)

        if not outlier_counts:
            st.success("✅ No outliers detected with the selected method.")
        else:
            total_outliers = sum(outlier_counts.values())
            st.markdown(
                f"<p style='font-size:13px; color:#555;'>"
                f"Detected <b>{total_outliers:,}</b> outlier values across "
                f"<b>{len(outlier_counts)}</b> columns.</p>", unsafe_allow_html=True)

            stats = get_outlier_stats(df, method=method)
            if stats:
                cols_list    = [s["column"]  for s in stats]
                outlier_vals = [s["outliers"] for s in stats]
                pct_vals     = [round(s["outliers"] / s["total"] * 100, 1) for s in stats]
                labels       = [f"{v:,}  ({p}%)" for v, p in zip(outlier_vals, pct_vals)]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=outlier_vals, y=cols_list, orientation="h",
                    marker=dict(color=outlier_vals,
                                colorscale=[[0, "#f9c784"], [0.5, "#e87c5a"], [1, "#c0392b"]],
                                showscale=False),
                    text=labels, textposition="outside", cliponaxis=False))
                fig.update_layout(
                    title=dict(text="Outliers Detected per Column", x=0,
                               font=dict(size=14, color="#1a5276")),
                    xaxis=dict(title="Number of Outlier Values", showgrid=True,
                               gridcolor="#f0f4f8", zeroline=False),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(family="Inter", size=12),
                    margin=dict(t=50, b=40, l=20, r=120),
                    height=280, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            section_header("Choose Outlier Action")
            action = st.selectbox(
                "Action",
                options=["flag", "cap", "remove", "mean"],
                format_func=lambda x: {
                    "flag":   "Flag only — mark outlier rows with a new column (data unchanged)",
                    "cap":    "Cap (Winsorize) — clip extreme values to the boundary",
                    "remove": "Remove rows — delete any row containing an outlier",
                    "mean":   "Replace with mean — substitute outlier with column average",
                }[x], key="outlier_action")

            if st.button("Apply Outlier Fix", key="apply_outliers"):
                before     = len(st.session_state["cleaned_df"])
                before_out = sum(detect_outliers(st.session_state["cleaned_df"], method=method).values())
                st.session_state["cleaned_df"] = handle_outliers(
                    st.session_state["cleaned_df"], method=method, action=action)
                after     = len(st.session_state["cleaned_df"])
                after_out = sum(detect_outliers(st.session_state["cleaned_df"], method=method).values())

                if action == "flag":
                    flag_cols = [c for c in st.session_state["cleaned_df"].columns if c.endswith("_outlier_flag")]
                    msg = f"✅ Flagged outliers — added {len(flag_cols)} flag column(s): {', '.join(flag_cols)}."
                elif action == "remove":
                    msg = f"✅ Removed {before - after:,} rows. Outliers before: {before_out:,} → after: {after_out:,}"
                elif action == "cap":
                    msg = f"✅ Capped {before_out:,} outlier values. Outliers remaining: {after_out:,}"
                else:
                    msg = f"✅ Replaced {before_out:,} outliers with column mean. Remaining: {after_out:,}"
                store_message("outlier", "success", msg)
                st.rerun()

    # ── Measure 3 ─────────────────────────────────────────────────────────────
    with st.expander("Measure 3 — Format & Type Validation", expanded=False):
        st.markdown(
            '<p style="font-size:15px; font-weight:600; color:#1a5276; margin-bottom:12px;">'
            '<i class="bi bi-file-earmark-check" style="color:#17a589; margin-right:8px;"></i>'
            'Format & Type Validation</p>', unsafe_allow_html=True)
        show_message("format")
        st.markdown(
            "<p style='font-size:13px; color:#555;'>"
            "Fixes date formats, postal code zero-padding, numeric type casting, "
            "and strips whitespace from all text columns.</p>", unsafe_allow_html=True)
        if st.button("Apply Format Fixes", key="apply_format"):
            fixed_df, fixes = format_validate(st.session_state["cleaned_df"])
            st.session_state["cleaned_df"] = fixed_df
            store_message("format", "success", f"✅ {len(fixes)} fixes applied: {' | '.join(fixes)}")
            st.rerun()

    # ── Measure 4 ─────────────────────────────────────────────────────────────
    with st.expander("Measure 4 — Duplicate Row Removal", expanded=False):
        st.markdown(
            '<p style="font-size:15px; font-weight:600; color:#1a5276; margin-bottom:12px;">'
            '<i class="bi bi-subtract" style="color:#17a589; margin-right:8px;"></i>'
            'Duplicate Row Removal</p>', unsafe_allow_html=True)
        show_message("dedup")
        current_dups = int(st.session_state["cleaned_df"].duplicated().sum())
        st.markdown(
            f"<p style='font-size:13px; color:#555;'>"
            f"Current duplicate rows: <b>{current_dups:,}</b>.</p>", unsafe_allow_html=True)
        if st.button("Apply Duplicate Removal", key="apply_dedup"):
            cleaned, removed = remove_duplicates(st.session_state["cleaned_df"])
            st.session_state["cleaned_df"] = cleaned
            if removed == 0:
                store_message("dedup", "info", "ℹ️ No duplicate rows found — dataset is already clean.")
            else:
                store_message("dedup", "success",
                    f"✅ Removed {removed:,} duplicate rows. Rows remaining: {len(cleaned):,}")
            st.rerun()

    # ── Measure 5 ─────────────────────────────────────────────────────────────
    with st.expander("Measure 5 — Shipping Duration Calculation", expanded=False):
        st.markdown(
            '<p style="font-size:15px; font-weight:600; color:#1a5276; margin-bottom:12px;">'
            '<i class="bi bi-truck" style="color:#17a589; margin-right:8px;"></i>'
            'Shipping Duration Calculation</p>', unsafe_allow_html=True)
        show_message("ship")
        already_added = "Shipping Duration (Days)" in st.session_state["cleaned_df"].columns
        st.markdown(
            "<p style='font-size:13px; color:#555;'>"
            "Adds a <b>Shipping Duration (Days)</b> column = Ship Date − Order Date.</p>",
            unsafe_allow_html=True)
        if already_added:
            st.info("ℹ️ 'Shipping Duration (Days)' column already exists.")
        else:
            if st.button("Apply Shipping Duration", key="apply_ship"):
                st.session_state["cleaned_df"] = add_shipping_duration(st.session_state["cleaned_df"])
                if "Shipping Duration (Days)" in st.session_state["cleaned_df"].columns:
                    avg_days = st.session_state["cleaned_df"]["Shipping Duration (Days)"].mean()
                    store_message("ship", "success",
                        f"✅ Column added. Average shipping time: {avg_days:.1f} days.")
                else:
                    store_message("ship", "warning",
                        "⚠️ Could not calculate duration. Apply Format Fixes first.")
                st.rerun()

    # ── Measure 6 ─────────────────────────────────────────────────────────────
    with st.expander("Measure 6 — Text Standardisation", expanded=False):
        st.markdown(
            '<p style="font-size:15px; font-weight:600; color:#1a5276; margin-bottom:12px;">'
            '<i class="bi bi-fonts" style="color:#17a589; margin-right:8px;"></i>'
            'Text Standardisation</p>', unsafe_allow_html=True)
        show_message("text")
        st.markdown(
            "<p style='font-size:13px; color:#555;'>"
            "Converts Category, Sub-Category, Segment, Region, Ship Mode to Title Case.</p>",
            unsafe_allow_html=True)
        if st.button("Apply Text Standardisation", key="apply_text"):
            cleaned, fixed_cols = standardise_text(st.session_state["cleaned_df"])
            st.session_state["cleaned_df"] = cleaned
            store_message("text", "success",
                f"✅ Standardised {len(fixed_cols)} columns: {', '.join(fixed_cols)}")
            st.rerun()

    # ── Preview + Reset ───────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    cleaned_df = st.session_state["cleaned_df"]
    n_rows = _row_count_control("cleaning", len(cleaned_df))
    st.dataframe(cleaned_df.head(n_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    if st.button("↺  Reset to Original Data", key="reset_cleaning"):
        st.session_state["cleaned_df"] = st.session_state["raw_df"].copy()
        st.session_state["clean_messages"] = {}
        st.rerun()