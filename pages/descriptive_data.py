import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.style import apply_style, page_header, section_header, kpi_card

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL    = "#17a589"
TEAL2   = "#1abc9c"
TEAL3   = "#a8d8d0"
CORAL   = "#e07b54"
NAVY    = "#1a5276"
BLUE    = "#2e86c1"
PURPLE  = "#7d3c98"
GOLD    = "#d4ac0d"
PALETTE = [TEAL, CORAL, BLUE, GOLD, PURPLE, "#27ae60", "#e74c3c", "#f39c12"]

BASE = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Inter", size=12, color="#2c3e50"),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
)

def _layout(title, height=380, legend_below=False):
    leg = dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5) if legend_below else dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5)
    return {**BASE,
            "title": dict(text=f"<b>{title}</b>", x=0, font=dict(size=15, color=NAVY, family="Inter")),
            "height": height,
            "margin": dict(t=60, b=60, l=50, r=30),
            "legend": leg}

def _fmt(v):
    if pd.isna(v): return "N/A"
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

def _fmt_n(v):
    if pd.isna(v): return "N/A"
    if abs(v) >= 1_000_000: return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"{v/1_000:.1f}K"
    return f"{v:,.0f}"

def _get_df():
    return st.session_state.get("cleaned_df", st.session_state.get("raw_df")).copy()

def _prep(df):
    for c in ["Order Date","Ship Date"]:
        if c in df.columns: df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["Sales","Profit","Quantity","Discount"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _graph_filter(key_prefix, df, show_region=True, show_category=True,
                  show_segment=False, show_shipmode=False, show_date=False):
    """
    Renders a small per-graph filter expander.
    Returns a filtered copy of df based on local selections.
    """
    with st.expander("⚙ Graph Filters", expanded=False):
        cols_to_show = sum([show_region, show_category, show_segment, show_shipmode])
        cols_to_show = max(cols_to_show, 1)
        gcols = st.columns(cols_to_show)
        idx = 0
        local_region = local_cat = local_seg = local_ship = "All"

        if show_region and "Region" in df.columns:
            with gcols[idx]:
                opts = ["All"] + sorted(df["Region"].dropna().unique().tolist())
                local_region = st.selectbox("Region", opts, key=f"{key_prefix}_region")
            idx += 1

        if show_category and "Category" in df.columns:
            with gcols[idx]:
                opts = ["All"] + sorted(df["Category"].dropna().unique().tolist())
                local_cat = st.selectbox("Category", opts, key=f"{key_prefix}_cat")
            idx += 1

        if show_segment and "Segment" in df.columns:
            with gcols[idx]:
                opts = ["All"] + sorted(df["Segment"].dropna().unique().tolist())
                local_seg = st.selectbox("Segment", opts, key=f"{key_prefix}_seg")
            idx += 1

        if show_shipmode and "Ship Mode" in df.columns:
            with gcols[idx]:
                opts = ["All"] + sorted(df["Ship Mode"].dropna().unique().tolist())
                local_ship = st.selectbox("Ship Mode", opts, key=f"{key_prefix}_ship")

        if show_date and "Order Date" in df.columns:
            min_d = df["Order Date"].min().date()
            max_d = df["Order Date"].max().date()
            date_range = st.date_input("Date Range", value=(min_d, max_d),
                                       min_value=min_d, max_value=max_d,
                                       key=f"{key_prefix}_date")
        else:
            date_range = None

    # Apply local filters
    lf = df.copy()
    if local_region != "All" and "Region" in lf.columns:    lf = lf[lf["Region"]    == local_region]
    if local_cat    != "All" and "Category" in lf.columns:  lf = lf[lf["Category"]  == local_cat]
    if local_seg    != "All" and "Segment" in lf.columns:   lf = lf[lf["Segment"]   == local_seg]
    if local_ship   != "All" and "Ship Mode" in lf.columns: lf = lf[lf["Ship Mode"] == local_ship]
    if date_range and len(date_range) == 2 and "Order Date" in lf.columns:
        lf = lf[(lf["Order Date"] >= pd.Timestamp(date_range[0])) &
                (lf["Order Date"] <= pd.Timestamp(date_range[1]))]
    return lf


def show():
    apply_style()

    st.markdown("""
    <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
    /* Style per-graph filter expanders to look compact */
    div[data-testid="stExpander"] summary p {
        font-size: 12px !important;
        color: #17a589 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSelectbox"] label {
        font-weight: 700 !important;
        font-size: 12px !important;
        color: #1a5276 !important;
    }
    div[data-testid="stDateInput"] label,
    div[data-testid="stSlider"] label {
        font-weight: 700 !important;
        font-size: 13px !important;
        color: #1a5276 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    page_header(
        "Descriptive Analysis",
        "Explore sales performance, profitability trends, and customer behaviour across your dataset."
    )

    if "raw_df" not in st.session_state:
        st.warning("⚠️ No dataset loaded. Please upload your data in the Data Collection page first.")
        return

    df = _prep(_get_df())

    # ── Global filter bar ─────────────────────────────────────────────────────
    with st.expander("⚙  Dashboard Filters", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            regions = ["All"] + sorted(df["Region"].dropna().unique().tolist()) if "Region" in df.columns else ["All"]
            region  = st.selectbox("Region", regions, key="f_region")
        with fc2:
            cats = ["All"] + sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else ["All"]
            category = st.selectbox("Category", cats, key="f_cat")
        with fc3:
            segs = ["All"] + sorted(df["Segment"].dropna().unique().tolist()) if "Segment" in df.columns else ["All"]
            segment = st.selectbox("Segment", segs, key="f_seg")
        with fc4:
            ships = ["All"] + sorted(df["Ship Mode"].dropna().unique().tolist()) if "Ship Mode" in df.columns else ["All"]
            ship_mode = st.selectbox("Ship Mode", ships, key="f_ship")

        if "Order Date" in df.columns:
            min_d = df["Order Date"].min().date()
            max_d = df["Order Date"].max().date()
            dc1, dc2 = st.columns([3,1])
            with dc1:
                date_range = st.date_input("Date Range", value=(min_d, max_d),
                                           min_value=min_d, max_value=max_d, key="f_date")
            with dc2:
                st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                if st.button("↺ Reset All", key="reset_filters", use_container_width=True):
                    for k in ["f_region","f_cat","f_seg","f_ship","f_date","f_topn","f_metric","f_trend"]:
                        if k in st.session_state: del st.session_state[k]
                    st.rerun()
        else:
            date_range = None
            if st.button("↺ Reset All", key="reset_filters"):
                for k in ["f_region","f_cat","f_seg","f_ship","f_topn","f_metric","f_trend"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()

    # ── Visualization Parameters ──────────────────────────────────────────────
    with st.expander("⚙  Visualization Parameters", expanded=False):
        pc1, pc2, pc3, pc4 = st.columns([3, 3, 3, 1.5])
        with pc1:
            top_n = st.slider("Top N Sub-Categories", min_value=3, max_value=17,
                              value=10, step=1, key="f_topn")
        with pc2:
            metric = st.selectbox("Primary Metric", options=["Sales", "Profit", "Quantity"], key="f_metric")
        with pc3:
            trend_type = st.selectbox("Trend Chart Type", options=["Line", "Bar"], key="f_trend")
        with pc4:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            if st.button("↺  Reset", key="reset_params", use_container_width=True):
                for k in ["f_topn", "f_metric", "f_trend"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()

    # Apply global filters
    f = df.copy()
    if region    != "All" and "Region"    in df.columns: f = f[f["Region"]    == region]
    if category  != "All" and "Category"  in df.columns: f = f[f["Category"]  == category]
    if segment   != "All" and "Segment"   in df.columns: f = f[f["Segment"]   == segment]
    if ship_mode != "All" and "Ship Mode" in df.columns: f = f[f["Ship Mode"] == ship_mode]
    if date_range and len(date_range) == 2 and "Order Date" in df.columns:
        f = f[(f["Order Date"] >= pd.Timestamp(date_range[0])) &
              (f["Order Date"] <= pd.Timestamp(date_range[1]))]

    if f.empty:
        st.warning("No data matches the selected filters. Please adjust your selections.")
        return

    active = [v for v in [region, category, segment, ship_mode] if v != "All"]
    if active:
        st.markdown(
            f"<div style='background:#e8f8f5; border:1.5px solid #17a589; border-radius:8px; "
            f"padding:8px 14px; font-size:13px; color:#1a5276; margin-bottom:12px;'>"
            f"<i class='bi bi-funnel-fill' style='margin-right:6px;'></i>"
            f"<b>Active filters:</b> {' · '.join(active)} &nbsp;|&nbsp; "
            f"<b>{len(f):,}</b> records shown</div>",
            unsafe_allow_html=True
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "  Sales Dashboard",
        "  Profit Dashboard",
        "  Customer & Product"
    ])

    st.markdown("""
    <style>
    button[data-baseweb="tab"]:nth-child(1)::before { content: "\\F1A7"; font-family: "bootstrap-icons"; margin-right: 6px; }
    button[data-baseweb="tab"]:nth-child(2)::before { content: "\\F57E"; font-family: "bootstrap-icons"; margin-right: 6px; }
    button[data-baseweb="tab"]:nth-child(3)::before { content: "\\F4C0"; font-family: "bootstrap-icons"; margin-right: 6px; }
    button[data-baseweb="tab"] { font-size: 14px !important; font-weight: 600 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #17a589 !important; border-bottom: 3px solid #17a589 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — SALES
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        section_header("Sales Overview")

        total_sales  = f["Sales"].sum()
        total_orders = f["Order ID"].nunique() if "Order ID" in f.columns else len(f)
        avg_order    = f.groupby("Order ID")["Sales"].sum().mean() if "Order ID" in f.columns else f["Sales"].mean()
        top_region   = f.groupby("Region")["Sales"].sum().idxmax() if "Region" in f.columns else "N/A"

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi_card("bi-currency-dollar","Total Sales",    _fmt(total_sales))
        with c2: kpi_card("bi-cart3",          "Total Orders",   f"{total_orders:,}")
        with c3: kpi_card("bi-receipt",        "Avg Order Value",_fmt(avg_order))
        with c4: kpi_card("bi-geo-alt-fill",   "Top Region",     top_region)

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if "Category" in f.columns:
                lf = _graph_filter("s_cat", f, show_region=True, show_category=False, show_segment=True)
                d = lf.groupby("Category")[metric].sum().reset_index().sort_values(metric, ascending=False)
                fig = go.Figure(go.Bar(
                    x=d["Category"], y=d[metric],
                    text=d[metric].apply(_fmt), textposition="outside",
                    marker=dict(color=[TEAL, BLUE, CORAL], line_width=0)
                ))
                fig.update_layout(**_layout(f"{metric} by Category"), showlegend=False,
                    yaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f", title=""),
                    xaxis=dict(showgrid=False, title=""))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Region" in f.columns:
                lf = _graph_filter("s_reg", f, show_region=False, show_category=True, show_segment=True)
                d = lf.groupby("Region")["Sales"].sum().reset_index()
                fig = px.pie(d, names="Region", values="Sales", hole=0.48,
                             color_discrete_sequence=PALETTE)
                fig.update_traces(textposition="outside", textinfo="percent+label", pull=[0.03]*len(d))
                fig.update_layout(**_layout("Sales Distribution by Region"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "Order Date" in f.columns:
                lf = _graph_filter("s_trend", f, show_region=True, show_category=True, show_date=True)
                m = lf.set_index("Order Date").resample("ME")[metric].sum().reset_index()
                m.columns = ["Month", metric]
                fig = go.Figure()
                if trend_type == "Line":
                    fig.add_trace(go.Scatter(
                        x=m["Month"], y=m[metric], mode="lines+markers",
                        line=dict(color=TEAL, width=2.5), marker=dict(size=4),
                        fill="tozeroy", fillcolor="rgba(23,165,137,0.08)",
                        hovertemplate="<b>%{x|%b %Y}</b><br>" + metric + ": $%{y:,.0f}<extra></extra>"
                    ))
                else:
                    fig.add_trace(go.Bar(
                        x=m["Month"], y=m[metric], marker_color=TEAL,
                        hovertemplate="<b>%{x|%b %Y}</b><br>" + metric + ": $%{y:,.0f}<extra></extra>"
                    ))
                fig.update_layout(**_layout(f"Monthly {metric} Trend ({trend_type} Chart)"), showlegend=False,
                    yaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
                    xaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            if "Ship Mode" in f.columns:
                lf = _graph_filter("s_ship", f, show_region=True, show_category=True, show_segment=True)
                d = lf.groupby("Ship Mode")["Sales"].sum().reset_index().sort_values("Sales")
                fig = go.Figure(go.Bar(
                    x=d["Sales"], y=d["Ship Mode"], orientation="h",
                    text=d["Sales"].apply(_fmt), textposition="outside",
                    marker=dict(color=d["Sales"], colorscale=[[0,TEAL3],[1,TEAL]],
                                showscale=False, line_width=0)
                ))
                fig.update_layout(**_layout("Sales by Shipping Mode"), showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
                    yaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True)

        if "Sub-Category" in f.columns:
            lf = _graph_filter("s_subcat", f, show_region=True, show_category=True, show_segment=True)
            d = lf.groupby("Sub-Category")[metric].sum().reset_index().sort_values(metric, ascending=False).head(top_n)
            fig = go.Figure(go.Bar(
                x=d["Sub-Category"], y=d[metric],
                text=d[metric].apply(_fmt), textposition="outside",
                marker=dict(color=d[metric], colorscale=[[0,TEAL3],[1,NAVY]],
                            showscale=False, line_width=0)
            ))
            fig.update_layout(**_layout(f"Top {top_n} Sub-Categories by {metric}", height=420), showlegend=False,
                yaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
                xaxis=dict(showgrid=False, tickangle=-30))
            st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — PROFIT
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        section_header("Profit Overview")

        total_profit = f["Profit"].sum()
        margin       = (total_profit / f["Sales"].sum() * 100) if f["Sales"].sum() > 0 else 0
        avg_disc     = f["Discount"].mean() * 100 if "Discount" in f.columns else 0
        loss_count   = int((f["Profit"] < 0).sum())

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi_card("bi-graph-up",           "Total Profit",  _fmt(total_profit))
        with c2: kpi_card("bi-percent",            "Profit Margin", f"{margin:.1f}%")
        with c3: kpi_card("bi-tag-fill",           "Avg Discount",  f"{avg_disc:.1f}%")
        with c4: kpi_card("bi-exclamation-circle", "Loss Records",  f"{loss_count:,}")

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if "Category" in f.columns:
                lf = _graph_filter("p_cat", f, show_region=True, show_category=False, show_segment=True)
                d = lf.groupby("Category")["Profit"].sum().reset_index().sort_values("Profit", ascending=False)
                fig = go.Figure(go.Bar(
                    x=d["Category"], y=d["Profit"],
                    text=d["Profit"].apply(_fmt), textposition="outside",
                    marker=dict(color=[TEAL if v>=0 else CORAL for v in d["Profit"]], line_width=0)
                ))
                fig.update_layout(**_layout("Profit by Category"), showlegend=False,
                    yaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
                    xaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Region" in f.columns:
                lf = _graph_filter("p_reg", f, show_region=False, show_category=True, show_segment=True)
                d = lf.groupby("Region")["Profit"].sum().reset_index()
                fig = px.pie(d, names="Region", values="Profit", hole=0.48,
                             color_discrete_sequence=PALETTE)
                fig.update_traces(textposition="outside", textinfo="percent+label", pull=[0.03]*len(d))
                fig.update_layout(**_layout("Profit by Region"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "Order Date" in f.columns:
                lf = _graph_filter("p_trend", f, show_region=True, show_category=True, show_date=True)
                m = lf.set_index("Order Date").resample("ME")["Profit"].sum().reset_index()
                m.columns = ["Month","Profit"]
                fig = go.Figure(go.Bar(
                    x=m["Month"], y=m["Profit"],
                    marker_color=[TEAL if v>=0 else CORAL for v in m["Profit"]],
                    hovertemplate="<b>%{x|%b %Y}</b><br>Profit: $%{y:,.0f}<extra></extra>"
                ))
                fig.update_layout(**_layout("Monthly Profit Trend"), showlegend=False,
                    yaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
                    xaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            if "Discount" in f.columns:
                lf = _graph_filter("p_disc", f, show_region=True, show_category=True, show_segment=True)
                s = lf.sample(min(600,len(lf)), random_state=42)
                fig = px.scatter(s, x="Discount", y="Profit",
                                 color="Category" if "Category" in s.columns else None,
                                 color_discrete_sequence=PALETTE, opacity=0.6)
                fig.update_layout(**_layout("Discount vs Profit", legend_below=True),
                    xaxis=dict(tickformat=".0%", showgrid=True, gridcolor="#f0f4f8", title="Discount"),
                    yaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f", title="Profit"))
                st.plotly_chart(fig, use_container_width=True)

        if "Sub-Category" in f.columns:
            lf = _graph_filter("p_subcat", f, show_region=True, show_category=True, show_segment=True)
            d = lf.groupby("Sub-Category")["Profit"].sum().reset_index().sort_values("Profit")
            fig = go.Figure(go.Bar(
                x=d["Profit"], y=d["Sub-Category"], orientation="h",
                text=d["Profit"].apply(_fmt), textposition="outside",
                marker=dict(color=[CORAL if v<0 else TEAL for v in d["Profit"]], line_width=0),
                hovertemplate="<b>%{y}</b><br>Profit: $%{x:,.0f}<extra></extra>"
            ))
            fig.add_vline(x=0, line_color="#bdc3c7", line_width=1.5)
            fig.update_layout(**_layout("Profit by Sub-Category  ·  Green = Profitable   Coral = Loss", height=460),
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
                yaxis=dict(showgrid=False, autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — CUSTOMER & PRODUCT
    # ═════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        section_header("Customer & Product Overview")

        total_cust   = f["Customer ID"].nunique() if "Customer ID" in f.columns else 0
        repeat_cust  = (f.groupby("Customer ID")["Order ID"].nunique()>1).sum() if "Customer ID" in f.columns and "Order ID" in f.columns else 0
        total_prod   = f["Product ID"].nunique() if "Product ID" in f.columns else 0
        top_seg      = f.groupby("Segment")["Sales"].sum().idxmax() if "Segment" in f.columns else "N/A"

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi_card("bi-people-fill",  "Unique Customers",f"{total_cust:,}")
        with c2: kpi_card("bi-person-check", "Repeat Customers",f"{repeat_cust:,}")
        with c3: kpi_card("bi-box-seam",     "Unique Products", f"{total_prod:,}")
        with c4: kpi_card("bi-star-fill",    "Top Segment",     top_seg)

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if "Segment" in f.columns:
                lf = _graph_filter("c_seg", f, show_region=True, show_category=True, show_segment=False)
                d = lf.groupby("Segment")["Sales"].sum().reset_index()
                fig = px.pie(d, names="Segment", values="Sales", hole=0.48,
                             color_discrete_sequence=PALETTE)
                fig.update_traces(textposition="outside", textinfo="percent+label", pull=[0.03]*len(d))
                fig.update_layout(**_layout("Sales by Customer Segment"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Segment" in f.columns and "Category" in f.columns:
                lf = _graph_filter("c_segcat", f, show_region=True, show_segment=True, show_category=False)
                d = lf.groupby(["Segment","Category"])["Sales"].sum().reset_index()
                fig = px.bar(d, x="Segment", y="Sales", color="Category",
                             barmode="group", color_discrete_sequence=PALETTE)
                fig.update_traces(marker_line_width=0)
                fig.update_layout(**_layout("Sales by Segment & Category", legend_below=True),
                    yaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
                    xaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "Product Name" in f.columns:
                lf = _graph_filter("c_prod", f, show_region=True, show_category=True, show_segment=True)
                d = (lf.groupby("Product Name")["Sales"].sum()
                     .nlargest(10).reset_index().sort_values("Sales"))
                d["Short"] = d["Product Name"].str[:28] + "…"
                fig = go.Figure(go.Bar(
                    x=d["Sales"], y=d["Short"], orientation="h",
                    text=d["Sales"].apply(_fmt), textposition="outside",
                    marker=dict(color=d["Sales"], colorscale=[[0,TEAL3],[1,NAVY]],
                                showscale=False, line_width=0),
                    hovertext=d["Product Name"], hoverinfo="text+x"
                ))
                fig.update_layout(**_layout("Top 10 Products by Sales", height=420), showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor="#f0f4f8", tickformat="$,.0f"),
                    yaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            if "Category" in f.columns and "Quantity" in f.columns:
                lf = _graph_filter("c_qty", f, show_region=True, show_category=False, show_segment=True)
                d = lf.groupby("Category")["Quantity"].sum().reset_index().sort_values("Quantity", ascending=False)
                fig = go.Figure(go.Bar(
                    x=d["Category"], y=d["Quantity"],
                    text=d["Quantity"].apply(_fmt_n), textposition="outside",
                    marker=dict(color=[BLUE, TEAL, CORAL], line_width=0)
                ))
                fig.update_layout(**_layout("Units Sold by Category"), showlegend=False,
                    yaxis=dict(showgrid=True, gridcolor="#f0f4f8"),
                    xaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True)

        if "Region" in f.columns and "Category" in f.columns:
            lf = _graph_filter("c_heat", f, show_region=False, show_category=False, show_segment=True, show_shipmode=True)
            pivot = lf.pivot_table(values="Sales", index="Region",
                                   columns="Category", aggfunc="sum").fillna(0)
            mean_val = pivot.values.mean()
            fig = px.imshow(pivot,
                            color_continuous_scale=[[0,"#e8f8f5"],[0.5,TEAL3],[1,NAVY]],
                            aspect="auto", text_auto=False)
            for i, row in enumerate(pivot.index):
                for j, col_name in enumerate(pivot.columns):
                    v = pivot.loc[row, col_name]
                    fig.add_annotation(x=j, y=i,
                        text=f"${v/1000:.1f}K",
                        showarrow=False,
                        font=dict(size=13, color="white" if v > mean_val else NAVY))
            fig.update_layout(**_layout("Sales Heatmap — Region × Category", height=320),
                coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)