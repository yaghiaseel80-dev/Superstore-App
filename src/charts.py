import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── Shared theme ──────────────────────────────────────────────────────────────
TEAL        = "#17a589"
CORAL       = "#e87c5a"
NAVY        = "#1a5276"
LIGHT_TEAL  = "#a8d8d0"
PALETTE     = [TEAL, CORAL, "#2980b9", "#f4a261", "#8e44ad", "#27ae60", "#e74c3c", "#f39c12"]

LAYOUT = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Inter", size=12, color="#2c3e50"),
    margin=dict(t=50, b=40, l=40, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
)


def _apply(fig, title, height=380):
    fig.update_layout(
        **LAYOUT,
        title=dict(text=title, x=0, font=dict(size=14, color=NAVY)),
        height=height,
    )
    return fig


# ── 1. Bar chart ──────────────────────────────────────────────────────────────
def bar_chart(df, x, y, title, color=TEAL, orientation="v", text_auto=True):
    if orientation == "h":
        fig = px.bar(df, x=y, y=x, orientation="h",
                     color_discrete_sequence=[color], text_auto=text_auto)
    else:
        fig = px.bar(df, x=x, y=y,
                     color_discrete_sequence=[color], text_auto=text_auto)
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f4f8")
    return _apply(fig, title)


# ── 2. Line chart ─────────────────────────────────────────────────────────────
def line_chart(df, x, y, title, color=TEAL, markers=True):
    fig = px.line(df, x=x, y=y, markers=markers,
                  color_discrete_sequence=[color])
    fig.update_traces(line_width=2.5)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f4f8")
    return _apply(fig, title)


# ── 3. Multi-line chart ───────────────────────────────────────────────────────
def multi_line_chart(df, x, y_cols, title):
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], name=col,
            mode="lines+markers",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5)
        ))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f4f8")
    return _apply(fig, title)


# ── 4. Pie / Donut chart ──────────────────────────────────────────────────────
def donut_chart(df, names, values, title):
    fig = px.pie(df, names=names, values=values, hole=0.45,
                 color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(
        **LAYOUT,
        title=dict(text=title, x=0, font=dict(size=14, color=NAVY)),
        height=380,
        showlegend=False,
    )
    return fig


# ── 5. Scatter chart ──────────────────────────────────────────────────────────
def scatter_chart(df, x, y, title, color_col=None, size_col=None):
    fig = px.scatter(
        df, x=x, y=y,
        color=color_col,
        size=size_col,
        color_discrete_sequence=PALETTE,
        opacity=0.7
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f4f8")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f4f8")
    return _apply(fig, title)


# ── 6. Grouped bar chart ──────────────────────────────────────────────────────
def grouped_bar_chart(df, x, y, group, title):
    fig = px.bar(df, x=x, y=y, color=group, barmode="group",
                 color_discrete_sequence=PALETTE)
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f4f8")
    return _apply(fig, title)


# ── 7. Treemap ────────────────────────────────────────────────────────────────
def treemap_chart(df, path, values, title):
    fig = px.treemap(df, path=path, values=values,
                     color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="label+value+percent root")
    fig.update_layout(
        **LAYOUT,
        title=dict(text=title, x=0, font=dict(size=14, color=NAVY)),
        height=400,
    )
    return fig


# ── 8. Heatmap ────────────────────────────────────────────────────────────────
def heatmap_chart(pivot_df, title):
    fig = px.imshow(
        pivot_df,
        color_continuous_scale=["#ffffff", LIGHT_TEAL, TEAL, NAVY],
        aspect="auto",
        text_auto=True
    )
    fig.update_layout(
        **LAYOUT,
        title=dict(text=title, x=0, font=dict(size=14, color=NAVY)),
        height=400,
        coloraxis_showscale=False,
    )
    return fig


# ── 9. Funnel chart ───────────────────────────────────────────────────────────
def funnel_chart(df, x, y, title):
    fig = px.funnel(df, x=x, y=y, color_discrete_sequence=PALETTE)
    fig.update_layout(
        **LAYOUT,
        title=dict(text=title, x=0, font=dict(size=14, color=NAVY)),
        height=380,
    )
    return fig


# ── 10. Box plot ──────────────────────────────────────────────────────────────
def box_plot(df, x, y, title):
    fig = px.box(df, x=x, y=y, color=x,
                 color_discrete_sequence=PALETTE)
    fig.update_layout(
        **LAYOUT,
        title=dict(text=title, x=0, font=dict(size=14, color=NAVY)),
        height=400,
        showlegend=False,
    )
    return fig