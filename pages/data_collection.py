import streamlit as st
import pandas as pd
from src.style import apply_style, page_header, section_header, kpi_card
from src.validation import validate_superstore_schema, get_schema_summary
from src.google_sheets import load_google_sheet


def _show_validation_error(result):
    if result["message"] == "missing_columns":
        st.error("❌ This file is missing required columns and cannot be processed.")
        st.markdown(
            "<div style='background:#fff5f5; border:1.5px solid #e74c3c; border-radius:8px; "
            "padding:14px 18px; margin-top:8px;'>"
            "<div style='font-size:13px; font-weight:700; color:#c0392b; margin-bottom:8px;'>"
            "Missing Columns:</div>"
            "<div style='font-size:13px; color:#555;'>" +
            "".join([f"<span style='background:#fde8e8; border-radius:4px; padding:2px 8px; "
                     f"margin:2px; display:inline-block;'>✗ {c}</span>"
                     for c in result["missing_columns"]]) +
            "</div></div>",
            unsafe_allow_html=True
        )
    elif result["message"] == "type_errors":
        st.error("❌ This file has incompatible data types in required columns.")
        st.markdown(
            "<div style='background:#fff5f5; border:1.5px solid #e74c3c; border-radius:8px; "
            "padding:14px 18px; margin-top:8px;'>"
            "<div style='font-size:13px; font-weight:700; color:#c0392b; margin-bottom:8px;'>"
            "Data Type Issues:</div>"
            "<div style='font-size:13px; color:#555;'>" +
            "".join([f"<div style='margin-bottom:4px;'>⚠️ {e}</div>"
                     for e in result["type_errors"]]) +
            "</div></div>",
            unsafe_allow_html=True
        )


def _row_count_control(key, total_rows):
    """Renders a sleek row count selector. Returns selected number of rows."""
    st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:space-between;
                background:#f8fffe; border:1.5px solid #17a589; border-radius:10px;
                padding:10px 18px; margin-bottom:12px;'>
        <div style='display:flex; align-items:center; gap:8px;'>
            <i class='bi bi-table' style='color:#17a589; font-size:16px;'></i>
            <span style='font-weight:600; color:#1a2940; font-size:14px;'>Data Preview</span>
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


def _show_data_summary(df, key="dc"):
    nulls = int(df.isnull().sum().sum())
    dups  = int(df.duplicated().sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("bi-table",                "Total Rows",     f"{len(df):,}")
    with c2: kpi_card("bi-layout-three-columns", "Columns",        f"{len(df.columns)}")
    with c3: kpi_card("bi-exclamation-triangle", "Missing Values", f"{nulls:,}")
    with c4: kpi_card("bi-copy",                 "Duplicate Rows", f"{dups:,}")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    n_rows = _row_count_control(key, len(df))
    st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True)

    with st.expander("Full Schema Summary"):
        st.dataframe(get_schema_summary(df), use_container_width=True, hide_index=True)


def show():
    apply_style()

    page_header(
        "Data Collection",
        "Upload your Superstore dataset to begin. The application will validate and preview your data before you proceed."
    )

    if "raw_df" in st.session_state:
        st.success(
            f"✅ Dataset already loaded — "
            f"{len(st.session_state['raw_df']):,} rows × "
            f"{len(st.session_state['raw_df'].columns)} columns. "
            f"You can re-upload below to replace it."
        )
        _show_data_summary(st.session_state["raw_df"], key="dc_existing")
        st.markdown("---")

    section_header("Select Data Source")

    mode = st.radio(
        "",
        ["📁  Upload a File", "🔗  Google Sheet URL"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── MODE A ────────────────────────────────────────────────────────────────
    if mode == "📁  Upload a File":
        section_header("Upload File")

        st.markdown("""
        <div style="background:#f0faf8; border:1.5px solid #17a589; border-radius:10px;
                    padding:12px 16px; display:flex; align-items:center; gap:12px; margin-bottom:16px;">
            <i class="bi bi-cloud-arrow-up-fill" style="font-size:22px; color:#17a589; flex-shrink:0;"></i>
            <span style="font-size:13px;">
                Accepted formats: <b>.csv</b>, <b>.xls</b>, <b>.xlsx</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your file here or click to browse",
            type=["csv", "xls", "xlsx"],
            label_visibility="visible"
        )

        if uploaded:
            try:
                fname = uploaded.name.lower()
                if fname.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    xl    = pd.ExcelFile(uploaded)
                    sheet = "Orders" if "Orders" in xl.sheet_names else xl.sheet_names[0]
                    df    = xl.parse(sheet)

                df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
                result = validate_superstore_schema(df)

                if not result["valid"]:
                    _show_validation_error(result)
                    return

                st.session_state["raw_df"] = df
                st.success(f"✅ Dataset loaded successfully — {len(df):,} rows × {len(df.columns)} columns")

                if result["warnings"]:
                    st.info("ℹ️ Some optional columns are missing but the app will work normally: " +
                            ", ".join(result["warnings"]))

                _show_data_summary(df, key="dc_upload")

            except Exception as e:
                st.error(f"❌ Could not read file: {e}")

    # ── MODE B ────────────────────────────────────────────────────────────────
    else:
        section_header("Google Sheet URL")

        st.markdown("""
        <div style="background:#f0faf8; border:1.5px solid #17a589; border-radius:10px;
                    padding:12px 16px; display:flex; align-items:center; gap:12px; margin-bottom:16px;">
            <i class="bi bi-link-45deg" style="font-size:22px; color:#17a589; flex-shrink:0;"></i>
            <span style="font-size:13px;">
                Paste the shareable link of your public Google Sheet below.
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        div[data-testid="InputInstructions"] { display: none !important; }
        </style>
        """, unsafe_allow_html=True)

        url = st.text_input(
            "Google Sheet URL",
            placeholder="https://docs.google.com/spreadsheets/d/...",
        )
        st.markdown(
            "<p style='font-size:11px; color:#888; margin-top:-12px;'>"
            "Press Enter to load the sheet.</p>",
            unsafe_allow_html=True
        )

        if url:
            with st.spinner("Loading sheet..."):
                result = load_google_sheet(url)

            if not result["success"]:
                st.error(f"❌ {result['message']}")
                return

            df  = result["data"]
            df  = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
            val = validate_superstore_schema(df)

            if not val["valid"]:
                _show_validation_error(val)
                return

            st.session_state["raw_df"] = df
            st.success(f"✅ Sheet loaded — {len(df):,} rows × {len(df.columns)} columns")

            with st.sidebar:
                st.markdown(f"""
                <div style='padding:14px;'>
                    <div style='font-size:15px; font-weight:700; color:#ffffff; margin-bottom:14px;'>
                        <i class='bi bi-file-earmark-spreadsheet-fill' style='color:#17a589; margin-right:6px;'></i>
                        Sheet Info
                    </div>
                    <div style='font-size:13px; color:#ffffff; margin-bottom:8px;'>
                        <i class='bi bi-file-text' style='color:#17a589; margin-right:6px;'></i>
                        <b>Name:</b> {result.get('sheet_name', 'Unknown')}
                    </div>
                    <div style='font-size:13px; color:#ffffff; margin-bottom:8px;'>
                        <i class='bi bi-clock' style='color:#17a589; margin-right:6px;'></i>
                        <b>Loaded at:</b> {result.get('loaded_at', 'N/A')}
                    </div>
                    <div style='font-size:13px; color:#ffffff; margin-bottom:8px;'>
                        <i class='bi bi-table' style='color:#17a589; margin-right:6px;'></i>
                        <b>Rows:</b> {len(df):,}
                    </div>
                    <div style='font-size:13px; color:#ffffff; margin-bottom:8px;'>
                        <i class='bi bi-layout-three-columns' style='color:#17a589; margin-right:6px;'></i>
                        <b>Columns:</b> {len(df.columns)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if val["warnings"]:
                st.info("ℹ️ Some optional columns are missing but the app will work normally: " +
                        ", ".join(val["warnings"]))

            _show_data_summary(df, key="dc_sheet")