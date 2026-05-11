import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED CLEANING MEASURE 1 — Missing Data
# ─────────────────────────────────────────────────────────────────────────────

def get_missing_summary(df):
    """
    Returns a DataFrame showing columns that have missing values,
    how many, and what percentage of the total rows they represent.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    summary = pd.DataFrame({
        "Column":         missing.index,
        "Missing Count":  missing.values,
        "Missing %":      (missing.values / len(df) * 100).round(2)
    }).reset_index(drop=True)
    return summary


def impute_missing(df, strategy="drop"):
    """
    Handle missing values across the DataFrame.

    Strategies:
      - "drop"      : Remove rows with any missing value.
      - "mean"      : Fill numeric columns with the column mean.
      - "median"    : Fill numeric columns with the column median.
      - "mode"      : Fill all columns with the most frequent value.
      - "customer"  : Fill numeric values using that Customer ID's historical
                      average (e.g. average Sales for that customer). Falls back
                      to the global column mean if no history exists.
      - "ffill"     : Forward-fill — each missing value takes the value of the
                      row above it. Useful for time-ordered data.
      - "constant"  : Fill all remaining nulls with 0 for numbers and
                      "Unknown" for text columns.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols    = df.select_dtypes(include=["object"]).columns.tolist()

    if strategy == "drop":
        df = df.dropna()

    elif strategy == "mean":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())

    elif strategy == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

    elif strategy == "mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col])

    elif strategy == "customer":
        # Fill from customer history: use the average value for that Customer ID
        if "Customer ID" in df.columns:
            for col in numeric_cols:
                customer_means = df.groupby("Customer ID")[col].transform("mean")
                global_mean    = df[col].mean()
                df[col] = df[col].fillna(customer_means).fillna(global_mean)
        else:
            # Fallback to mean if no Customer ID column
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())

    elif strategy == "ffill":
        df = df.ffill()

    elif strategy == "constant":
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        for col in text_cols:
            df[col] = df[col].fillna("Unknown")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED CLEANING MEASURE 2 — Outlier Handling
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers(df, method="iqr"):
    """
    Detect outliers in numeric columns.

    Methods:
      - "iqr"    : Interquartile Range — values below Q1-1.5*IQR or above Q3+1.5*IQR.
      - "zscore" : Z-Score — values more than 3 standard deviations from the mean.

    Returns a dict: { column_name: number_of_outliers }
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    result = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if method == "iqr":
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR    = Q3 - Q1
            mask   = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        else:  # zscore
            mean, std = series.mean(), series.std()
            mask = (df[col] - mean).abs() > 3 * std

        count = int(mask.sum())
        if count > 0:
            result[col] = count

    return result


def handle_outliers(df, method="iqr", action="cap"):
    """
    Handle outliers in numeric columns.

    Actions:
      - "flag"   : Add a boolean column '{col}_outlier_flag' marking outlier rows.
                   Data is NOT modified — safest option for investigation.
      - "cap"    : Cap values at the boundary (Winsorization).
      - "remove" : Drop rows that contain any outlier value.
      - "mean"   : Replace outlier values with the column mean.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    mask_any = pd.Series([False] * len(df), index=df.index)

    for col in numeric_cols:
        series = df[col].dropna()
        if method == "iqr":
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR    = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        else:
            mean, std    = series.mean(), series.std()
            lower, upper = mean - 3 * std, mean + 3 * std

        col_mask = (df[col] < lower) | (df[col] > upper)
        mask_any = mask_any | col_mask

        if action == "flag":
            df[f"{col}_outlier_flag"] = col_mask
        elif action == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif action == "mean":
            col_mean = df[col].mean()
            df.loc[col_mask, col] = col_mean

    if action == "remove":
        df = df[~mask_any]

    return df


def get_outlier_stats(df, method="iqr"):
    """
    Returns before/after comparison data for the outlier chart.
    Returns a list of dicts with col, lower, upper, original_count, outlier_count.
    """
    numeric_cols = ["Sales", "Profit", "Discount", "Quantity"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    stats = []

    for col in numeric_cols:
        series = df[col].dropna()
        if method == "iqr":
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR    = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        else:
            mean, std    = series.mean(), series.std()
            lower, upper = mean - 3 * std, mean + 3 * std

        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        stats.append({
            "column":        col,
            "total":         len(df),
            "outliers":      outlier_count,
            "clean":         len(df) - outlier_count,
            "lower":         round(lower, 2),
            "upper":         round(upper, 2),
        })

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED CLEANING MEASURE 3 — Format & Type Validation
# ─────────────────────────────────────────────────────────────────────────────

def format_validate(df):
    """
    Fix data types and format issues:
      - Parse Order Date and Ship Date as proper datetime objects.
      - Cast Postal Code to string and zero-pad to 5 digits (US format).
      - Cast Sales, Quantity, Discount, Profit to numeric.
      - Strip leading/trailing whitespace from all text columns.

    Returns the cleaned DataFrame and a list of fix descriptions.
    """
    df   = df.copy()
    fixes = []

    # Date columns
    for col in ["Order Date", "Ship Date"]:
        if col in df.columns:
            before_nulls = df[col].isnull().sum()
            df[col]      = pd.to_datetime(df[col], errors="coerce")
            after_nulls  = df[col].isnull().sum()
            new_nulls    = after_nulls - before_nulls
            fixes.append(
                f"'{col}' converted to datetime"
                + (f" ({new_nulls} unparseable values set to NaT)" if new_nulls > 0 else "")
            )

    # Postal Code → 5-digit string
    if "Postal Code" in df.columns:
        df["Postal Code"] = (
            df["Postal Code"]
            .astype(str)
            .str.strip()
            .str.replace(r'\.0$', '', regex=True)   # remove trailing .0 from floats
            .str.zfill(5)
        )
        fixes.append("'Postal Code' standardised to 5-digit string format")

    # Numeric columns
    for col in ["Sales", "Quantity", "Discount", "Profit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            fixes.append(f"'{col}' cast to numeric")

    # Strip whitespace from text columns
    text_cols = df.select_dtypes(include="object").columns.tolist()
    for col in text_cols:
        df[col] = df[col].str.strip()
    if text_cols:
        fixes.append(f"Whitespace stripped from {len(text_cols)} text columns")

    return df, fixes


# ─────────────────────────────────────────────────────────────────────────────
# BONUS CLEANING MEASURES
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df):
    """
    Remove fully duplicate rows. Returns cleaned df and count of removed rows.
    If the same Order ID appears twice, sales figures would be double-counted —
    removing duplicates ensures every transaction is counted exactly once.
    """
    before = len(df)
    df     = df.drop_duplicates()
    removed = before - len(df)
    return df, removed


def add_shipping_duration(df):
    """
    Add a 'Shipping Duration (Days)' column = Ship Date - Order Date.
    This is a real business KPI: late shipments correlate with lower satisfaction.
    Requires both date columns to already be datetime (run format_validate first).
    """
    df = df.copy()
    if "Order Date" in df.columns and "Ship Date" in df.columns:
        df["Shipping Duration (Days)"] = (
            pd.to_datetime(df["Ship Date"]) - pd.to_datetime(df["Order Date"])
        ).dt.days
    return df


def standardise_text(df):
    """
    Standardise capitalisation in categorical columns (Category, Sub-Category,
    Segment, Region, Ship Mode) to Title Case so grouping and filtering work
    correctly. E.g. 'furniture', 'FURNITURE', 'Furniture' all become 'Furniture'.
    """
    df   = df.copy()
    cols = ["Category", "Sub-Category", "Segment", "Region", "Ship Mode", "Country", "State", "City"]
    fixed = []
    for col in cols:
        if col in df.columns:
            df[col] = df[col].str.title()
            fixed.append(col)
    return df, fixed