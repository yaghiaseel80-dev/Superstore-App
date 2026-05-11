import pandas as pd
import numpy as np

# ── Critical columns — app cannot function without these ──────────────────────
CRITICAL_COLUMNS = [
    "Order ID", "Order Date", "Ship Date", "Customer ID", "Customer Name",
    "Segment", "City", "State", "Region", "Category", "Sub-Category",
    "Product Name", "Sales", "Quantity", "Discount", "Profit"
]

# ── Expected numeric columns ──────────────────────────────────────────────────
NUMERIC_COLUMNS = ["Sales", "Quantity", "Discount", "Profit"]

# ── Expected date columns ─────────────────────────────────────────────────────
DATE_COLUMNS = ["Order Date", "Ship Date"]


def validate_superstore_schema(df):
    """
    Two-level validation:
    - Missing critical columns → block upload, list exactly which ones
    - Type incompatibility → block upload, list exactly which columns have issues
    - Returns valid, missing_columns, type_errors, warnings, message
    """
    existing = set(df.columns.tolist())

    # Check missing critical columns
    missing_critical = [c for c in CRITICAL_COLUMNS if c not in existing]

    if missing_critical:
        return {
            "valid":          False,
            "missing_columns": missing_critical,
            "type_errors":    [],
            "warnings":       [],
            "message":        "missing_columns"
        }

    # Check numeric type compatibility
    type_errors = []
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            pct_bad   = converted.isnull().sum() / len(df)
            if pct_bad > 0.10:
                type_errors.append(
                    f"'{col}' appears non-numeric ({round(pct_bad*100,1)}% of values could not be converted)"
                )

    if type_errors:
        return {
            "valid":          False,
            "missing_columns": [],
            "type_errors":    type_errors,
            "warnings":       [],
            "message":        "type_errors"
        }

    # Soft warnings for optional columns
    optional = ["Ship Mode", "Postal Code", "Product ID", "Row ID",
                "Country", "Country/Region"]
    missing_optional = [c for c in optional if c not in existing]

    return {
        "valid":          True,
        "missing_columns": [],
        "type_errors":    [],
        "warnings":       missing_optional,
        "message":        "valid"
    }


def get_schema_summary(df):
    """
    Returns a clean table: column name, data type, non-null count, null count.
    """
    summary = pd.DataFrame({
        "Column":     df.columns,
        "Type":       [str(df[c].dtype) for c in df.columns],
        "Non-Null":   [int(df[c].notnull().sum()) for c in df.columns],
        "Null Count": [int(df[c].isnull().sum())  for c in df.columns],
    })
    return summary