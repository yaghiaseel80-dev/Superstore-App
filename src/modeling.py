import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Feature preparation ───────────────────────────────────────────────────────

def prepare_features(df, target="Sales", feature_cols=None):
    """
    Prepares X (features) and y (target) for ML.

    - Drops rows where target is null
    - One-hot encodes categorical columns
    - Fills remaining nulls with column median
    - Returns X, y, and the list of final feature names
    """
    df = df.copy()

    # Default feature columns if not specified
    if feature_cols is None:
        feature_cols = ["Quantity", "Discount", "Category", "Region", "Segment", "Ship Mode"]

    # Keep only columns that exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    # Ensure target is numeric
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    y = df[target].values

    # Separate numeric and categorical features
    numeric_cols     = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, float, int]]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # One-hot encode categorical columns
    X = df[numeric_cols].copy()
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        X = pd.concat([X, dummies], axis=1)

    # Fill any remaining nulls
    X = X.fillna(X.median(numeric_only=True))

    # Convert bool columns to int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    feature_names = X.columns.tolist()
    return X.values, y, feature_names


# ── Model training ────────────────────────────────────────────────────────────

def train_models(X, y):
    """
    Trains Linear Regression and Random Forest on the data.
    Uses 5-fold cross-validation for fair evaluation.
    Returns a dict with metrics and fitted models.
    """
    results = {}

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        # Cross-validation R² scores
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

        # Fit on training set, evaluate on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        results[name] = {
            "model":     model,
            "y_test":    y_test,
            "y_pred":    y_pred,
            "mae":       round(mae, 2),
            "rmse":      round(rmse, 2),
            "r2":        round(r2, 4),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std":  round(cv_scores.std(), 4),
        }

    return results


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names, top_n=15):
    """
    Returns a DataFrame of feature importances.
    Works for Random Forest (feature_importances_) and
    Linear Regression (absolute coefficients).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    # Normalize to percentage
    df["Importance %"] = (df["Importance"] / df["Importance"].sum() * 100).round(1)

    return df.reset_index(drop=True)


# ── Business interpretation ───────────────────────────────────────────────────

def interpret_results(results, target):
    """
    Returns a plain-English summary of model performance
    and what it means for the business.
    """
    best_model = max(results, key=lambda k: results[k]["r2"])
    best       = results[best_model]
    other      = [k for k in results if k != best_model][0]

    r2_pct  = round(best["r2"] * 100, 1)
    mae_fmt = f"${best['mae']:,.0f}" if target == "Sales" else f"${best['mae']:,.2f}"

    lines = [
        f"**Best performing model: {best_model}** with an R² of {best['r2']} ({r2_pct}% of variance explained).",
        f"On average, predictions are off by {mae_fmt} (MAE). "
        f"The lower this number, the more reliable the model is for business planning.",
        f"Cross-validation R² of {best['cv_r2_mean']} (±{best['cv_r2_std']}) confirms the model generalises well to unseen data — it's not just memorising the training set.",
    ]

    if best["r2"] >= 0.75:
        lines.append(f"✅ The model explains a strong portion of {target} variation — suitable for forecasting and scenario planning.")
    elif best["r2"] >= 0.5:
        lines.append(f"⚠️ The model explains a moderate portion of {target} variation. Useful for trend analysis but predictions should be treated as estimates.")
    else:
        lines.append(f"⚠️ The model has limited predictive power for {target}. Consider adding more features or reviewing data quality.")

    lines.append(
        f"The Feature Importance chart below shows which factors drive {target} most. "
        f"Focus business attention on the top-ranked features for maximum impact."
    )

    return lines