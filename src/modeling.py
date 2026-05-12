import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)
from sklearn.model_selection import cross_val_score, train_test_split


# ── Target definitions ────────────────────────────────────────────────────────

REGRESSION_CONFIG = {
    "Shipping Duration": {
        "col":         "Shipping Duration",
        "features":    ["Ship Mode", "Region", "Category", "Segment", "Sub-Category"],
        "unit":        "days",
        "description": "Predict the exact number of days an order will take to ship.",
    },
}

CLASSIFICATION_CONFIG = {
    "Shipping Speed": {
        "col":         "Shipping Speed",
        "features":    ["Ship Mode", "Region", "Category", "Segment", "Sub-Category"],
        "description": "Classify orders as Fast, Normal, or Slow based on shipping details.",
        "classes":     ["Fast", "Normal", "Slow"],
        "thresholds":  (2, 4),   # ≤2 = Fast, 3-4 = Normal, ≥5 = Slow
    },
    "Discount Level": {
        "col":         "Discount Level",
        "features":    ["Category", "Sub-Category", "Region", "Segment", "Quantity"],
        "description": "Classify orders as Low, Medium, or High discount.",
        "classes":     ["Low", "Medium", "High"],
        "thresholds":  (0.15, 0.35),  # <15% = Low, 15-35% = Medium, >35% = High
    },
}

# Combined for easy lookup
TARGET_CONFIG = {**REGRESSION_CONFIG, **CLASSIFICATION_CONFIG}


# ── Feature preparation ───────────────────────────────────────────────────────

def _encode_features(df, feature_cols):
    """One-hot encode categorical features, return X array and feature names."""
    numeric_cols     = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, float, int]]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    X = df[numeric_cols].copy()
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        X = pd.concat([X, dummies], axis=1)

    X = X.fillna(X.median(numeric_only=True))
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    return X.values, X.columns.tolist()


def prepare_features(df, target, feature_cols=None):
    """Prepare X and y for regression or classification."""
    df = df.copy()

    # ── Regression: Shipping Duration ────────────────────────────────────────
    if target == "Shipping Duration":
        if "Shipping Duration" not in df.columns:
            if "Order Date" in df.columns and "Ship Date" in df.columns:
                df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
                df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  errors="coerce")
                df["Shipping Duration"] = (df["Ship Date"] - df["Order Date"]).dt.days
            else:
                raise ValueError("Dataset must contain 'Order Date' and 'Ship Date'.")
        df = df[df["Shipping Duration"] >= 0].dropna(subset=["Shipping Duration"])
        df["Shipping Duration"] = pd.to_numeric(df["Shipping Duration"], errors="coerce")
        df = df.dropna(subset=["Shipping Duration"])
        y = df["Shipping Duration"].values

    # ── Classification: Shipping Speed ───────────────────────────────────────
    elif target == "Shipping Speed":
        if "Shipping Duration" not in df.columns:
            if "Order Date" in df.columns and "Ship Date" in df.columns:
                df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
                df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  errors="coerce")
                df["Shipping Duration"] = (df["Ship Date"] - df["Order Date"]).dt.days
            else:
                raise ValueError("Dataset must contain 'Order Date' and 'Ship Date'.")
        df = df[df["Shipping Duration"] >= 0].dropna(subset=["Shipping Duration"])
        t1, t2 = CLASSIFICATION_CONFIG["Shipping Speed"]["thresholds"]
        df["Shipping Speed"] = df["Shipping Duration"].apply(
            lambda x: "Fast" if x <= t1 else ("Normal" if x <= t2 else "Slow"))
        y = df["Shipping Speed"].values

    # ── Classification: Discount Level ───────────────────────────────────────
    elif target == "Discount Level":
        if "Discount" not in df.columns:
            raise ValueError("Dataset must contain 'Discount' column.")
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
        df = df.dropna(subset=["Discount"])
        t1, t2 = CLASSIFICATION_CONFIG["Discount Level"]["thresholds"]
        df["Discount Level"] = df["Discount"].apply(
            lambda x: "Low" if x < t1 else ("Medium" if x < t2 else "High"))
        y = df["Discount Level"].values

    else:
        raise ValueError(f"Unknown target: {target}")

    cfg          = TARGET_CONFIG.get(target, {})
    feature_cols = feature_cols or cfg.get("features", [])
    feature_cols = [c for c in feature_cols if c in df.columns]

    X, feature_names = _encode_features(df, feature_cols)
    return X, y, feature_names


# ── Regression training ───────────────────────────────────────────────────────

def train_regression_models(X, y):
    results = {}
    models  = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        cv    = cross_val_score(model, X, y, cv=5, scoring="r2")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model":      model,
            "y_test":     y_test,
            "y_pred":     y_pred,
            "mae":        round(mean_absolute_error(y_test, y_pred), 2),
            "rmse":       round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "r2":         round(r2_score(y_test, y_pred), 4),
            "cv_r2_mean": round(cv.mean(), 4),
            "cv_r2_std":  round(cv.std(), 4),
            "type":       "regression",
        }
    return results


# ── Classification training ───────────────────────────────────────────────────

def train_classification_models(X, y):
    results = {}
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    classes = le.classes_

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    for name, model in models.items():
        cv    = cross_val_score(model, X, y_enc, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm        = confusion_matrix(y_test, y_pred)
        acc       = round(accuracy_score(y_test, y_pred), 4)
        precision = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        recall    = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        f1        = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)

        # Specificity: mean of per-class TN/(TN+FP)
        specificity_list = []
        for i in range(len(classes)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        specificity = round(np.mean(specificity_list), 4)

        results[name] = {
            "model":        model,
            "y_test":       y_test,
            "y_pred":       y_pred,
            "classes":      classes,
            "le":           le,
            "acc":          acc,
            "precision":    precision,
            "recall":       recall,
            "specificity":  specificity,
            "f1":           f1,
            "cv_acc_mean":  round(cv.mean(), 4),
            "cv_acc_std":   round(cv.std(), 4),
            "cm":           cm,
            "type":         "classification",
        }
    return results


# ── Unified train entry point ─────────────────────────────────────────────────

def train_models(X, y, target):
    if target in REGRESSION_CONFIG:
        return train_regression_models(X, y)
    else:
        return train_classification_models(X, y)


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)
    df["Importance %"] = (df["Importance"] / df["Importance"].sum() * 100).round(1)
    return df.reset_index(drop=True)


# ── Interpretation ────────────────────────────────────────────────────────────

def interpret_results(results, target):
    if target in REGRESSION_CONFIG:
        best = max(results, key=lambda k: results[k]["r2"])
        b    = results[best]
        unit = REGRESSION_CONFIG[target].get("unit", "")
        lines = [
            f"**Best model: {best}** — R² of {b['r2']} ({round(b['r2']*100,1)}% variance explained).",
            f"Average prediction error: {b['mae']:.2f} {unit} (MAE).",
            f"Cross-validation R²: {b['cv_r2_mean']} (±{b['cv_r2_std']}).",
        ]
        if b["r2"] >= 0.75:   lines.append("Strong predictive power — suitable for operational planning.")
        elif b["r2"] >= 0.5:  lines.append("Moderate predictive power — useful for trend analysis.")
        else:                  lines.append("Limited predictive power — consider adding more features.")
    else:
        best = max(results, key=lambda k: results[k]["acc"])
        b    = results[best]
        lines = [
            f"**Best model: {best}** — Accuracy of {round(b['acc']*100,1)}%.",
            f"Precision: {round(b['precision']*100,1)}% | Recall: {round(b['recall']*100,1)}% | F1: {round(b['f1']*100,1)}%.",
            f"Specificity: {round(b['specificity']*100,1)}% | CV Accuracy: {round(b['cv_acc_mean']*100,1)}% (±{round(b['cv_acc_std']*100,1)}%).",
        ]
        if b["acc"] >= 0.80:   lines.append("Strong classification performance — reliable for business decisions.")
        elif b["acc"] >= 0.65: lines.append("Moderate classification performance — use with caution.")
        else:                   lines.append("Limited classification performance — consider feature engineering.")
    return lines


# ── Single prediction ─────────────────────────────────────────────────────────

def predict_single(model, feature_names, input_dict, is_classification=False, le=None):
    row      = pd.DataFrame([input_dict])
    cat_cols = [c for c in input_dict if isinstance(input_dict[c], str)]
    num_cols = [c for c in input_dict if not isinstance(input_dict[c], str)]

    X = row[num_cols].copy() if num_cols else pd.DataFrame(index=[0])
    for col in cat_cols:
        dummies = pd.get_dummies(row[col], prefix=col)
        X = pd.concat([X, dummies], axis=1)

    X = X.reindex(columns=feature_names, fill_value=0)
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    pred = model.predict(X.values)[0]
    if is_classification and le is not None:
        return le.inverse_transform([pred])[0]
    return pred


# ── Clustering ────────────────────────────────────────────────────────────────

CLUSTER_FEATURES = ["Sales", "Profit", "Discount", "Quantity"]


def prepare_cluster_data(df):
    df = df.copy()
    for col in CLUSTER_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg_dict = {}
    if "Sales"    in df.columns: agg_dict["Sales"]    = ("Sales",    "sum")
    if "Profit"   in df.columns: agg_dict["Profit"]   = ("Profit",   "sum")
    if "Discount" in df.columns: agg_dict["Discount"] = ("Discount", "mean")
    if "Quantity" in df.columns: agg_dict["Quantity"] = ("Quantity", "sum")
    if "Order ID" in df.columns: agg_dict["Orders"]   = ("Order ID", "nunique")

    group_col = (
        "Customer ID"   if "Customer ID"   in df.columns else
        "Customer Name" if "Customer Name" in df.columns else None
    )
    if group_col is None:
        raise ValueError("Dataset must contain 'Customer ID' or 'Customer Name'.")

    customer_df  = df.groupby(group_col).agg(**agg_dict).reset_index()
    customer_df  = customer_df.dropna()
    feature_cols = [c for c in ["Sales", "Profit", "Discount", "Quantity", "Orders"]
                    if c in customer_df.columns]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(customer_df[feature_cols].values)
    return X_scaled, scaler, customer_df, feature_cols


def find_optimal_clusters(X_scaled, k_min=2, k_max=8):
    inertias   = []
    sil_scores = []
    k_range    = list(range(k_min, k_max + 1))

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    drops   = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    elbow_k = k_range[drops.index(max(drops)) + 1]
    sil_k   = k_range[sil_scores.index(max(sil_scores))]

    if elbow_k == sil_k:
        optimal_k = elbow_k
        method    = "both the Elbow and Silhouette methods agree"
    else:
        optimal_k = sil_k
        method    = "the Silhouette method (highest score)"

    return optimal_k, {
        "k_range": k_range, "inertias": inertias, "sil_scores": sil_scores,
        "elbow_k": elbow_k, "sil_k": sil_k, "method": method, "optimal_k": optimal_k,
    }


def run_clustering(X_scaled, customer_df, feature_cols, optimal_k):
    km          = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    customer_df = customer_df.copy()
    customer_df["Cluster"] = km.fit_predict(X_scaled)

    profile = customer_df.groupby("Cluster")[feature_cols].mean().round(2)
    profile["Customer Count"] = customer_df.groupby("Cluster").size().values

    if "Sales" in profile.columns:
        ranks = profile["Sales"].rank(ascending=False).astype(int)
        n     = optimal_k
        labels_map = {}
        for idx, rank in ranks.items():
            if rank == 1:   labels_map[idx] = "High-Value Customers"
            elif rank == n: labels_map[idx] = "Low-Value Customers"
            elif rank == 2: labels_map[idx] = "Growth Potential Customers"
            else:           labels_map[idx] = "Mid-Tier Customers"
        profile["Label"] = profile.index.map(labels_map)
    else:
        profile["Label"] = [f"Segment {i+1}" for i in profile.index]

    return customer_df, profile, km


def predict_cluster(km, scaler, feature_cols, input_dict):
    row = pd.DataFrame([{col: input_dict.get(col, 0) for col in feature_cols}])
    X   = scaler.transform(row.values)
    return int(km.predict(X)[0])