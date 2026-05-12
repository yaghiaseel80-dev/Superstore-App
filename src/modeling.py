import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
from sklearn.model_selection import cross_val_score, train_test_split


# ── Target definitions ────────────────────────────────────────────────────────

TARGET_CONFIG = {
    "Shipping Duration": {
        "col":         "Shipping Duration",
        "features":    ["Ship Mode", "Region", "Category", "Segment", "Sub-Category"],
        "unit":        "days",
        "description": "Predict how many days shipping will take based on logistics and order details.",
    },
    "Discount": {
        "col":         "Discount",
        "features":    ["Category", "Sub-Category", "Region", "Segment", "Quantity"],
        "unit":        "%",
        "description": "Predict the discount rate likely applied to an order.",
    },
}


# ── Feature preparation (supervised) ─────────────────────────────────────────

def prepare_features(df, target="Shipping Duration", feature_cols=None):
    df = df.copy()

    if target == "Shipping Duration":
        if "Shipping Duration" not in df.columns:
            if "Order Date" in df.columns and "Ship Date" in df.columns:
                df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
                df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  errors="coerce")
                df["Shipping Duration"] = (df["Ship Date"] - df["Order Date"]).dt.days
            else:
                raise ValueError("Dataset must contain 'Order Date' and 'Ship Date'.")

    if feature_cols is None:
        feature_cols = TARGET_CONFIG.get(target, {}).get("features", [])

    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=[target])
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    if target == "Shipping Duration":
        df = df[df[target] >= 0]

    y = df[target].values

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

    return X.values, y, X.columns.tolist()


# ── Model training (supervised) ───────────────────────────────────────────────

def train_models(X, y):
    results = {}
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model":      model,
            "y_test":     y_test,
            "y_pred":     y_pred,
            "mae":        round(mean_absolute_error(y_test, y_pred), 2),
            "rmse":       round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "r2":         round(r2_score(y_test, y_pred), 4),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std":  round(cv_scores.std(), 4),
        }

    return results


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names, top_n=15):
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

    df["Importance %"] = (df["Importance"] / df["Importance"].sum() * 100).round(1)
    return df.reset_index(drop=True)


# ── Single prediction (supervised) ───────────────────────────────────────────

def predict_single(model, feature_names, input_dict):
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

    return model.predict(X.values)[0]


# ── Business interpretation (supervised) ─────────────────────────────────────

def interpret_results(results, target):
    best_model = max(results, key=lambda k: results[k]["r2"])
    best       = results[best_model]
    unit_map   = {"Shipping Duration": "days", "Discount": "%"}
    unit       = unit_map.get(target, "")

    lines = [
        f"**Best model: {best_model}** — R² of {best['r2']} ({round(best['r2']*100,1)}% variance explained).",
        f"Average prediction error: {best['mae']:.2f} {unit} (MAE).",
        f"Cross-validation R²: {best['cv_r2_mean']} (±{best['cv_r2_std']}).",
    ]

    if best["r2"] >= 0.75:
        lines.append("Strong predictive power — suitable for operational planning.")
    elif best["r2"] >= 0.5:
        lines.append("Moderate predictive power — useful for trend analysis.")
    else:
        lines.append("Limited predictive power — consider adding more features.")

    return lines


# ── Clustering helpers (unsupervised) ─────────────────────────────────────────

CLUSTER_FEATURES = ["Sales", "Profit", "Discount", "Quantity"]


def prepare_cluster_data(df):
    """Aggregate to customer level and scale for KMeans."""
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
    """Elbow + Silhouette → optimal K."""
    inertias   = []
    sil_scores = []
    k_range    = list(range(k_min, k_max + 1))

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # Elbow: largest inertia drop
    drops   = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    elbow_k = k_range[drops.index(max(drops)) + 1]

    # Silhouette: highest score
    sil_k   = k_range[sil_scores.index(max(sil_scores))]

    if elbow_k == sil_k:
        optimal_k = elbow_k
        method    = "both the Elbow and Silhouette methods agree"
    else:
        optimal_k = sil_k
        method    = "the Silhouette method (highest score)"

    return optimal_k, {
        "k_range":   k_range,
        "inertias":  inertias,
        "sil_scores": sil_scores,
        "elbow_k":   elbow_k,
        "sil_k":     sil_k,
        "method":    method,
        "optimal_k": optimal_k,
    }


def run_clustering(X_scaled, customer_df, feature_cols, optimal_k):
    """Fit KMeans, label clusters, return enriched customer_df and profile."""
    km           = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    customer_df  = customer_df.copy()
    customer_df["Cluster"] = km.fit_predict(X_scaled)

    profile = customer_df.groupby("Cluster")[feature_cols].mean().round(2)
    profile["Customer Count"] = customer_df.groupby("Cluster").size().values

    # Label by Sales rank
    if "Sales" in profile.columns:
        ranks = profile["Sales"].rank(ascending=False).astype(int)
        n     = optimal_k
        labels_map = {}
        for idx, rank in ranks.items():
            if rank == 1:       labels_map[idx] = "High-Value Customers"
            elif rank == n:     labels_map[idx] = "Low-Value Customers"
            elif rank == 2:     labels_map[idx] = "Growth Potential Customers"
            else:               labels_map[idx] = "Mid-Tier Customers"
        profile["Label"] = profile.index.map(labels_map)
    else:
        profile["Label"] = [f"Segment {i+1}" for i in profile.index]

    return customer_df, profile, km


def predict_cluster(km, scaler, feature_cols, input_dict):
    """Predict cluster for a single new customer."""
    row = pd.DataFrame([{col: input_dict.get(col, 0) for col in feature_cols}])
    X   = scaler.transform(row.values)
    return int(km.predict(X)[0])