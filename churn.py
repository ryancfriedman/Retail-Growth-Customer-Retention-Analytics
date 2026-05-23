import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split


def build_churn_dataset(df):
    """
    Builds a churn modeling dataset.

    The final 3 months are used as the future period.
    A customer is labeled as churned if they purchased in the historical period
    but did not purchase again in the future period.
    """

    data = df.copy()

    max_date = data["invoice_date"].max()
    cutoff_date = max_date - pd.DateOffset(months=3)

    historical = data[data["invoice_date"] <= cutoff_date].copy()
    future = data[data["invoice_date"] > cutoff_date].copy()

    snapshot_date = historical["invoice_date"].max() + pd.Timedelta(days=1)

    features = historical.groupby("customer_id").agg(
        recency_days=("invoice_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
        total_items=("quantity", "sum"),
        unique_products=("stock_code", "nunique"),
        avg_line_revenue=("revenue", "mean"),
        active_months=("invoice_month", "nunique"),
    ).reset_index()

    features["revenue_per_order"] = features["monetary"] / features["frequency"]
    features["items_per_order"] = features["total_items"] / features["frequency"]
    features["orders_per_active_month"] = (
        features["frequency"] / features["active_months"]
    )

    future_buyers = set(future["customer_id"].unique())

    features["purchased_future"] = features["customer_id"].apply(
        lambda customer_id: 1 if customer_id in future_buyers else 0
    )

    features["churned"] = 1 - features["purchased_future"]

    return features


def train_churn_model(churn_df):
    """
    Trains a random forest churn model and generates customer churn-risk scores.
    """

    data = churn_df.copy()

    feature_cols = [
        "recency_days",
        "frequency",
        "monetary",
        "total_items",
        "unique_products",
        "avg_line_revenue",
        "active_months",
        "revenue_per_order",
        "items_per_order",
        "orders_per_active_month",
    ]

    X = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = data["churned"]

    if y.nunique() < 2:
        raise ValueError(
            "Churn target has only one class. Adjust the churn labeling window."
        )

    stratify_arg = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_arg,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "auc": round(roc_auc_score(y_test, y_prob), 3),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
    }

    data["churn_risk_score"] = model.predict_proba(X)[:, 1]

    data["risk_tier"] = pd.cut(
        data["churn_risk_score"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True,
    )

    data["recommended_action"] = data.apply(assign_recommended_action, axis=1)

    return data, metrics


def assign_recommended_action(row):
    """
    Converts churn score and customer value into a business action.
    """

    if row["risk_tier"] == "High Risk" and row["monetary"] >= 500:
        return "Priority reactivation campaign"

    if row["risk_tier"] == "High Risk":
        return "Low-cost win-back email"

    if row["risk_tier"] == "Medium Risk" and row["frequency"] >= 3:
        return "Loyalty incentive"

    if row["risk_tier"] == "Medium Risk":
        return "Standard retention email"

    return "Maintain standard engagement"