import pandas as pd


def build_rfm_table(df):
    """
    Builds Recency, Frequency, Monetary customer table.
    """

    data = df.copy()
    snapshot_date = data["invoice_date"].max() + pd.Timedelta(days=1)

    rfm = data.groupby("customer_id").agg(
        recency=("invoice_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()

    rfm = rfm[rfm["monetary"] > 0].copy()

    return rfm


def assign_rfm_segments(rfm):
    """
    Assigns RFM scores and business-friendly customer segments.
    """

    data = rfm.copy()

    data["r_score"] = pd.qcut(
        data["recency"].rank(method="first"),
        q=5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop",
    ).astype(int)

    data["f_score"] = pd.qcut(
        data["frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop",
    ).astype(int)

    data["m_score"] = pd.qcut(
        data["monetary"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop",
    ).astype(int)

    data["rfm_score"] = (
        data["r_score"].astype(str)
        + data["f_score"].astype(str)
        + data["m_score"].astype(str)
    )

    def segment_customer(row):
        r = row["r_score"]
        f = row["f_score"]
        m = row["m_score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 4 and m >= 3:
            return "Loyal Customers"
        if r <= 2 and f >= 4 and m >= 4:
            return "High-Value At Risk"
        if r >= 4 and f <= 2:
            return "New or Recent Customers"
        if r <= 2 and f <= 2 and m <= 2:
            return "Low-Value Inactive"
        if r <= 2 and m >= 3:
            return "At Risk"
        if r <= 2:
            return "Lost or Dormant"

        return "Needs Attention"

    data["segment"] = data.apply(segment_customer, axis=1)

    return data