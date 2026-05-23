import pandas as pd


def build_cohort_retention(df):
    """
    Builds monthly cohort retention table.

    Customers are grouped by first purchase month.
    Each value shows the share of customers from that cohort
    who purchased again in each later month.
    """

    data = df.copy()

    data["order_month"] = data["invoice_date"].dt.to_period("M")

    data["cohort_month"] = (
        data.groupby("customer_id")["invoice_date"]
        .transform("min")
        .dt.to_period("M")
    )

    data["cohort_index"] = (
        (data["order_month"].dt.year - data["cohort_month"].dt.year) * 12
        + (data["order_month"].dt.month - data["cohort_month"].dt.month)
        + 1
    )

    cohort_counts = (
        data.groupby(["cohort_month", "cohort_index"])["customer_id"]
        .nunique()
        .reset_index()
    )

    cohort_pivot = cohort_counts.pivot_table(
        index="cohort_month",
        columns="cohort_index",
        values="customer_id",
    )

    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0)

    retention.index = retention.index.astype(str)
    retention.columns = [f"month_{int(col)}" for col in retention.columns]

    return retention