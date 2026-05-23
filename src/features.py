import pandas as pd


def build_customer_features(df):
    """
    Creates customer-level behavioral and revenue features.
    """

    data = df.copy()
    snapshot_date = data["invoice_date"].max() + pd.Timedelta(days=1)

    customer = data.groupby("customer_id").agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        total_orders=("invoice_no", "nunique"),
        total_items=("quantity", "sum"),
        total_revenue=("revenue", "sum"),
        avg_line_revenue=("revenue", "mean"),
        unique_products=("stock_code", "nunique"),
        active_months=("invoice_month", "nunique"),
        country=("country", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
    ).reset_index()

    customer["recency_days"] = (snapshot_date - customer["last_purchase"]).dt.days

    customer["customer_lifetime_days"] = (
        customer["last_purchase"] - customer["first_purchase"]
    ).dt.days + 1

    customer["orders_per_active_month"] = (
        customer["total_orders"] / customer["active_months"]
    )

    customer["revenue_per_order"] = (
        customer["total_revenue"] / customer["total_orders"]
    )

    customer["items_per_order"] = (
        customer["total_items"] / customer["total_orders"]
    )

    return customer


def build_monthly_revenue(df):
    """
    Creates monthly revenue summary for trend analysis.
    """

    monthly = df.groupby("invoice_month").agg(
        monthly_revenue=("revenue", "sum"),
        total_orders=("invoice_no", "nunique"),
        total_customers=("customer_id", "nunique"),
        total_items=("quantity", "sum"),
    ).reset_index()

    monthly["average_order_value"] = (
        monthly["monthly_revenue"] / monthly["total_orders"]
    )

    return monthly


def build_product_summary(df):
    """
    Creates product-level sales summary, excluding non-product charges
    such as postage, manual adjustments, and fees.
    """

    data = df.copy()

    non_product_terms = [
        "POSTAGE",
        "DOTCOM POSTAGE",
        "Manual",
        "MANUAL",
        "CARRIAGE",
        "AMAZON FEE",
        "BANK CHARGES",
        "Discount",
        "SAMPLES",
    ]

    data = data[
        ~data["description"].isin(non_product_terms)
    ].copy()

    product = data.groupby(["stock_code", "description"]).agg(
        total_revenue=("revenue", "sum"),
        total_quantity=("quantity", "sum"),
        total_orders=("invoice_no", "nunique"),
        avg_unit_price=("unit_price", "mean"),
    ).reset_index()

    product = product[product["total_orders"] >= 25].copy()

    product = product.sort_values("total_revenue", ascending=False)
    
    return product


def build_country_summary(df):
    """
    Creates country-level revenue summary.
    """

    country = df.groupby("country").agg(
        total_revenue=("revenue", "sum"),
        total_orders=("invoice_no", "nunique"),
        total_customers=("customer_id", "nunique"),
        total_items=("quantity", "sum"),
    ).reset_index()

    country["average_order_value"] = (
        country["total_revenue"] / country["total_orders"]
    )

    country = country.sort_values("total_revenue", ascending=False)

    return country

def build_segment_summary(rfm_segments):
    """
    Creates segment-level customer and revenue summary.
    """

    segment = rfm_segments.groupby("segment").agg(
        customers=("customer_id", "nunique"),
        total_revenue=("monetary", "sum"),
        avg_customer_value=("monetary", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_recency=("recency", "mean"),
    ).reset_index()

    total_revenue = segment["total_revenue"].sum()
    total_customers = segment["customers"].sum()

    segment["revenue_share"] = segment["total_revenue"] / total_revenue
    segment["customer_share"] = segment["customers"] / total_customers

    segment = segment.sort_values("total_revenue", ascending=False)

    return segment