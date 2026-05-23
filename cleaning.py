import pandas as pd


def load_raw_data(path):
    """
    Loads the UCI Online Retail Excel dataset.
    """
    return pd.read_excel(path)


def clean_transactions(df):
    """
    Cleans the raw transaction-level retail data.

    Returns three datasets:

    1. full_clean:
       Valid non-cancelled sales rows, including rows without CustomerID.
       Use this for product, country, and total revenue analysis.

    2. customer_clean:
       Valid non-cancelled sales rows with CustomerID.
       Use this for RFM, cohort retention, churn scoring, and customer analytics.

    3. cancellations:
       Cancelled transactions.
       Use this for cancellation and return analysis.
    """

    data = df.copy()

    expected_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]

    missing = [col for col in expected_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    data = data[expected_columns]

    data.columns = [
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    ]

    data["invoice_no"] = data["invoice_no"].astype(str)
    data["stock_code"] = data["stock_code"].astype(str)
    data["description"] = data["description"].astype(str).str.strip()
    data["country"] = data["country"].astype(str).str.strip()
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")

    data["is_cancelled"] = data["invoice_no"].str.startswith("C")
    data["revenue"] = data["quantity"] * data["unit_price"]

    data["invoice_month"] = data["invoice_date"].dt.to_period("M").astype(str)
    data["invoice_date_only"] = data["invoice_date"].dt.date

    cancellations = data[data["is_cancelled"]].copy()

    full_clean = data[
        (~data["is_cancelled"])
        & (data["quantity"] > 0)
        & (data["unit_price"] > 0)
        & (data["invoice_date"].notna())
        & (data["description"].notna())
        & (data["description"] != "")
        & (data["description"].str.lower() != "nan")
    ].copy()

    customer_clean = full_clean.dropna(subset=["customer_id"]).copy()
    customer_clean["customer_id"] = customer_clean["customer_id"].astype(int).astype(str)

    return full_clean, customer_clean, cancellations