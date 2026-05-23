from pathlib import Path

from src.cleaning import load_raw_data, clean_transactions
from src.features import (
    build_customer_features,
    build_monthly_revenue,
    build_product_summary,
    build_country_summary,
    build_segment_summary,
)
from src.rfm import build_rfm_table, assign_rfm_segments
from src.cohort import build_cohort_retention
from src.churn import build_churn_dataset, train_churn_model
from src.basket import build_market_basket_rules
from src.sql_utils import save_tables_to_sqlite
from src.visualization import save_key_figures


RAW_PATH = Path("data/raw/Online Retail.xlsx")
PROCESSED_DIR = Path("data/processed")
DASHBOARD_DIR = Path("outputs/dashboard_exports")
FIGURE_DIR = Path("outputs/figures")
DB_PATH = Path("data/retail_analytics.db")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_PATH}. Make sure Online Retail.xlsx is inside data/raw/."
        )

    print("1. Loading raw data...")
    raw_df = load_raw_data(RAW_PATH)

    print("2. Cleaning transactions...")
    full_clean, customer_clean, cancellations = clean_transactions(raw_df)

    full_clean.to_csv(PROCESSED_DIR / "transactions_clean_full.csv", index=False)
    customer_clean.to_csv(PROCESSED_DIR / "transactions_clean_customer.csv", index=False)
    cancellations.to_csv(PROCESSED_DIR / "cancelled_transactions.csv", index=False)

    print("3. Building customer-level features...")
    customer_features = build_customer_features(customer_clean)
    customer_features.to_csv(PROCESSED_DIR / "customer_features.csv", index=False)

    print("4. Building RFM customer segmentation...")
    rfm = build_rfm_table(customer_clean)
    rfm_segments = assign_rfm_segments(rfm)
    rfm_segments.to_csv(PROCESSED_DIR / "rfm_segments.csv", index=False)

    print("5. Building cohort retention table...")
    cohort_retention = build_cohort_retention(customer_clean)
    cohort_retention.to_csv(PROCESSED_DIR / "cohort_retention.csv")

    print("6. Building churn-risk model...")
    churn_data = build_churn_dataset(customer_clean)
    churn_scores, model_metrics = train_churn_model(churn_data)
    churn_scores.to_csv(PROCESSED_DIR / "churn_scores.csv", index=False)

    print("7. Building market basket rules...")
    market_basket_rules = build_market_basket_rules(
        full_clean,
        min_support=0.02,
        max_products=500,
    )
    market_basket_rules.to_csv(PROCESSED_DIR / "market_basket_rules.csv", index=False)

    print("8. Building dashboard summary tables...")
    monthly_revenue = build_monthly_revenue(full_clean)
    product_summary = build_product_summary(full_clean)
    country_summary = build_country_summary(full_clean)
    segment_summary = build_segment_summary(rfm_segments)

    monthly_revenue.to_csv(DASHBOARD_DIR / "monthly_revenue.csv", index=False)
    product_summary.to_csv(DASHBOARD_DIR / "product_summary.csv", index=False)
    country_summary.to_csv(DASHBOARD_DIR / "country_summary.csv", index=False)
    segment_summary.to_csv(DASHBOARD_DIR / "segment_summary.csv", index=False)
    rfm_segments.to_csv(DASHBOARD_DIR / "rfm_segments.csv", index=False)
    churn_scores.to_csv(DASHBOARD_DIR / "churn_scores.csv", index=False)
    cohort_retention.to_csv(DASHBOARD_DIR / "cohort_retention.csv")
    
    print("9. Saving SQLite database...")
    save_tables_to_sqlite(
        db_path=DB_PATH,
        tables={
            "transactions_clean_full": full_clean,
            "transactions_clean_customer": customer_clean,
            "cancelled_transactions": cancellations,
            "customer_features": customer_features,
            "rfm_segments": rfm_segments,
            "cohort_retention": cohort_retention.reset_index(),
            "churn_scores": churn_scores,
            "market_basket_rules": market_basket_rules,
            "monthly_revenue": monthly_revenue,
            "product_summary": product_summary,
            "country_summary": country_summary,
            "segment_summary": segment_summary,
        },
    )

    print("10. Saving key figures...")
    save_key_figures(
        monthly_revenue=monthly_revenue,
        rfm_segments=rfm_segments,
        cohort_retention=cohort_retention,
        product_summary=product_summary,
        output_dir=FIGURE_DIR,
    )

    print("\nPipeline complete.")
    print("\nModel metrics:")
    for metric, value in model_metrics.items():
        print(f"{metric}: {value}")

    print("\nDashboard-ready files are in outputs/dashboard_exports/.")
    print("Figures are in outputs/figures/.")
    print("Processed data is in data/processed/.")


if __name__ == "__main__":
    main()