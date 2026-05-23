from pathlib import Path

import matplotlib.pyplot as plt


def save_key_figures(
    monthly_revenue,
    rfm_segments,
    cohort_retention,
    product_summary,
    output_dir,
):
    """
    Saves a few portfolio-ready figures.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_monthly_revenue_chart(
        monthly_revenue,
        output_dir / "monthly_revenue.png",
    )

    save_segment_revenue_chart(
        rfm_segments,
        output_dir / "segment_revenue.png",
    )

    save_top_products_chart(
        product_summary,
        output_dir / "top_products.png",
    )

    save_cohort_heatmap(
        cohort_retention,
        output_dir / "cohort_retention_heatmap.png",
    )


def save_monthly_revenue_chart(monthly_revenue, path):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        monthly_revenue["invoice_month"],
        monthly_revenue["monthly_revenue"],
        marker="o",
    )

    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_segment_revenue_chart(rfm_segments, path):
    segment_revenue = (
        rfm_segments.groupby("segment")["monetary"]
        .sum()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        segment_revenue.index,
        segment_revenue.values,
    )

    ax.set_title("Revenue by Customer Segment")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Segment")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_top_products_chart(product_summary, path):
    top_products = product_summary.head(10).sort_values(
        "total_revenue",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        top_products["description"],
        top_products["total_revenue"],
    )

    ax.set_title("Top 10 Products by Revenue")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Product")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_cohort_heatmap(cohort_retention, path):
    fig, ax = plt.subplots(figsize=(12, 7))

    image = ax.imshow(
        cohort_retention.values,
        aspect="auto",
    )

    ax.set_title("Monthly Cohort Retention")
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort Month")

    ax.set_xticks(range(len(cohort_retention.columns)))
    ax.set_xticklabels(
        cohort_retention.columns,
        rotation=45,
    )

    ax.set_yticks(range(len(cohort_retention.index)))
    ax.set_yticklabels(cohort_retention.index)

    fig.colorbar(image, ax=ax, label="Retention Rate")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)