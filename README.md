# Retail Growth & Customer Retention Analytics Platform

## Project Overview

This project analyzes 500,000+ e-commerce transactions from the UCI Online Retail dataset to identify customer value, retention patterns, churn risk, product revenue drivers, and cross-sell opportunities.

The goal of the project is to turn raw transaction-level data into business recommendations that could support growth, retention, merchandising, and customer lifecycle strategy.

The final deliverables include:

- Python analytics pipeline
- Cleaned and processed datasets
- SQLite database and SQL queries
- RFM customer segmentation
- Cohort retention analysis
- Churn-risk scoring model
- Market basket analysis
- Power BI dashboard
- Consulting-style case study

---

## Business Questions

This project answers five core business questions:

1. Which customers generate the most revenue?
2. Which customer segments are at risk of churn?
3. How well do customer cohorts retain after their first purchase?
4. Which products drive the most revenue and order volume?
5. Which product combinations create cross-sell or bundling opportunities?

---

## Dataset

The project uses the UCI Online Retail dataset, which contains transaction data from a UK-based online retailer between December 2010 and December 2011.

The raw dataset includes:

- Invoice number
- Product code
- Product description
- Quantity purchased
- Invoice date
- Unit price
- Customer ID
- Country

The raw Excel file is not included in this repository. To run the project, download the dataset and place it in:

```text
data/raw/Online Retail.xlsx
