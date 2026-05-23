--Monthly Revenue Trend
SELECT 
    invoice_month,
    ROUND(SUM(revenue), 2) AS monthly_revenue,
    COUNT(DISTINCT invoice_no) AS orders,
    COUNT(DISTINCT customer_id) AS customers
FROM transactions_clean_full
GROUP BY invoice_month
ORDER BY invoice_month;



--Revenue by Customer Segment
SELECT
    segment,
    COUNT(DISTINCT customer_id) AS customers,
    ROUND(SUM(monetary), 2) AS total_revenue,
    ROUND(AVG(monetary), 2) AS avg_customer_value
FROM rfm_segments
GROUP BY segment
ORDER BY total_revenue DESC;



--Priority High-Risk Customers
SELECT
    customer_id,
    monetary,
    frequency,
    recency_days,
    churn_risk_score,
    risk_tier,
    recommended_action
FROM churn_scores
WHERE risk_tier = 'High Risk'
ORDER BY monetary DESC
LIMIT 25;



--Top Product Revenue Drivers
SELECT
    description,
    ROUND(SUM(total_revenue), 2) AS revenue,
    SUM(total_quantity) AS quantity_sold,
    SUM(total_orders) AS orders
FROM product_summary
GROUP BY description
ORDER BY revenue DESC
LIMIT 10;



--Cross-Sell Opportunities
SELECT
    antecedents,
    consequents,
    ROUND(support, 4) AS support,
    ROUND(confidence, 4) AS confidence,
    ROUND(lift, 2) AS lift
FROM market_basket_rules
ORDER BY lift DESC, confidence DESC
LIMIT 20;
