import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def build_market_basket_rules(df, min_support=0.02, max_products=500):
    """
    Builds product association rules using invoice-level baskets.

    To keep the analysis practical on a laptop, the function limits the basket
    to the top max_products products by total quantity sold.
    """

    data = df.copy()

    top_products = (
        data.groupby("description")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(max_products)
        .index
    )

    data = data[data["description"].isin(top_products)].copy()

    basket = (
        data.groupby(["invoice_no", "description"])["quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    basket = basket.map(lambda value: 1 if value > 0 else 0)

    frequent_itemsets = apriori(
        basket.astype(bool),
        min_support=min_support,
        use_colnames=True,
    )

    if frequent_itemsets.empty:
        return pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
            ]
        )

    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=1.0,
    )

    if rules.empty:
        return pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
            ]
        )

    rules = rules.sort_values(["lift", "confidence"], ascending=False)

    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )

    rules["consequents"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )

    return rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
        ]
    ].reset_index(drop=True)
