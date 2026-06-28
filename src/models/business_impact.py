import pandas as pd

from src.config.settings import POSITIVE_CLASS


def _as_positive_mask(y_true):
    return pd.Series(y_true).reset_index(drop=True).eq(POSITIVE_CLASS)


def _customer_values(X):
    if "CLTV" not in X:
        return pd.Series([0.0] * len(X))
    return pd.to_numeric(X["CLTV"], errors="coerce").fillna(0.0).reset_index(drop=True)


def churn_capture_by_decile(X, y_true, probabilities):
    ranked = pd.DataFrame(
        {
            "probability": probabilities,
            "is_churner": _as_positive_mask(y_true),
            "customer_value": _customer_values(X),
        }
    ).sort_values("probability", ascending=False, ignore_index=True)

    total_churners = int(ranked["is_churner"].sum())
    if ranked.empty:
        return []

    ranked["decile"] = [
        f"Top {min(10, int(index * 10 / len(ranked)) + 1) * 10}%"
        for index in ranked.index
    ]

    rows = []
    cumulative_churners = 0
    cumulative_value = 0.0
    decile_order = [f"Top {i * 10}%" for i in range(1, 11)]
    for decile in decile_order:
        group = ranked[ranked["decile"] == decile]
        if group.empty:
            continue
        churners = int(group["is_churner"].sum())
        value = float(group.loc[group["is_churner"], "customer_value"].sum())
        cumulative_churners += churners
        cumulative_value += value
        rows.append(
            {
                "segment": str(decile),
                "customers": int(len(group)),
                "churners": churners,
                "churn_capture_rate": round(cumulative_churners / total_churners, 4)
                if total_churners
                else 0.0,
                "at_risk_cltv": round(value, 2),
                "cumulative_at_risk_cltv": round(cumulative_value, 2),
            }
        )

    return rows


def campaign_scenarios(
    X,
    y_true,
    probabilities,
    target_rates=(0.1, 0.2, 0.3),
    retention_success_rate=0.15,
    contact_cost_per_customer=10.0,
):
    ranked = pd.DataFrame(
        {
            "probability": probabilities,
            "is_churner": _as_positive_mask(y_true),
            "customer_value": _customer_values(X),
        }
    ).sort_values("probability", ascending=False, ignore_index=True)

    scenarios = []
    total_churners = int(ranked["is_churner"].sum())
    for target_rate in target_rates:
        target_count = max(1, int(round(len(ranked) * target_rate))) if len(ranked) else 0
        targeted = ranked.head(target_count)
        captured_churners = int(targeted["is_churner"].sum())
        at_risk_value = float(targeted.loc[targeted["is_churner"], "customer_value"].sum())
        expected_saved_value = at_risk_value * retention_success_rate
        campaign_cost = target_count * contact_cost_per_customer

        scenarios.append(
            {
                "target_rate": target_rate,
                "targeted_customers": target_count,
                "captured_churners": captured_churners,
                "churn_capture_rate": round(captured_churners / total_churners, 4)
                if total_churners
                else 0.0,
                "at_risk_cltv": round(at_risk_value, 2),
                "retention_success_rate": retention_success_rate,
                "expected_saved_value": round(expected_saved_value, 2),
                "campaign_cost": round(campaign_cost, 2),
                "estimated_net_value": round(expected_saved_value - campaign_cost, 2),
            }
        )

    return scenarios


def build_business_impact_report(X, y_true, probabilities):
    return {
        "assumptions": {
            "retention_success_rate": 0.15,
            "contact_cost_per_customer": 10.0,
            "customer_value_field": "CLTV",
        },
        "decile_capture": churn_capture_by_decile(X, y_true, probabilities),
        "campaign_scenarios": campaign_scenarios(X, y_true, probabilities),
    }
