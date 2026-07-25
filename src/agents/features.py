def engineer_features(raw: dict) -> dict:
    """Turn raw applicant fields into the same engineered features used at training time."""
    bill_cols = [raw[f"BILL_AMT{i}"] for i in range(1, 7)]
    pay_amt_cols = [raw[f"PAY_AMT{i}"] for i in range(1, 7)]
    pay_status_cols = [raw["PAY_0"]] + [raw[f"PAY_{i}"] for i in range(2, 7)]
    avg_bill = sum(bill_cols) / 6
    utilization_ratio = round(avg_bill / raw["LIMIT_BAL"], 4) if raw["LIMIT_BAL"] else 0.0
    avg_pay_delay = round(sum(pay_status_cols) / 6, 3)
    max_pay_delay = max(pay_status_cols)
    months_late = sum(1 for p in pay_status_cols if p > 0)
    total_paid = sum(pay_amt_cols)
    total_billed = sum(bill_cols)
    payment_ratio = round(total_paid / total_billed, 4) if total_billed > 0 else 1.0
    delay_trend = raw["PAY_0"] - raw["PAY_6"]
    return {
        **raw,
        "utilization_ratio": utilization_ratio,
        "avg_pay_delay": avg_pay_delay,
        "max_pay_delay": max_pay_delay,
        "months_late": months_late,
        "payment_ratio": payment_ratio,
        "delay_trend": delay_trend,
    }