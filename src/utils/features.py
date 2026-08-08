def engineer_features(raw: dict) -> dict:
    """Turn raw applicant fields into the same engineered features used at training time."""
    
    # Extract lists of values
    bill_cols = [raw[f"BILL_AMT{i}"] for i in range(1, 7)]
    pay_amt_cols = [raw[f"PAY_AMT{i}"] for i in range(1, 7)]
    pay_status_cols = [raw["PAY_0"]] + [raw[f"PAY_{i}"] for i in range(2, 7)]
    
    # Calculate aggregates first to avoid redundant loops
    total_billed = sum(bill_cols)
    total_paid = sum(pay_amt_cols)
    n_months = len(bill_cols)
    
    # Calculate ratios and averages using the pre-computed totals
    avg_bill = total_billed / n_months
    utilization_ratio = round(avg_bill / raw["LIMIT_BAL"], 4) if raw["LIMIT_BAL"] else 0.0
    payment_ratio = round(total_paid / total_billed, 4) if total_billed != 0 else 1.0
    
    return {
        **raw,
        "utilization_ratio": utilization_ratio,
        "avg_pay_delay": round(sum(pay_status_cols) / n_months, 3),
        "max_pay_delay": max(pay_status_cols),
        "months_late": sum(1 for p in pay_status_cols if p > 0),
        "payment_ratio": payment_ratio,
        "delay_trend": raw["PAY_0"] - raw["PAY_6"],
    }    