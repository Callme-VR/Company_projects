# Loan Default PD Model Summary

## Data

Source: `d:\software App\Task 3 and 4_Loan_Data.csv`

- Rows: 10,000
- Target: `default`
- Historical default rate: 18.51%
- Missing values: none
- Excluded from modeling: `customer_id`
- Model features: `credit_lines_outstanding`, `loan_amt_outstanding`, `total_debt_outstanding`, `income`, `years_employed`, `fico_score`

## Expected Loss Formula

Expected loss is calculated as:

```text
expected_loss = PD * exposure_at_default * (1 - recovery_rate)
```

The implementation assumes a 10% recovery rate by default, so loss given default is 90%.
If no separate exposure is provided, `loan_amt_outstanding` is used as exposure at default.

## Holdout Comparison

The comparison uses a deterministic stratified 80/20 train/test split.

| Model | Accuracy | Log loss | Brier score | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Baseline default rate | 0.814593 | 0.479495 | 0.151032 | 0.500000 |
| Logistic regression | 0.965517 | 0.068478 | 0.020187 | 0.999974 |
| Decision tree | 0.985007 | 0.037641 | 0.011702 | 0.998138 |

## Recommendation

Use the logistic regression in `loan_default_pd.py` as the production-style PD function because it gives smooth probability estimates and very strong discrimination. The shallow tree performed slightly better on this holdout split, but its probabilities are stepwise leaf default rates, which can be less stable for loans near split thresholds.

## Example

```python
from loan_default_pd import expected_loss, predict_pd

loan = {
    "credit_lines_outstanding": 1,
    "loan_amt_outstanding": 4500,
    "total_debt_outstanding": 7000,
    "income": 72000,
    "years_employed": 5,
    "fico_score": 640,
}

pd = predict_pd(loan)
el = expected_loss(loan)
```
