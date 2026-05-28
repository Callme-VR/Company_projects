import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# ===============================
# LOAD DATASET
# ===============================

file_path = "Task 3 and 4_Loan_Data.csv"
df = pd.read_csv(file_path)

# ===============================
# FEATURES & TARGET
# ===============================

X = df.drop(columns=["default", "customer_id"])
y = df["default"]

# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# LOGISTIC REGRESSION MODEL
# ===============================

log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)

# ===============================
# RANDOM FOREST MODEL
# ===============================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

# ===============================
# MODEL EVALUATION
# ===============================

# Logistic Regression
log_probs = log_model.predict_proba(X_test)[:, 1]
log_preds = log_model.predict(X_test)

log_auc = roc_auc_score(y_test, log_probs)
log_acc = accuracy_score(y_test, log_preds)

# Random Forest
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_preds = rf_model.predict(X_test)

rf_auc = roc_auc_score(y_test, rf_probs)
rf_acc = accuracy_score(y_test, rf_preds)

print("\n===== MODEL PERFORMANCE =====")

print("\nLogistic Regression")
print("ROC-AUC Score :", round(log_auc, 5))
print("Accuracy Score:", round(log_acc, 5))

print("\nRandom Forest")
print("ROC-AUC Score :", round(rf_auc, 5))
print("Accuracy Score:", round(rf_acc, 5))

# ===============================
# SELECT BEST MODEL
# ===============================

best_model = log_model

# ===============================
# EXPECTED LOSS FUNCTION
# ===============================

RECOVERY_RATE = 0.10
LGD = 1 - RECOVERY_RATE  # Loss Given Default

def calculate_expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):

    input_data = pd.DataFrame({
        "credit_lines_outstanding": [credit_lines_outstanding],
        "loan_amt_outstanding": [loan_amt_outstanding],
        "total_debt_outstanding": [total_debt_outstanding],
        "income": [income],
        "years_employed": [years_employed],
        "fico_score": [fico_score]
    })

    # Predict Probability of Default (PD)
    pd_probability = best_model.predict_proba(input_data)[0][1]

    # Expected Loss Formula
    expected_loss = pd_probability * LGD * loan_amt_outstanding

    return {
        "Probability_of_Default": round(pd_probability, 4),
        "Expected_Loss": round(expected_loss, 2)
    }

# ===============================
# EXAMPLE PREDICTION
# ===============================

result = calculate_expected_loss(
    credit_lines_outstanding=3,
    loan_amt_outstanding=15000,
    total_debt_outstanding=10000,
    income=55000,
    years_employed=5,
    fico_score=620
)

print("\n===== SAMPLE PREDICTION =====")
print(result)