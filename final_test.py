import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# ============================================
# SORT DATA BY FICO SCORE
# ============================================

df = df.sort_values(by="fico_score").reset_index(drop=True)

fico_scores = df["fico_score"].values
defaults = df["default"].values

# ============================================
# DYNAMIC PROGRAMMING FOR OPTIMAL BUCKETING
# USING LOG-LIKELIHOOD MAXIMIZATION
# ============================================

NUM_BUCKETS = 10

n = len(df)

# Prefix sums for fast calculations
cum_defaults = np.cumsum(defaults)
cum_total = np.arange(1, n + 1)

# ============================================
# LOG-LIKELIHOOD FUNCTION
# ============================================

def bucket_log_likelihood(start, end):

    """
    Compute log-likelihood for bucket [start, end]
    """

    total = end - start + 1

    if start == 0:
        defaults_in_bucket = cum_defaults[end]
    else:
        defaults_in_bucket = cum_defaults[end] - cum_defaults[start - 1]

    non_defaults = total - defaults_in_bucket

    # Avoid division by zero
    if defaults_in_bucket == 0 or non_defaults == 0:
        return 0

    p = defaults_in_bucket / total

    ll = (
        defaults_in_bucket * np.log(p) +
        non_defaults * np.log(1 - p)
    )

    return ll

# ============================================
# DP TABLES
# ============================================

dp = np.full((NUM_BUCKETS + 1, n), -np.inf)
split = np.zeros((NUM_BUCKETS + 1, n), dtype=int)

# ============================================
# BASE CASE: 1 BUCKET
# ============================================

for i in range(n):
    dp[1][i] = bucket_log_likelihood(0, i)

# ============================================
# DYNAMIC PROGRAMMING
# ============================================

for buckets in range(2, NUM_BUCKETS + 1):

    for end in range(n):

        best_ll = -np.inf
        best_split = 0

        for mid in range(buckets - 2, end):

            current_ll = (
                dp[buckets - 1][mid] +
                bucket_log_likelihood(mid + 1, end)
            )

            if current_ll > best_ll:
                best_ll = current_ll
                best_split = mid

        dp[buckets][end] = best_ll
        split[buckets][end] = best_split

# ============================================
# RECOVER OPTIMAL BUCKET BOUNDARIES
# ============================================

boundaries = []

b = NUM_BUCKETS
idx = n - 1

while b > 1:

    s = split[b][idx]

    boundaries.append(fico_scores[s])

    idx = s
    b -= 1

boundaries = sorted(boundaries)

# ============================================
# CREATE RATING MAP
# LOWER RATING = BETTER CREDIT SCORE
# ============================================

rating_map = {}

bucket_ranges = []

lower = fico_scores.min()

for i, boundary in enumerate(boundaries):

    upper = boundary

    rating = i + 1

    bucket_ranges.append((rating, lower, upper))

    lower = upper + 1

bucket_ranges.append((NUM_BUCKETS, lower, fico_scores.max()))

# ============================================
# DISPLAY RATING MAP
# ============================================

print("\n========== FICO RATING MAP ==========\n")

for rating, low, high in bucket_ranges:

    print(f"Rating {rating}: FICO Score {low} - {high}")

# ============================================
# ASSIGN RATINGS
# ============================================

def assign_rating(fico_score):

    for rating, low, high in bucket_ranges:

        if low <= fico_score <= high:
            return rating

    return NUM_BUCKETS

# ============================================
# APPLY RATINGS TO DATASET
# ============================================

df["rating"] = df["fico_score"].apply(assign_rating)

# ============================================
# DISPLAY SAMPLE
# ============================================

print("\n========== SAMPLE DATA ==========\n")

print(df[["fico_score", "rating", "default"]].head(20))

# ============================================
# OPTIONAL:
# TRAIN MODEL USING RATING
# ============================================

X = df[["rating"]]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression()

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, probs)

print("\n========== MODEL PERFORMANCE ==========\n")
print("ROC-AUC Score:", round(auc, 5))