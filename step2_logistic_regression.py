"""Step 2 - Logistic Regression with the same 60/20/20 split as XGBoost."""

import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DATA_RAW, OUTPUT_DIR, TARGET, ID_COLUMN, LEAKAGE_COLUMNS, RANDOM_STATE, TEST_SIZE
)

MODEL_PATH = os.path.join(OUTPUT_DIR, "logistic_regression_model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "logistic_regression_metrics.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "logistic_regression_predictions.csv")
MIN_RECALL = 0.85
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data and exclude identifiers, future outcomes, and dates.
df = pd.read_csv(DATA_RAW, low_memory=False)
assert df[TARGET].notna().all() and df[ID_COLUMN].is_unique

y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET, ID_COLUMN, "join_date", *LEAKAGE_COLUMNS]).copy()
if "home_product" in X:
    X["home_product"] = X["home_product"].fillna("No Home Product")

numerical = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(include=["object", "str", "bool"]).columns.tolist()

# Match the two splits and random states in step3_xgboost.py exactly.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=TEST_SIZE / (1 - TEST_SIZE),
    stratify=y_dev, random_state=RANDOM_STATE
)

print(f"Train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}")
print(f"Churn rate: {y.mean():.2%}")

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numerical),
    ("categorical", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced", max_iter=2000,
        solver="liblinear", random_state=RANDOM_STATE
    )),
])
pipeline.fit(X_train, y_train)

# Choose the highest-precision threshold that meets 85% recall on validation.
val_probability = pipeline.predict_proba(X_val)[:, 1]
val_precision, val_recall, thresholds = precision_recall_curve(y_val, val_probability)
eligible = np.flatnonzero(val_recall[:-1] >= MIN_RECALL)
threshold = float(thresholds[eligible[np.argmax(val_precision[eligible])]])

print(f"\nValidation threshold: {threshold:.4f}")
print(f"Validation precision: {precision_score(y_val, val_probability >= threshold):.2%}")
print(f"Validation recall: {recall_score(y_val, val_probability >= threshold):.2%}")

# Evaluate the selected threshold on the untouched test set.
probability = pipeline.predict_proba(X_test)[:, 1]
prediction = (probability >= threshold).astype(int)

ranking = pd.DataFrame({
    "actual": y_test.to_numpy(), "probability": probability
}).sort_values("probability", ascending=False)
top_decile = ranking.head(math.ceil(len(ranking) * 0.10))
recall_at_10 = top_decile["actual"].sum() / ranking["actual"].sum()
lift_at_10 = top_decile["actual"].mean() / ranking["actual"].mean()

metrics = {
    "model": "Logistic Regression",
    "threshold": threshold,
    "accuracy": accuracy_score(y_test, prediction),
    "precision": precision_score(y_test, prediction, zero_division=0),
    "recall": recall_score(y_test, prediction, zero_division=0),
    "f1_score": f1_score(y_test, prediction, zero_division=0),
    "roc_auc": roc_auc_score(y_test, probability),
    "pr_auc": average_precision_score(y_test, probability),
    "recall_at_10": recall_at_10,
    "lift_at_10": lift_at_10,
}

pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
pd.DataFrame({
    ID_COLUMN: df.loc[X_test.index, ID_COLUMN].to_numpy(),
    "actual_churn": y_test.to_numpy(),
    "predicted_churn": prediction,
    "churn_probability": probability,
}).sort_values("churn_probability", ascending=False).to_csv(
    PREDICTIONS_PATH, index=False
)
joblib.dump({"pipeline": pipeline, "threshold": threshold}, MODEL_PATH)

print("\nTEST RESULTS")
print(pd.Series(metrics).to_string())
print("\nClassification report:")
print(classification_report(
    y_test, prediction, target_names=["Non-churn", "Churn"], zero_division=0
))
print(f"\nSaved results in: {OUTPUT_DIR}")