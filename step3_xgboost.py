"""Step 3 - XGBoost model for 30-day subscriber churn."""
import math
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from config import DATA_RAW, OUTPUT_DIR, TARGET, ID_COLUMN, LEAKAGE_COLUMNS, RANDOM_STATE, TEST_SIZE

MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "xgboost_metrics.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "xgboost_predictions.csv")
MIN_RECALL = 0.85
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_RAW, low_memory=False)
assert df[TARGET].notna().all() and df[ID_COLUMN].is_unique, "Check missing targets or duplicate IDs."
y = df[TARGET].astype(int)
drop_columns = [TARGET, ID_COLUMN, "join_date", *LEAKAGE_COLUMNS]
X = df.drop(columns=drop_columns).copy()
if "home_product" in X: X["home_product"] = X["home_product"].fillna("No Home Product")
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
assert len(num_cols) + len(cat_cols) == X.shape[1], "Unrecognized feature dtype."

# 60/20/20 with TEST_SIZE=.20. Keep test rows untouched while choosing the threshold.
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=TEST_SIZE / (1 - TEST_SIZE),
                                                  stratify=y_dev, random_state=RANDOM_STATE)
print(f"Train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}; churn rate {y.mean():.2%}")

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), num_cols),
    ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
# scale_pos_weight changes probability calibration. Keep natural prevalence and tune the threshold on validation.
model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.85,
                      colsample_bytree=.85, objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", random_state=RANDOM_STATE, n_jobs=4)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
pipeline.fit(X_train, y_train)

val_probability = pipeline.predict_proba(X_val)[:, 1]
pr, re, thresholds = precision_recall_curve(y_val, val_probability)
eligible = np.flatnonzero(re[:-1] >= MIN_RECALL)
threshold = float(thresholds[eligible[np.argmax(pr[eligible])]]) if len(eligible) else 0.5
print(f"Validation threshold: {threshold:.4f}; precision {precision_score(y_val, val_probability >= threshold):.2%}; "
      f"recall {recall_score(y_val, val_probability >= threshold):.2%}")

y_probability = pipeline.predict_proba(X_test)[:, 1]
y_prediction = (y_probability >= threshold).astype(int)
ranking = pd.DataFrame({"actual": y_test.to_numpy(), "probability": y_probability}).sort_values("probability", ascending=False)
top = ranking.head(math.ceil(len(ranking) * .1))
recall_at_10 = top.actual.sum() / ranking.actual.sum()
lift_at_10 = top.actual.mean() / ranking.actual.mean()
scores = {"model": "XGBoost", "threshold": threshold, "accuracy": accuracy_score(y_test, y_prediction),
          "precision": precision_score(y_test, y_prediction, zero_division=0),
          "recall": recall_score(y_test, y_prediction, zero_division=0),
          "f1_score": f1_score(y_test, y_prediction, zero_division=0),
          "roc_auc": roc_auc_score(y_test, y_probability), "pr_auc": average_precision_score(y_test, y_probability),
          "recall_at_10": recall_at_10, "lift_at_10": lift_at_10}
pd.DataFrame([scores]).to_csv(METRICS_PATH, index=False)
pd.DataFrame({ID_COLUMN: df.loc[X_test.index, ID_COLUMN].to_numpy(), "actual_churn": y_test.to_numpy(),
              "predicted_churn": y_prediction, "churn_probability": y_probability}) \
    .sort_values("churn_probability", ascending=False).to_csv(PREDICTIONS_PATH, index=False)
joblib.dump({"pipeline": pipeline, "threshold": threshold}, MODEL_PATH)
print(pd.Series(scores).to_string())
print(classification_report(y_test, y_prediction, target_names=["Non-churn", "Churn"], zero_division=0))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_prediction), annot=True, fmt="d", cmap="Greens",
            xticklabels=["Non-churn", "Churn"], yticklabels=["Non-churn", "Churn"], ax=ax)
ax.set(xlabel="Predicted", ylabel="Actual", title="XGBoost confusion matrix")
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "xgboost_confusion_matrix.png"), dpi=200); plt.close(fig)

fpr, tpr, _ = roc_curve(y_test, y_probability)
p_curve, r_curve, _ = precision_recall_curve(y_test, y_probability)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(fpr, tpr, label=f"ROC-AUC {scores['roc_auc']:.3f}"); ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve"); ax[0].legend()
ax[1].plot(r_curve, p_curve, label=f"PR-AUC {scores['pr_auc']:.3f}")
ax[1].axhline(y_test.mean(), ls="--", color="gray", label=f"Baseline {y_test.mean():.3f}")
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "xgboost_curves.png"), dpi=200); plt.close(fig)
print(f"Saved model and results to {OUTPUT_DIR}")
