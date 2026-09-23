"""Step 4 - Explain the selected Logistic Regression model with SHAP."""
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.model_selection import train_test_split
from config import DATA_RAW, OUTPUT_DIR, TARGET, ID_COLUMN, LEAKAGE_COLUMNS, RANDOM_STATE, TEST_SIZE

MODEL_PATH = os.path.join(OUTPUT_DIR, "logistic_regression_model.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_RAW, low_memory=False)
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET, ID_COLUMN, "join_date", *LEAKAGE_COLUMNS]).copy()
if "home_product" in X: X["home_product"] = X["home_product"].fillna("No Home Product")

# Reproduce Step 2's exact split; model and preprocessor are loaded, never refitted.
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=TEST_SIZE / (1 - TEST_SIZE),
                                                  stratify=y_dev, random_state=RANDOM_STATE)
bundle = joblib.load(MODEL_PATH)
pipeline, threshold = bundle["pipeline"], bundle["threshold"]
preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
feature_names = preprocessor.get_feature_names_out()

# Random training rows establish the SHAP baseline; explain highest-risk test rows.
background = preprocessor.transform(X_train.sample(n=min(100, len(X_train)), random_state=RANDOM_STATE))
probabilities = pipeline.predict_proba(X_test)[:, 1]
selected = np.argsort(probabilities)[-100:][::-1]
X_selected = X_test.iloc[selected]
encoded = preprocessor.transform(X_selected)
if hasattr(background, "toarray"): background = background.toarray()
if hasattr(encoded, "toarray"): encoded = encoded.toarray()

explainer = shap.LinearExplainer(model, background)
explanation = explainer(encoded)
values = explanation.values
base = float(np.asarray(explanation.base_values).flat[0])
assert np.allclose(expit(base + values.sum(axis=1)), probabilities[selected], atol=1e-5)

# Global importance is averaged over the explained high-risk sample only.
importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap_log_odds": np.abs(values).mean(axis=0)})
importance.sort_values("mean_abs_shap_log_odds", ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, "logistic_shap_importance_top100.csv"), index=False)
shap.summary_plot(values, encoded, feature_names=feature_names, max_display=20, show=False)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "logistic_shap_summary_top100.png"), dpi=180, bbox_inches="tight"); plt.close()

# Individual explanations for chatbot lookup. Values are log-odds contributions, not causes.
rows = []
for i, (subscriber_id, probability) in enumerate(zip(df.loc[X_selected.index, ID_COLUMN], probabilities[selected])):
    order = np.argsort(np.abs(values[i]))[::-1][:10]
    for rank, j in enumerate(order, start=1):
        rows.append({ID_COLUMN: subscriber_id, "risk_probability": probability,
                     "above_selected_threshold": bool(probability >= threshold), "rank": rank,
                     "feature": feature_names[j], "encoded_value": encoded[i, j],
                     "shap_log_odds": values[i, j]})
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "logistic_shap_top100_subscribers.csv"), index=False)

shap.plots.waterfall(explanation[0], max_display=15, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_shap_highest_risk_waterfall.png"), dpi=180, bbox_inches="tight"); plt.close()
print(f"Explained {len(X_selected)} highest-risk test subscribers; baseline log-odds {base:.4f}.")
print(f"Highest-risk subscriber probability: {probabilities[selected[0]]:.2%}; threshold: {threshold:.4f}")
print(f"SHAP files saved to {OUTPUT_DIR}")
