# Jio Retention Analytics and Assistant

This project explores a synthetic subscriber dataset, compares 30-day churn models, explains selected Logistic Regression predictions with SHAP, loads reporting tables into SQL Server, and answers approved questions in a Streamlit chat app.

## Results

The dataset has 64,738 subscribers; 1,122 (1.73%) have the 30-day churn label. On the held-out test split, Logistic Regression achieved PR-AUC **0.2974**, ROC-AUC **0.9615**, precision **18.70%**, recall **89.73%**, and top-decile recall **91.96%**. XGBoost achieved PR-AUC **0.2659**, ROC-AUC **0.9558**, precision **14.46%**, recall **90.18%**, and top-decile recall **89.29%**. Thresholds were selected on validation data to target at least 85% recall; test metrics were measured once using those thresholds. Logistic Regression was selected for the assistant.

The churn class is rare, so PR-AUC, precision, recall, and top-decile recall are more useful than accuracy alone. The class-weighted model scores are not calibrated probabilities. These results describe the synthetic project data and do not establish causes of churn.

## Run locally

1. Install Python dependencies: `pip install pandas numpy scipy matplotlib seaborn scikit-learn xgboost shap joblib openpyxl sqlalchemy pyodbc streamlit openai`.
2. Install **ODBC Driver 18 for SQL Server**, and have SQL Server running locally. Set `JIO_SQL_SERVER` to your actual SQL Server instance. The defaults in the provided scripts are `localhost\\SQLEXPRESS` and database `JioRetention`.
3. Place `config.py`, `subscribers.csv`, and `Jio_Retention_Dataset.xlsx` in the project root. Ensure the Excel workbook contains `service_requests`, `network_sites`, `circle_monthly_kpi`, `circle_targets`, and `offer_catalogue` sheets.
4. Run `python step1_eda.py`, `python step2_logistic_regression.py`, `python step3_xgboost.py`, and `python step4_churn_shap.py` in that order. The model scripts create files under `outputs/`; step 4 and step 5 require `outputs/logistic_regression_model.pkl` saved by step 2. Inspect outputs and confirm the model scripts use the same train/validation/test split.
5. Set your server in PowerShell: `$env:JIO_SQL_SERVER = "YOUR-SERVER\\SQLEXPRESS"`. Run `python step5_build_sqlserver.py` to create/load the database and seven tables. **This script replaces its tables**; do not run it against a database whose tables you need to preserve. Windows authentication is used unless both `JIO_SQL_USER` and `JIO_SQL_PASSWORD` are set.
6. Start either `streamlit run step6_jio_chatbot.py` (fixed question interface) or `streamlit run step7_agentic_chatbot.py` (chat interface). Step 7 uses local keyword routing when `OPENAI_API_KEY` is unset. Set a valid key in your environment to enable LLM question routing; the LLM selects among five approved SQL templates and does not write SQL.

Example review questions: `Which circles have the most high-risk subscribers?` and `Top five circles by port-out last month`. The latter means **the most recent month present in the workbook**, not necessarily the current calendar month. For a subscriber question, use an ID present in the loaded dataset. Offer questions need a circle and an offer with valid dates.

## What to submit

- Source files `config.py`, `step1_eda.py` through `step7_agentic_chatbot.py`, and a pinned `requirements.txt` generated from the tested environment.
- This README, the project report, and screenshots or a short screen recording of EDA, model metrics, SHAP, SSMS tables, and working chat questions.
- The synthetic data/workbook only if redistribution is allowed. If you omit the data, say that reviewers can inspect results and the demo but cannot reproduce training or database loading without it.

## Limits and access

The current app connects to a SQL Server instance on the developer's Windows computer. Reviewers cannot access that database from a GitHub link or a cloud app. A cloud demo needs a reachable hosted database and restricted read-only credentials; an OpenAI key must be configured as a secret. The SQL load uses training-data scores for the full dataset as a demonstration, while the reported evaluation metrics come from the held-out test split. The saved SHAP subscriber explanations cover 100 selected high-risk test subscribers; other IDs may not have a saved explanation. The chatbot supports five approved query types and does not provide document retrieval or unrestricted SQL queries.
