"""Step 5 - Load the Jio data and selected model's risk scores into SQL Server."""
import os
import re

import joblib
import pandas as pd
from sqlalchemy import Unicode, create_engine, text
from sqlalchemy.engine import URL
from config import DATA_RAW, EXCEL_DATA, OUTPUT_DIR, TARGET, ID_COLUMN, LEAKAGE_COLUMNS

# Example: set JIO_SQL_SERVER=localhost\SQLEXPRESS before running.
SERVER = os.environ.get("JIO_SQL_SERVER", "localhost\\SQLEXPRESS")
DATABASE = os.environ.get("JIO_SQL_DATABASE", "JioRetention")
DRIVER = os.environ.get("JIO_SQL_DRIVER", "ODBC Driver 18 for SQL Server")
USERNAME = os.environ.get("JIO_SQL_USER")
PASSWORD = os.environ.get("JIO_SQL_PASSWORD")
SHEETS = ["service_requests", "network_sites", "circle_monthly_kpi", "circle_targets", "offer_catalogue"]
if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", DATABASE):
    raise ValueError("JIO_SQL_DATABASE must contain only letters, digits, and underscores.")
if bool(USERNAME) != bool(PASSWORD):
    raise ValueError("Set both JIO_SQL_USER and JIO_SQL_PASSWORD, or neither for Windows authentication.")

def engine_for(database, autocommit=False):
    auth = f"UID={USERNAME};PWD={PASSWORD};" if USERNAME else "Trusted_Connection=yes;"
    connection = (f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={database};"
                  f"{auth}Encrypt=yes;TrustServerCertificate=yes;")
    url = URL.create("mssql+pyodbc", query={"odbc_connect": connection})
    return create_engine(url, fast_executemany=True, isolation_level="AUTOCOMMIT" if autocommit else None)

def sql_ready(frame):
    frame = frame.copy()
    for col in frame.select_dtypes(include="bool").columns:
        frame[col] = frame[col].astype(int)
    return frame

master = engine_for("master", autocommit=True)
with master.connect() as conn:
    exists = conn.execute(text("SELECT 1 FROM sys.databases WHERE name = :name"), {"name": DATABASE}).scalar()
    if not exists: conn.exec_driver_sql(f"CREATE DATABASE [{DATABASE}]")
master.dispose()
engine = engine_for(DATABASE)

subscribers = pd.read_csv(DATA_RAW, low_memory=False)
assert subscribers[ID_COLUMN].is_unique and subscribers[TARGET].notna().all()
bundle = joblib.load(os.path.join(OUTPUT_DIR, "logistic_regression_model.pkl"))
pipeline, threshold = bundle["pipeline"], bundle["threshold"]
features = pipeline.named_steps["preprocessor"].feature_names_in_.tolist()
X = subscribers[features].copy()
if "home_product" in X: X["home_product"] = X["home_product"].fillna("No Home Product")
scores = pd.DataFrame({ID_COLUMN: subscribers[ID_COLUMN], "risk_probability": pipeline.predict_proba(X)[:, 1]})
scores["risk_flag"] = (scores.risk_probability >= threshold).astype(int)
scores["risk_decile"] = pd.qcut(scores.risk_probability.rank(method="first"), 10, labels=False).astype(int) + 1

# Keep target and future outcomes out of the table intended for chatbot lookup.
allowed = [c for c in subscribers.columns if c not in {TARGET, "join_date", *LEAKAGE_COLUMNS}]
chat_subscribers = subscribers[allowed].copy()
if "home_product" in chat_subscribers:
    chat_subscribers["home_product"] = chat_subscribers.home_product.fillna("No Home Product")

with engine.begin() as conn:
    for name, frame in [("subscribers", chat_subscribers), ("model_scores", scores)]:
        sql_ready(frame).to_sql(name, conn, schema="dbo", if_exists="replace", index=False,
                                chunksize=1000, dtype={ID_COLUMN: Unicode(64)})
        print(f"{name}: {len(frame):,} rows")
    workbook = pd.ExcelFile(EXCEL_DATA)
    for name in SHEETS:
        frame = pd.read_excel(workbook, sheet_name=name)
        types = {"circle_code": Unicode(20), "subscriber_id": Unicode(64)}
        types = {k: v for k, v in types.items() if k in frame.columns}
        sql_ready(frame).to_sql(name, conn, schema="dbo", if_exists="replace", index=False,
                                chunksize=1000, dtype=types)
        print(f"{name}: {len(frame):,} rows")
    conn.exec_driver_sql("CREATE UNIQUE INDEX IX_subscriber_id ON dbo.subscribers(subscriber_id)")
    conn.exec_driver_sql("CREATE UNIQUE INDEX IX_score_id ON dbo.model_scores(subscriber_id)")
    conn.exec_driver_sql("CREATE INDEX IX_score_decile ON dbo.model_scores(risk_decile)")
    conn.exec_driver_sql("CREATE INDEX IX_request_subscriber ON dbo.service_requests(subscriber_id)")
    conn.exec_driver_sql("CREATE INDEX IX_kpi_month_circle ON dbo.circle_monthly_kpi(month_end, circle_code)")
    result = conn.execute(text("SELECT COUNT(*) FROM dbo.subscribers s JOIN dbo.model_scores m ON s.subscriber_id = m.subscriber_id")).scalar()
    assert result == len(subscribers), "Subscriber and score tables do not match."
print(f"Loaded {result:,} subscribers and scores into SQL Server database {DATABASE}; threshold {threshold:.4f}")

