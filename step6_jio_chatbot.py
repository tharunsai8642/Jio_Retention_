"""Step 6 - Read-only Jio analytics chatbot prototype (Streamlit + SQL Server)."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from config import OUTPUT_DIR

SERVER = os.getenv("JIO_SQL_SERVER", "localhost\\SQLEXPRESS")
DATABASE = os.getenv("JIO_SQL_DATABASE", "JioRetention")
DRIVER = os.getenv("JIO_SQL_DRIVER", "ODBC Driver 18 for SQL Server")
USER, PASSWORD = os.getenv("JIO_SQL_USER"), os.getenv("JIO_SQL_PASSWORD")
SHAP_FILE = Path(OUTPUT_DIR) / "logistic_shap_top100_subscribers.csv"

@st.cache_resource
def database():
    auth = f"UID={USER};PWD={PASSWORD};" if USER and PASSWORD else "Trusted_Connection=yes;"
    connection = (f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};{auth}"
                  "Encrypt=yes;TrustServerCertificate=yes;")
    return create_engine(URL.create("mssql+pyodbc", query={"odbc_connect": connection}))

def query(sql, params=None):
    with database().connect() as con:
        return pd.read_sql_query(text(sql), con, params=params or {})

QUERIES = {
    "port_out": """SELECT TOP (5) circle, SUM(port_out_requests) AS port_out_requests,
                   ROUND(AVG(arpu_inr), 2) AS average_arpu_inr
                   FROM dbo.circle_monthly_kpi
                   WHERE month_end = (SELECT MAX(month_end) FROM dbo.circle_monthly_kpi)
                   GROUP BY circle ORDER BY port_out_requests DESC""",
    "high_risk_circle": """SELECT TOP (10) s.circle, COUNT(*) AS high_risk_subscribers
                            FROM dbo.subscribers s JOIN dbo.model_scores m
                              ON s.subscriber_id = m.subscriber_id
                            WHERE m.risk_decile = 10 GROUP BY s.circle
                            ORDER BY high_risk_subscribers DESC""",
    "monthly_churn": """SELECT month_end, SUM(churned_subscribers) AS churned_subscribers,
                       SUM(closing_base) AS closing_base,
                       ROUND(100.0 * SUM(churned_subscribers) / NULLIF(SUM(closing_base), 0), 2) AS churn_pct
                       FROM dbo.circle_monthly_kpi GROUP BY month_end ORDER BY month_end""",
    "offers": """SELECT offer_code, offer_name, target_segment, approved_circles,
                cost_per_sub_inr, valid_from, valid_to FROM dbo.offer_catalogue
                WHERE approval_status = 'Active'
                  AND (approved_circles LIKE :circle_match OR approved_circles = 'ALL')
                  AND (valid_from IS NULL OR CAST(valid_from AS date) <= CAST(GETDATE() AS date))
                  AND (valid_to IS NULL OR CAST(valid_to AS date) >= CAST(GETDATE() AS date))
                ORDER BY offer_code""",
    "subscriber": """SELECT s.subscriber_id, s.circle, s.plan_type, s.arpu_last_month_inr,
                    s.days_since_last_recharge, s.unresolved_complaints, m.risk_probability,
                    m.risk_flag, m.risk_decile
                    FROM dbo.subscribers s JOIN dbo.model_scores m
                      ON s.subscriber_id = m.subscriber_id WHERE s.subscriber_id = :subscriber_id""",
}

def route(question, subscriber_id, circle_code):
    q = question.lower().strip()
    if subscriber_id.strip(): return "subscriber", {"subscriber_id": subscriber_id.strip()}
    if "offer" in q: return "offers", {"circle_match": f"%{circle_code.strip().upper()}%"}
    if "port" in q and ("out" in q or "leave" in q): return "port_out", {}
    if "circle" in q and ("risk" in q or "decile" in q): return "high_risk_circle", {}
    if "churn" in q and ("month" in q or "trend" in q): return "monthly_churn", {}
    return None, {}

st.set_page_config(page_title="Jio Retention Assistant", layout="wide")
st.title("Jio Retention Assistant")
st.caption("Answers use the synthetic Jio project data. Model risk is a prediction, not a proven cause of churn.")
with st.sidebar:
    st.subheader("Inputs for specific questions")
    subscriber_id = st.text_input("Subscriber ID", placeholder="JIO10037241")
    circle_code = st.text_input("Offer circle code", value="BH", help="For example BH, UPE, UPW, MP")

examples = ["Top five circles by port-out last month", "Which circles have the most high-risk subscribers?",
            "Show the monthly churn trend", "What active offers are approved for Bihar?"]
question = st.selectbox("Choose a question or type your own", [""] + examples)
typed = st.text_input("Your question", placeholder="Type one of the supported questions")
if st.button("Ask"):
    chosen = typed or question
    action, params = route(chosen, subscriber_id, circle_code)
    if not action:
        st.info("Supported: latest port-outs, high-risk circles, monthly churn, active offers, or a subscriber ID.")
    else:
        try:
            result = query(QUERIES[action], params)
            if result.empty: st.warning("No matching data found. For offers, check the circle code and active dates.")
            else:
                st.dataframe(result, use_container_width=True, hide_index=True)
                if action == "subscriber":
                    risk = result.iloc[0]
                    st.write(f"Model risk score: **{risk.risk_probability:.1%}**; risk decile: **{risk.risk_decile}/10**.")
                    if SHAP_FILE.exists():
                        explanations = pd.read_csv(SHAP_FILE, dtype={"subscriber_id": str})
                        detail = explanations.loc[explanations.subscriber_id == str(risk.subscriber_id)]
                        if not detail.empty:
                            st.write("Largest contributions to this model score (positive raises log-odds):")
                            st.dataframe(detail[["feature", "shap_log_odds"]], hide_index=True)
                        else:
                            st.caption("No saved individual SHAP explanation for this subscriber; Step 4 saved only the top 100 test scores.")
                with st.expander("Verified SQL template and parameters"):
                    st.code(QUERIES[action], language="sql"); st.json(params)
        except Exception as exc:
            st.error(f"Could not query SQL Server: {exc}")
