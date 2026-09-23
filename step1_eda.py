"""Step 1 - Day 1 EDA for Jio Subscriber Retention."""

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATA_RAW, EXCEL_DATA, OUTPUT_DIR, TARGET


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ============================================================
# 1. LOAD SUBSCRIBER DATA
# ============================================================
df = pd.read_csv(
    DATA_RAW,
    na_values=["", "NA", "N/A", "None", "null", "-", "?"],
    low_memory=False,
)

print("=" * 70)
print("JIO SUBSCRIBER RETENTION - DAY 1 EDA")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"Duplicate rows: {df.duplicated().sum():,}")
print(f"Unique subscribers: {df['subscriber_id'].nunique():,}")


# ============================================================
# 2. DATA TYPES AND DATE CONVERSION
# ============================================================
print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(df.dtypes.to_string())

for column in ["join_date", "churn_date"]:
    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

print("\nDate range:")
print(f"First joining date: {df['join_date'].min()}")
print(f"Latest joining date: {df['join_date'].max()}")


# ============================================================
# 3. MISSING VALUES
# ============================================================
missing_report = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": (
        df.isna().mean().mul(100).round(2)
    ),
})

missing_report = missing_report[
    missing_report["missing_count"] > 0
].sort_values(
    "missing_percentage",
    ascending=False,
)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

if missing_report.empty:
    print("No missing values found.")
else:
    print(missing_report.to_string())

missing_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_values_report.csv",
    ),
    index_label="column",
)


# ============================================================
# 4. DATA QUALITY CHECKS
# ============================================================
print("\n" + "=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

print(f"Duplicate rows: {df.duplicated().sum():,}")
print(
    "Duplicate subscriber IDs:",
    f"{df['subscriber_id'].duplicated().sum():,}",
)
print(
    "Missing subscriber IDs:",
    f"{df['subscriber_id'].isna().sum():,}",
)
print(
    "Missing target values:",
    f"{df[TARGET].isna().sum():,}",
)


# ============================================================
# 5. NUMERICAL SUMMARY
# ============================================================
numeric_summary = df.describe(
    include="number",
).T.round(2)

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)
print(numeric_summary.to_string())

numeric_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "numerical_summary.csv",
    )
)


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================
target_count = df[TARGET].value_counts()
target_percentage = (
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

target_report = pd.DataFrame({
    "count": target_count,
    "percentage": target_percentage,
})

churn_rate = df[TARGET].mean() * 100

print("\n" + "=" * 70)
print("30-DAY CHURN DISTRIBUTION")
print("=" * 70)

print(target_report.to_string())
print(f"\nOverall 30-day churn rate: {churn_rate:.2f}%")

plt.figure(figsize=(7, 5))

ax = sns.countplot(
    data=df,
    x=TARGET,
    color="#1565C0",
)

for container in ax.containers:
    ax.bar_label(container)

plt.title("30-Day Churn Distribution")
plt.xlabel("Churned Within 30 Days")
plt.ylabel("Subscribers")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "target_distribution.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# 7. PROBLEMATIC REGIONS
# ============================================================
circle_analysis = (
    df.groupby(
        ["circle_code", "circle"],
        as_index=False,
    )
    .agg(
        subscribers=("subscriber_id", "count"),
        churned_subscribers=(TARGET, "sum"),
        churn_rate=(TARGET, "mean"),
        average_arpu=(
            "arpu_last_month_inr",
            "mean",
        ),
        average_complaints=(
            "complaints_6m",
            "mean",
        ),
        average_unresolved_complaints=(
            "unresolved_complaints",
            "mean",
        ),
    )
)

circle_analysis["churn_rate"] *= 100

circle_analysis = circle_analysis.sort_values(
    "churn_rate",
    ascending=False,
)

print("\n" + "=" * 70)
print("CHURN BY CIRCLE")
print("=" * 70)

print(
    circle_analysis.round(2).to_string(
        index=False,
    )
)

circle_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "circle_churn_analysis.csv",
    ),
    index=False,
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=circle_analysis,
    x="churn_rate",
    y="circle",
    color="#D32F2F",
)

plt.title("30-Day Churn Rate by Circle")
plt.xlabel("Churn Rate (%)")
plt.ylabel("Circle")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_by_circle.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# 8. CHURN REASONS
# ============================================================
churn_reasons = (
    df.loc[df[TARGET] == True, "churn_reason"]
    .value_counts()
    .rename_axis("churn_reason")
    .reset_index(name="subscriber_count")
)

print("\n" + "=" * 70)
print("CHURN REASONS")
print("=" * 70)

print(churn_reasons.to_string(index=False))

churn_reasons.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "churn_reasons.csv",
    ),
    index=False,
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=churn_reasons,
    x="subscriber_count",
    y="churn_reason",
    color="#EF6C00",
)

plt.title("Main Reasons for 30-Day Churn")
plt.xlabel("Churned Subscribers")
plt.ylabel("Churn Reason")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_reasons.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# 9. MONTHLY CHURN TREND
# ============================================================
kpi = pd.read_excel(
    EXCEL_DATA,
    sheet_name="circle_monthly_kpi",
)

kpi["month_end"] = pd.to_datetime(
    kpi["month_end"],
    errors="coerce",
)

monthly_trend = (
    kpi.groupby(
        "month_end",
        as_index=False,
    )
    .agg(
        closing_base=("closing_base", "sum"),
        churned_subscribers=(
            "churned_subscribers",
            "sum",
        ),
        port_out_requests=(
            "port_out_requests",
            "sum",
        ),
        complaints_logged=(
            "complaints_logged",
            "sum",
        ),
    )
)

monthly_trend["churn_rate_pct"] = (
    monthly_trend["churned_subscribers"]
    / monthly_trend["closing_base"]
    * 100
)

print("\n" + "=" * 70)
print("MONTHLY CHURN TREND")
print("=" * 70)

print(
    monthly_trend.round(2).to_string(
        index=False,
    )
)

monthly_trend.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "monthly_churn_trend.csv",
    ),
    index=False,
)


# ---------- Monthly trend charts ----------
fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 5),
)

sns.lineplot(
    data=monthly_trend,
    x="month_end",
    y="churn_rate_pct",
    marker="o",
    color="#D32F2F",
    ax=axes[0],
)

axes[0].set_title("Monthly Churn Rate")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Churn Rate (%)")

sns.lineplot(
    data=monthly_trend,
    x="month_end",
    y="complaints_logged",
    marker="o",
    color="#EF6C00",
    ax=axes[1],
)

axes[1].set_title("Monthly Complaints")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Complaints Logged")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "monthly_churn_and_complaints.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# 10. CHURN TIMELINE FOR PROBLEMATIC CIRCLES
# ============================================================
top_circle_codes = (
    circle_analysis.head(5)["circle_code"]
    .tolist()
)

top_circle_trends = kpi[
    kpi["circle_code"].isin(top_circle_codes)
].copy()

plt.figure(figsize=(13, 7))

sns.lineplot(
    data=top_circle_trends,
    x="month_end",
    y="monthly_churn_pct",
    hue="circle_code",
    marker="o",
)

plt.title("Monthly Churn Trend for Problematic Circles")
plt.xlabel("Month")
plt.ylabel("Monthly Churn Rate (%)")
plt.legend(title="Circle")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "problematic_circle_churn_trends.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# 11. COMPLAINT AND CHURN ANALYSIS
# ============================================================
df["complaint_group"] = pd.cut(
    df["complaints_6m"],
    bins=[-1, 0, 1, float("inf")],
    labels=[
        "No complaints",
        "1 complaint",
        "2+ complaints",
    ],
)

complaint_analysis = (
    df.groupby(
        "complaint_group",
        observed=False,
        as_index=False,
    )
    .agg(
        subscribers=("subscriber_id", "count"),
        churned_subscribers=(TARGET, "sum"),
        churn_rate=(TARGET, "mean"),
        average_unresolved=(
            "unresolved_complaints",
            "mean",
        ),
        average_resolution_days=(
            "avg_resolution_days",
            "mean",
        ),
    )
)

complaint_analysis["churn_rate"] *= 100

print("\n" + "=" * 70)
print("COMPLAINT AND CHURN ANALYSIS")
print("=" * 70)

print(
    complaint_analysis.round(2).to_string(
        index=False,
    )
)

complaint_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "complaint_churn_analysis.csv",
    ),
    index=False,
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=complaint_analysis,
    x="complaint_group",
    y="churn_rate",
    color="#7B1FA2",
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f%%",
    )

plt.title("Churn Rate by Complaint Group")
plt.xlabel("Complaint Group")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_by_complaint_group.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# 12. COMPLAINT CORRELATIONS
# ============================================================
complaint_columns = [
    "complaints_6m",
    "unresolved_complaints",
    "avg_resolution_days",
    TARGET,
]

complaint_correlations = (
    df[complaint_columns]
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .sort_values(
        key=abs,
        ascending=False,
    )
)

print("\n" + "=" * 70)
print("COMPLAINT CORRELATIONS WITH CHURN")
print("=" * 70)

print(
    complaint_correlations
    .round(4)
    .to_string()
)

complaint_correlations.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "complaint_correlations.csv",
    ),
    header=["correlation_with_churn"],
)


# ============================================================
# 13. IMPORTANT CUSTOMER PATTERNS
# ============================================================
pattern_columns = [
    "mnp_enquiry_flag",
    "days_since_last_recharge",
    "avg_recharge_gap_days",
    "recharge_count_6m",
    "payment_failures_6m",
    "unresolved_complaints",
    "complaints_6m",
    "app_logins_30d",
    "arpu_last_month_inr",
    "site_congestion_score",
    "avg_sinr_db",
    "drop_call_rate_pct",
    "outgoing_to_competitor_pct",
    TARGET,
]

customer_patterns = (
    df[pattern_columns]
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .sort_values(
        key=abs,
        ascending=False,
    )
)

print("\n" + "=" * 70)
print("IMPORTANT CUSTOMER PATTERNS")
print("=" * 70)

print(
    customer_patterns
    .round(4)
    .to_string()
)

customer_patterns.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "important_customer_patterns.csv",
    ),
    header=["correlation_with_churn"],
)


# ============================================================
# 14. RECHARGE BEHAVIOUR
# ============================================================
df["recharge_recency_group"] = pd.cut(
    df["days_since_last_recharge"], bins=[-1, 7, 30, 60, float("inf")],
    labels=["0-7 days", "8-30 days", "31-60 days", "61+ days"],
)

recharge_analysis = (
    df.groupby("recharge_recency_group", observed=False, as_index=False)
    .agg(
        subscribers=("subscriber_id", "count"),
        churned_subscribers=(TARGET, "sum"),
        average_recharges_6m=("recharge_count_6m", "mean"),
        average_recharge_gap_days=("avg_recharge_gap_days", "mean"),
    )
)
recharge_analysis["churn_rate"] = (
    recharge_analysis["churned_subscribers"] / recharge_analysis["subscribers"] * 100
)

print("\n" + "=" * 70)
print("RECHARGE BEHAVIOUR")
print("=" * 70)
print(recharge_analysis.round(2).to_string(index=False))

recharge_analysis.to_csv(
    os.path.join(OUTPUT_DIR, "recharge_behaviour_analysis.csv"), index=False,
)

plt.figure(figsize=(9, 5))
ax = sns.barplot(
    data=recharge_analysis, x="recharge_recency_group", y="churn_rate", color="#1565C0",
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")

plt.title("Churn Rate by Recharge Recency")
plt.xlabel("Days Since Last Recharge")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "churn_by_recharge_recency.png"), dpi=300)
plt.close()


# ============================================================
# 15. SUBSCRIBER SEGMENTATION
# ============================================================
df["arpu_segment"] = pd.cut(
    df["arpu_last_month_inr"], bins=[0, 150, 250, 500, float("inf")],
    labels=["Low ARPU", "Medium ARPU", "High ARPU", "Premium ARPU"],
    include_lowest=True,
)

df["tenure_segment"] = pd.cut(
    df["tenure_months"], bins=[0, 6, 12, 24, 48, float("inf")],
    labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"],
    include_lowest=True,
)


def analyse_segment(column, segment_type):
    """Calculate count and churn rate for one subscriber segment."""
    result = (
        df.groupby(column, observed=False, as_index=False)
        .agg(
            subscribers=("subscriber_id", "count"),
            churned_subscribers=(TARGET, "sum"),
            average_arpu=("arpu_last_month_inr", "mean"),
        )
    )
    result["churn_rate"] = result["churned_subscribers"] / result["subscribers"] * 100
    result["segment_type"] = segment_type
    return result.rename(columns={column: "segment"})


subscriber_segments = pd.concat(
    [
        analyse_segment("arpu_segment", "ARPU"),
        analyse_segment("tenure_segment", "Tenure"),
        analyse_segment("plan_type", "Plan type"),
    ],
    ignore_index=True,
)

print("\n" + "=" * 70)
print("SUBSCRIBER SEGMENTATION")
print("=" * 70)
print(subscriber_segments.round(2).to_string(index=False))

subscriber_segments.to_csv(
    os.path.join(OUTPUT_DIR, "subscriber_segmentation.csv"), index=False,
)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(data=subscriber_segments[subscriber_segments["segment_type"] == "ARPU"],
            x="segment", y="churn_rate", color="#D32F2F", ax=axes[0])
sns.barplot(data=subscriber_segments[subscriber_segments["segment_type"] == "Tenure"],
            x="segment", y="churn_rate", color="#7B1FA2", ax=axes[1])

axes[0].set_title("Churn Rate by ARPU Segment")
axes[0].set_xlabel("ARPU Segment")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].tick_params(axis="x", rotation=20)
axes[1].set_title("Churn Rate by Tenure Segment")
axes[1].set_xlabel("Tenure Segment")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "subscriber_segment_churn.png"), dpi=300)
plt.close()


# ============================================================
# 16. METRIC VERIFICATION
# ============================================================
total_subscribers = len(df)
total_churners = int(df[TARGET].sum())
calculated_churn_rate = total_churners / total_subscribers * 100

verification = pd.DataFrame({
    "metric": [
        "Total subscribers", "Unique subscriber IDs", "Duplicate subscriber IDs",
        "Total 30-day churners", "Calculated churn rate (%)",
        "Circle subscriber total", "Circle churner total",
        "Recharge-group subscriber total", "Recharge-group churner total",
    ],
    "value": [
        total_subscribers, df["subscriber_id"].nunique(),
        df["subscriber_id"].duplicated().sum(), total_churners,
        round(calculated_churn_rate, 4), int(circle_analysis["subscribers"].sum()),
        int(circle_analysis["churned_subscribers"].sum()),
        int(recharge_analysis["subscribers"].sum()),
        int(recharge_analysis["churned_subscribers"].sum()),
    ],
})

assert df["subscriber_id"].nunique() == total_subscribers
assert circle_analysis["subscribers"].sum() == total_subscribers
assert circle_analysis["churned_subscribers"].sum() == total_churners
assert recharge_analysis["subscribers"].sum() == total_subscribers
assert recharge_analysis["churned_subscribers"].sum() == total_churners

verification.to_csv(os.path.join(OUTPUT_DIR, "metric_verification.csv"), index=False)

print("\n" + "=" * 70)
print("METRIC VERIFICATION")
print("=" * 70)
print(verification.to_string(index=False))
print("\nAll verification checks passed.")


# ============================================================
# 17. EDA SUMMARY
# ============================================================
summary = pd.DataFrame({
    "metric": [
        "Total subscribers",
        "Duplicate rows",
        "30-day churned subscribers",
        "30-day churn rate",
        "Highest-churn circle",
        "Highest circle churn rate",
        "Most common churn reason",
        "Analysis start month",
        "Analysis end month",
    ],
    "value": [
        len(df),
        df.duplicated().sum(),
        int(df[TARGET].sum()),
        round(churn_rate, 2),
        circle_analysis.iloc[0]["circle"],
        round(
            circle_analysis.iloc[0]["churn_rate"],
            2,
        ),
        churn_reasons.iloc[0]["churn_reason"],
        monthly_trend["month_end"].min().date(),
        monthly_trend["month_end"].max().date(),
    ],
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "eda_summary.csv",
    ),
    index=False,
)


# ============================================================
# 18. FINAL RESULTS
# ============================================================
print("\n" + "=" * 70)
print("DAY 1 EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "Highest-churn circle:",
    circle_analysis.iloc[0]["circle"],
)

print(
    "Highest circle churn rate:",
    f"{circle_analysis.iloc[0]['churn_rate']:.2f}%",
)

print(
    "Most common churn reason:",
    churn_reasons.iloc[0]["churn_reason"],
)

print(f"All outputs saved in: {OUTPUT_DIR}")
