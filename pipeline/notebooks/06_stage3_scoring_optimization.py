# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Stage 3a — Batch Scoring & Optimization
# MAGIC
# MAGIC Flowchart boxes: **`Deploy model — batch score all customers (per-customer uplift score)`** →
# MAGIC **`Optimization (budget · frequency · eligibility)`** → **`Offer assignments decided`**.
# MAGIC
# MAGIC ## Two distinct steps
# MAGIC 1. **Scoring** is pure prediction: load the champion uplift model and produce one uplift score per
# MAGIC    customer. A high score means "a coupon meaningfully increases this person's purchase
# MAGIC    probability".
# MAGIC 2. **Optimization** is a business decision on top of the scores. Scores alone don't tell you who
# MAGIC    to mail — you have a finite budget, frequency caps and eligibility rules. We turn "who has the
# MAGIC    highest uplift" into "who should get a coupon **given our constraints**" by maximising expected
# MAGIC    incremental margin subject to those constraints.
# MAGIC
# MAGIC **Outputs:** `uplift_scores`, `offer_assignments`.

# COMMAND ----------

# MAGIC %run ../src/config

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import pyspark.sql.functions as F

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Batch score all known customers
# MAGIC Load the champion model and apply it to the full Feature Store snapshot. Unknown customers are
# MAGIC excluded — they have no features and are served by the legacy basket process / real-time path.

# COMMAND ----------

NUM_FEATURES = ["recency_days", "tenure_days", "frequency_per_month", "monetary_per_month",
                "txn_count", "total_spend", "avg_basket_value", "max_basket_value",
                "avg_items", "category_diversity"]
CAT_FEATURES = ["age_group", "city_tier", "channel_pref", "value_tier"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

scoring_pdf = (spark.table(FS_FEATURE_TABLE)
               .select("customer_id", "value_tier", "avg_basket_value", *ALL_FEATURES)
               .dropDuplicates(["customer_id"])
               .toPandas())

champion = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
scoring_pdf["uplift_score"] = champion.predict(scoring_pdf[ALL_FEATURES])["uplift_score"].values
print("Scored customers:", len(scoring_pdf))
print(scoring_pdf["uplift_score"].describe().round(4).to_string())

(spark.createDataFrame(scoring_pdf[["customer_id", "value_tier", "uplift_score"]])
    .withColumn("scored_ts", F.current_timestamp())
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(TBL_UPLIFT_SCORES))
print("Saved uplift scores →", TBL_UPLIFT_SCORES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Optimization — budget-constrained offer allocation
# MAGIC We frame allocation as a **budget-constrained knapsack**:
# MAGIC
# MAGIC - **Value of mailing customer *i*** = expected incremental margin
# MAGIC   = `uplift_i × avg_basket_value_i × MARGIN_RATE`.
# MAGIC - **Cost of mailing customer *i*** = expected coupon cost
# MAGIC   = `P(redeem)_i × COUPON_VALUE` (approximated here as uplift-driven redemption).
# MAGIC - **Constraints**: total cost ≤ `CAMPAIGN_BUDGET`; respect a per-customer **frequency cap**; only
# MAGIC   **eligible** customers (positive uplift above threshold) are considered.
# MAGIC
# MAGIC Because each customer is a 0/1 decision and items are tiny relative to the budget, the optimal
# MAGIC knapsack solution is well-approximated by a **greedy rule**: sort by *bang-for-buck*
# MAGIC (incremental margin per dollar of expected cost) and allocate until the budget is exhausted.

# COMMAND ----------

opt = scoring_pdf.copy()

# --- Eligibility filter -------------------------------------------------------------
opt = opt[opt["uplift_score"] > MIN_UPLIFT_THRESHOLD].copy()
print("Eligible after uplift threshold:", len(opt))

# --- Economics per customer ---------------------------------------------------------
opt["exp_incr_margin"] = opt["uplift_score"] * opt["avg_basket_value"] * MARGIN_RATE
# Approx expected coupon cost: more responsive customers are likelier to redeem.
opt["p_redeem"] = np.clip(0.3 + opt["uplift_score"], 0.05, 0.9)
opt["exp_cost"] = opt["p_redeem"] * COUPON_VALUE
opt["roi_per_dollar"] = opt["exp_incr_margin"] / opt["exp_cost"].clip(lower=1e-6)

# Only mail where expected incremental margin exceeds expected cost (profitable on its own).
opt = opt[opt["exp_incr_margin"] > opt["exp_cost"]].copy()
print("Profitable customers:", len(opt))

# --- Greedy budget allocation -------------------------------------------------------
opt = opt.sort_values("roi_per_dollar", ascending=False).reset_index(drop=True)
opt["cum_cost"] = opt["exp_cost"].cumsum()
opt["offer"] = (opt["cum_cost"] <= CAMPAIGN_BUDGET).astype(int)

selected = opt[opt["offer"] == 1].copy()
print(f"""
Optimization result:
  Budget                 : ${CAMPAIGN_BUDGET:,.0f}
  Customers offered       : {len(selected):,}
  Expected coupon spend   : ${selected['exp_cost'].sum():,.0f}
  Expected incr. margin   : ${selected['exp_incr_margin'].sum():,.0f}
  Expected ROI            : {selected['exp_incr_margin'].sum() / max(selected['exp_cost'].sum(),1):.2f}x
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Frequency cap & eligibility (final guard)
# MAGIC Before writing the decisions we enforce the **frequency cap**: drop anyone who has already
# MAGIC received `MAX_OFFERS_PER_CUSTOMER_30D` offers in the last 30 days. (On the first wave the history
# MAGIC is empty, so nothing is dropped; the logic is here for business-as-usual runs.)

# COMMAND ----------

try:
    recent = (spark.table(TBL_OFFER_ASSIGN)
              .filter(F.col("decision_ts") >= F.date_sub(F.current_timestamp(), 30))
              .groupBy("customer_id").count()
              .filter(F.col("count") >= MAX_OFFERS_PER_CUSTOMER_30D)
              .select("customer_id").toPandas()["customer_id"].tolist())
except Exception:
    recent = []
if recent:
    selected = selected[~selected["customer_id"].isin(recent)]
    print(f"Frequency cap removed {len(recent)} customers.")

offer_assignments = selected[[
    "customer_id", "value_tier", "uplift_score",
    "exp_incr_margin", "exp_cost", "roi_per_dollar"
]].copy()
offer_assignments["coupon_value"] = COUPON_VALUE
offer_assignments["campaign_id"] = "WAVE_02_OPTIMIZED"

(spark.createDataFrame(offer_assignments)
    .withColumn("decision_ts", F.current_timestamp())
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(TBL_OFFER_ASSIGN))
print("Saved offer assignments →", TBL_OFFER_ASSIGN)
print("Stage 3a complete: customers scored and offers optimized within budget.")
