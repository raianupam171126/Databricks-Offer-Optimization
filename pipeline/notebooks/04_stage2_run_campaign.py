# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Stage 2b — Run Campaign & Collect Outcomes
# MAGIC
# MAGIC Flowchart box: **`Run campaign — collect data (redemption + purchase, both groups)`**.
# MAGIC
# MAGIC In production this notebook reads back what actually happened after the coupons went out:
# MAGIC who redeemed, who purchased, and how much they spent — for **both** the treatment and the
# MAGIC holdout group (the holdout is observed too; they simply never got a coupon).
# MAGIC
# MAGIC Here, because there is no live campaign, we **simulate** the outcomes using each customer's
# MAGIC latent responsiveness from notebook 01. The simulation encodes the ground truth the uplift model
# MAGIC must recover: a coupon lifts purchase probability by an amount that varies per customer.
# MAGIC
# MAGIC **Output:** `campaign_outcomes` (purchase flag + spend per customer, per group).

# COMMAND ----------

# MAGIC %run ../src/config

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.sql.functions as F

CAMPAIGN_ID = "WAVE_01_LEARNING"
rng = np.random.default_rng(RANDOM_STATE + 7)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate the response (stand-in for real redemption/purchase logs)
# MAGIC **Ground-truth model used only for simulation:**
# MAGIC - Every customer has a baseline purchase probability in the campaign window.
# MAGIC - Treatment customers get an **uplift** equal to their latent responsiveness, plus a chance to
# MAGIC   redeem the coupon (redemption only possible in the treatment group).
# MAGIC - Holdout customers get **no** uplift — they are the counterfactual.
# MAGIC
# MAGIC The model never sees `_responsiveness`; it must infer it from features.

# COMMAND ----------

assign = spark.table(TBL_CAMPAIGN_ASSIGN).filter(F.col("campaign_id") == CAMPAIGN_ID)
cust = spark.table(TBL_CUSTOMERS).select("customer_id", "_responsiveness", "_latent")

pdf = (assign.join(cust, "customer_id", "left")
            .select("customer_id", "group", "value_tier", "coupon_value",
                    "_responsiveness", "_latent")
            .toPandas())
pdf["_responsiveness"] = pdf["_responsiveness"].fillna(0.10)

# Baseline purchase probability in the window (varies a little by tier)
base_p = {"High Value": 0.45, "Medium-High Value": 0.35,
          "Medium Value": 0.25, "Low Value": 0.15}
pdf["p_base"] = pdf["value_tier"].map(base_p).fillna(0.20)

is_treat = (pdf["group"] == "treatment").values
uplift = pdf["_responsiveness"].values            # true individual treatment effect
p_purchase = pdf["p_base"].values + np.where(is_treat, uplift, 0.0)
p_purchase = np.clip(p_purchase, 0.01, 0.95)

purchased = rng.random(len(pdf)) < p_purchase

# Redemption only in treatment, and only among purchasers (you redeem when you buy)
redeemed = is_treat & purchased & (rng.random(len(pdf)) < 0.7)

# Spend: purchasers spend a tier-typical basket; coupon reduces realised price if redeemed
basket_mu = {"High Value": 70, "Medium-High Value": 55,
             "Medium Value": 42, "Low Value": 32}
spend = np.where(
    purchased,
    np.clip(rng.normal([basket_mu.get(t, 40) for t in pdf["value_tier"]], 12), 5, None),
    0.0,
)
spend = spend - np.where(redeemed, pdf["coupon_value"].values, 0.0)
spend = np.clip(spend, 0, None)

outcomes = pd.DataFrame({
    "customer_id": pdf["customer_id"],
    "campaign_id": CAMPAIGN_ID,
    "group": pdf["group"],
    "value_tier": pdf["value_tier"],
    "coupon_value": pdf["coupon_value"],
    "purchased": purchased.astype(int),
    "redeemed": redeemed.astype(int),
    "spend": np.round(spend, 2),
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Naive lift check (the number the holdout exists to produce)
# MAGIC Average purchase rate in treatment minus holdout = **average treatment effect (ATE)**. This is
# MAGIC the population-level lift; the uplift model's job is to break this single number down to the
# MAGIC *individual* level so we can target only the customers who actually drive it.

# COMMAND ----------

summary = (outcomes.groupby("group")
           .agg(n=("customer_id", "count"),
                purchase_rate=("purchased", "mean"),
                avg_spend=("spend", "mean"),
                redemption_rate=("redeemed", "mean"))
           .reset_index())
print(summary.round(4).to_string(index=False))

ate = (summary.loc[summary.group == "treatment", "purchase_rate"].values[0]
       - summary.loc[summary.group == "holdout", "purchase_rate"].values[0])
print(f"\nNaive ATE (treatment - holdout purchase rate): {ate:+.4f}")

(spark.createDataFrame(outcomes)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .partitionBy("campaign_id")
    .saveAsTable(TBL_CAMPAIGN_OUTCOMES))
print("Saved outcomes →", TBL_CAMPAIGN_OUTCOMES)
