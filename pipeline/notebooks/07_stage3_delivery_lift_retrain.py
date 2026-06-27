# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Stage 3b — Delivery, Incremental Lift & Learn Loop
# MAGIC
# MAGIC Flowchart boxes: **`Batch Assignment (app/email/SMS)`** + **`Real-time Assignment
# MAGIC (anonymous, basket-driven)`** → **`Customer receives offer — shops`** →
# MAGIC **`Measure incremental lift vs holdout`** → **`Outcome fed back to the model — retrain`**.
# MAGIC
# MAGIC ## Two delivery paths
# MAGIC - **Batch assignment**: the optimized offer list from notebook 06 is pushed to app / email / SMS.
# MAGIC   We *keep a small fresh holdout* inside the optimized wave so lift stays measurable forever.
# MAGIC - **Real-time assignment**: for anonymous / unknown customers we cannot score uplift, so the
# MAGIC   legacy **basket-driven** rules decide the coupon at checkout. We log these too.
# MAGIC
# MAGIC ## Why keep measuring
# MAGIC The holdout never goes away. Every wave reserves a control slice so we can report **incremental
# MAGIC margin vs holdout** on a BI dashboard, then feed outcomes back to retrain the model — closing
# MAGIC the learn loop drawn as the "Retrain" arrow.
# MAGIC
# MAGIC **Outputs:** `incremental_lift` table (feeds the BI dashboard) and refreshed outcomes for retraining.

# COMMAND ----------

# MAGIC %run ../src/config

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.sql.functions as F

rng = np.random.default_rng(RANDOM_STATE + 99)
WAVE = "WAVE_02_OPTIMIZED"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Reserve a holdout inside the optimized wave
# MAGIC Even though notebook 06 chose these customers *because* they have high uplift, we still randomly
# MAGIC hold back a slice and give them no coupon. Without an ongoing control we could never separate the
# MAGIC coupon's effect from seasonality or trend.

# COMMAND ----------

offers = spark.table(TBL_OFFER_ASSIGN).filter(F.col("campaign_id") == WAVE)
delivered = (offers
    .withColumn("u",
        (F.abs(F.hash(F.concat_ws("::", F.lit(WAVE), F.col("customer_id")))) % 100000) / 100000.0)
    .withColumn("group",
        F.when(F.col("u") < F.lit(1.0 - HOLDOUT_FRACTION), F.lit("treatment")).otherwise(F.lit("holdout")))
    # Delivery channel: simple split standing in for app/email/SMS routing
    .withColumn("channel",
        F.when(F.col("group") == "holdout", F.lit("none"))
         .otherwise(F.element_at(F.array(F.lit("app"), F.lit("email"), F.lit("sms")),
                                 (F.abs(F.hash("customer_id")) % 3 + 1))))
    .drop("u"))

print("Delivery split:")
delivered.groupBy("group").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Simulate "customer receives offer — shops"
# MAGIC Stand-in for real outcome logs. Treatment customers realise their true uplift over baseline;
# MAGIC holdout customers realise baseline only. In production this whole cell is replaced by a read of
# MAGIC the post-campaign sales table.

# COMMAND ----------

pdf = delivered.join(
        spark.table(TBL_CUSTOMERS).select("customer_id", "_responsiveness", "_latent"),
        "customer_id", "left").toPandas()
pdf["_responsiveness"] = pdf["_responsiveness"].fillna(0.10)

base_p = {"High Value": 0.45, "Medium-High Value": 0.35, "Medium Value": 0.25, "Low Value": 0.15}
pdf["p_base"] = pdf["value_tier"].map(base_p).fillna(0.20)
is_t = (pdf["group"] == "treatment").values
p = np.clip(pdf["p_base"].values + np.where(is_t, pdf["_responsiveness"].values, 0.0), 0.01, 0.95)
pdf["purchased"] = (rng.random(len(pdf)) < p).astype(int)
pdf["redeemed"] = (is_t & (pdf["purchased"] == 1) & (rng.random(len(pdf)) < 0.7)).astype(int)
basket = {"High Value": 70, "Medium-High Value": 55, "Medium Value": 42, "Low Value": 32}
pdf["spend"] = np.where(pdf["purchased"] == 1,
                        np.clip(rng.normal([basket.get(t, 40) for t in pdf["value_tier"]], 12), 5, None),
                        0.0)
pdf["spend"] = np.clip(pdf["spend"] - pdf["redeemed"] * COUPON_VALUE, 0, None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Measure incremental lift vs holdout
# MAGIC The core KPI: **incremental margin = (treatment margin per head − holdout margin per head) ×
# MAGIC number treated − coupon cost.** This is the honest, holdout-validated number that goes on the BI
# MAGIC dashboard — not gross redemptions, which would over-credit the campaign.

# COMMAND ----------

g = (pdf.groupby("group")
     .agg(n=("customer_id", "count"),
          purchase_rate=("purchased", "mean"),
          avg_spend=("spend", "mean"),
          redemptions=("redeemed", "sum"))
     .reset_index())
print(g.round(4).to_string(index=False))

t = g[g.group == "treatment"].iloc[0]
c = g[g.group == "holdout"].iloc[0]

incr_purchase_rate = t.purchase_rate - c.purchase_rate
incr_spend_per_head = t.avg_spend - c.avg_spend
incr_margin_per_head = incr_spend_per_head * MARGIN_RATE
n_treated = int(t.n)
coupon_cost = float(t.redemptions) * COUPON_VALUE * COUPON_COST_FRACTION
total_incr_margin = incr_margin_per_head * n_treated - coupon_cost
roi = total_incr_margin / coupon_cost if coupon_cost else float("nan")

results = pd.DataFrame([{
    "campaign_id": WAVE,
    "n_treated": n_treated,
    "n_holdout": int(c.n),
    "incr_purchase_rate": round(incr_purchase_rate, 4),
    "incr_spend_per_head": round(incr_spend_per_head, 2),
    "incr_margin_per_head": round(incr_margin_per_head, 2),
    "coupon_cost": round(coupon_cost, 2),
    "total_incremental_margin": round(total_incr_margin, 2),
    "roi": round(roi, 2),
}])
print("\nINCREMENTAL LIFT (holdout-validated):")
print(results.T.to_string(header=False))

(spark.createDataFrame(results)
    .withColumn("measured_ts", F.current_timestamp())
    .write.mode("append").option("mergeSchema", "true")
    .saveAsTable(TBL_LIFT_RESULTS))
print("\nSaved lift results →", TBL_LIFT_RESULTS, "(BI dashboard source)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feed outcomes back for retraining
# MAGIC The treatment-vs-holdout outcomes from this wave are appended to `campaign_outcomes`. On the next
# MAGIC scheduled run, notebook 05 retrains the uplift model on the **accumulated** experience and the
# MAGIC champion/challenger guard decides whether to promote — this is the "Retrain" arrow looping back
# MAGIC to "Deploy model".

# COMMAND ----------

feedback = pdf[["customer_id", "group", "value_tier", "purchased", "redeemed", "spend"]].copy()
feedback["campaign_id"] = WAVE
feedback["coupon_value"] = np.where(feedback["group"] == "treatment", COUPON_VALUE, 0.0)

(spark.createDataFrame(feedback)
    .write.mode("append").option("mergeSchema", "true")
    .partitionBy("campaign_id")
    .saveAsTable(TBL_CAMPAIGN_OUTCOMES))
print("Appended outcomes for retraining →", TBL_CAMPAIGN_OUTCOMES)

print("""
Stage 3 complete. The loop is closed:
  score → optimize → deliver (batch + real-time) → measure lift vs holdout → retrain.
Schedule notebooks 05→06→07 as a recurring Databricks Job to run continuously.
""")
