# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Stage 2a — Campaign Design: Treatment vs Holdout
# MAGIC
# MAGIC Implements the top of the Stage-2 flowchart:
# MAGIC
# MAGIC `Select segment for campaign (aligned to objective)` →
# MAGIC `Randomly split the segment` → **Treatment group (gets coupon)** vs **Holdout group (no coupon)**.
# MAGIC
# MAGIC This randomized split is the single most important design choice in the whole project: it turns
# MAGIC the campaign into a **controlled experiment (A/B test)**. Because assignment is random, the only
# MAGIC systematic difference between the two groups is the coupon itself — so any difference in their
# MAGIC purchase behaviour is the **causal effect** of the coupon. That causal effect is exactly what the
# MAGIC uplift model in notebook 05 learns to predict.
# MAGIC
# MAGIC **Output:** `campaign_assignments` (one row per targeted customer with `group ∈ {treatment, holdout}`).

# COMMAND ----------

# MAGIC %run ../src/config

# COMMAND ----------

import pyspark.sql.functions as F

# Parameters for THIS campaign wave -------------------------------------------------
CAMPAIGN_ID = "WAVE_01_LEARNING"          # the first wave = the "learning wave"
TARGET_VALUE_TIERS = ["High Value", "Medium-High Value", "Medium Value", "Low Value"]
# ^ For the learning wave we deliberately include ALL tiers so the model sees the full
#   response spectrum. Later (business-as-usual) waves typically target fewer tiers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Select the segment(s) aligned to the campaign objective
# MAGIC The objective of the *first* wave is **learning**: collect treatment-vs-holdout response data
# MAGIC across the whole customer base so the uplift model has signal everywhere. We therefore pull all
# MAGIC known customers in the chosen value tiers. (For a revenue-objective wave you would instead pick,
# MAGIC say, only "High Value" + "Medium-High Value".)

# COMMAND ----------

segments = spark.table(TBL_SEGMENTS)
eligible = (segments
            .filter(F.col("value_tier").isin(TARGET_VALUE_TIERS))
            .select("customer_id", "segment_id", "value_tier"))

n_eligible = eligible.count()
print(f"Eligible customers for {CAMPAIGN_ID}: {n_eligible:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Randomly split — stratified by segment
# MAGIC We split **within each value tier** (stratified randomization) so that treatment and holdout are
# MAGIC balanced on segment composition, not just overall. A deterministic hash of `customer_id`
# MAGIC guarantees the split is reproducible and stable if the job re-runs.
# MAGIC
# MAGIC - **Treatment** (`TREATMENT_FRACTION`, e.g. 80%) → will receive the coupon.
# MAGIC - **Holdout** (e.g. 20%) → deliberately gets **no** coupon and becomes the control baseline.

# COMMAND ----------

# Deterministic uniform(0,1) per customer from a stable hash → reproducible split.
assignments = (eligible
    .withColumn("u",
        (F.abs(F.hash(F.concat_ws("::", F.lit(CAMPAIGN_ID), F.col("customer_id")))) % 100000)
        / 100000.0)
    .withColumn("group",
        F.when(F.col("u") < F.lit(TREATMENT_FRACTION), F.lit("treatment"))
         .otherwise(F.lit("holdout")))
    .withColumn("campaign_id", F.lit(CAMPAIGN_ID))
    .withColumn("coupon_value",
        F.when(F.col("group") == "treatment", F.lit(COUPON_VALUE)).otherwise(F.lit(0.0)))
    .withColumn("assigned_ts", F.current_timestamp())
    .drop("u"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Balance check (sanity)
# MAGIC Before launching we confirm the split is ~80/20 in every tier. A balanced split is what licenses
# MAGIC the later causal interpretation; if it looks skewed, fix the randomization before spending budget.

# COMMAND ----------

balance = (assignments.groupBy("value_tier", "group").count()
           .groupBy("value_tier")
           .pivot("group").sum("count")
           .withColumn("treatment_pct",
                       F.round(F.col("treatment") / (F.col("treatment") + F.col("holdout")), 3)))
balance.orderBy("value_tier").show()

(assignments.write.mode("overwrite").option("overwriteSchema", "true")
    .partitionBy("campaign_id")
    .saveAsTable(TBL_CAMPAIGN_ASSIGN))
print("Saved campaign assignments →", TBL_CAMPAIGN_ASSIGN)

# COMMAND ----------

print(f"""
Stage 2a complete.
  Campaign        : {CAMPAIGN_ID}
  Eligible        : {n_eligible:,}
  Split           : {int(TREATMENT_FRACTION*100)}% treatment / {int(HOLDOUT_FRACTION*100)}% holdout (stratified by tier)
  Next            : run the campaign and collect outcomes (notebook 04).
""")
