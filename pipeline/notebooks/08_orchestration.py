# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Orchestration — Run the Full Pipeline
# MAGIC
# MAGIC Runs every stage in order using `dbutils.notebook.run`. Use this for a one-click end-to-end
# MAGIC execution, or wire the same sequence into a **Databricks Job** with one task per notebook and
# MAGIC task dependencies matching the arrows below.
# MAGIC
# MAGIC ```
# MAGIC 01 data ─▶ 02 Stage1 ─▶ 03 Stage2a ─▶ 04 Stage2b ─▶ 05 Stage2c ─▶ 06 Stage3a ─▶ 07 Stage3b
# MAGIC                                                          ▲                              │
# MAGIC                                                          └────────── retrain ───────────┘
# MAGIC ```
# MAGIC
# MAGIC ## Recommended Job schedule
# MAGIC - **One-off / on-board:** 01 → 02 → 03 → 04 → 05  (build features, run learning wave, train model)
# MAGIC - **Recurring (e.g. weekly):** 02 → 05 → 06 → 07  (refresh features, retrain, score, deliver, measure)
# MAGIC
# MAGIC The retrain arrow is realised simply by scheduling 05 again: it reads the *accumulated*
# MAGIC `campaign_outcomes` and the champion/challenger guard decides whether to promote.

# COMMAND ----------

# Relative paths assume all notebooks live in the same folder. Adjust if needed.
TIMEOUT = 3600

steps = [
    ("01_data_generation", {}),
    ("02_stage1_segmentation_feature_store", {}),
    ("03_stage2_campaign_design", {}),
    ("04_stage2_run_campaign", {}),
    ("05_stage2_uplift_model", {}),
    ("06_stage3_scoring_optimization", {}),
    ("07_stage3_delivery_lift_retrain", {}),
]

for name, params in steps:
    print(f"\n{'='*60}\n▶ Running {name}\n{'='*60}")
    result = dbutils.notebook.run(name, TIMEOUT, params)
    print(f"✓ {name} finished: {result}")

print("\nFull pipeline complete.")
