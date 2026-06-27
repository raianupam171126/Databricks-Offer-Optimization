# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Project Configuration
# MAGIC Shared configuration for the **Coupon Personalization Uplift** project.
# MAGIC Import this at the top of every other notebook with:
# MAGIC ```python
# MAGIC %run ./00_config
# MAGIC ```
# MAGIC (or `%run ../src/config` depending on where you place it).

# COMMAND ----------

# ============================================================
# CATALOG / SCHEMA  (Unity Catalog).  Change these to your env.
# ============================================================
CATALOG = "main"
SCHEMA = "coupon_uplift"

# Fully-qualified table names ---------------------------------
TBL_TRANSACTIONS   = f"{CATALOG}.{SCHEMA}.transactions"          # raw POS transactions
TBL_CUSTOMERS      = f"{CATALOG}.{SCHEMA}.customers"             # customer master
TBL_CUSTOMER_FEATS = f"{CATALOG}.{SCHEMA}.customer_features"     # engineered features (offline FS source)
TBL_SEGMENTS       = f"{CATALOG}.{SCHEMA}.customer_segments"     # segment assignment + profile
TBL_CAMPAIGN_ASSIGN= f"{CATALOG}.{SCHEMA}.campaign_assignments"  # treatment/holdout split per campaign
TBL_CAMPAIGN_OUTCOMES = f"{CATALOG}.{SCHEMA}.campaign_outcomes"  # redemption + purchase results
TBL_UPLIFT_SCORES  = f"{CATALOG}.{SCHEMA}.uplift_scores"         # per-customer uplift score (batch)
TBL_OFFER_ASSIGN   = f"{CATALOG}.{SCHEMA}.offer_assignments"     # post-optimization offer decisions
TBL_LIFT_RESULTS   = f"{CATALOG}.{SCHEMA}.incremental_lift"      # measured lift vs holdout

# Feature Store --------------------------------------------------
FS_FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.customer_features_fs"    # the Feature Store managed table
FS_PRIMARY_KEY = "customer_id"
FS_TIMESTAMP_KEY = "feature_ts"                                 # for point-in-time lookups

# MLflow ---------------------------------------------------------
MLFLOW_EXPERIMENT = f"/Shared/{SCHEMA}_uplift"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.coupon_uplift_model"  # UC-registered model

# ============================================================
# BUSINESS / MODELLING PARAMETERS
# ============================================================
REFERENCE_DATE = "2024-12-31"        # "today" for recency / tenure calculations
RANDOM_STATE = 42

# Segmentation
N_SEGMENTS = 4                        # k for k-means (validated by silhouette)
MIN_HISTORY_TXNS = 3                 # >= this many txns => "known customer with full features"

# Campaign / experiment design
TREATMENT_FRACTION = 0.80            # 80% treatment, 20% holdout
HOLDOUT_FRACTION = 1.0 - TREATMENT_FRACTION
COUPON_VALUE = 5.0                   # $ face value of the coupon used in the learning wave
COUPON_COST_FRACTION = 1.0           # cost the business bears per redeemed coupon (here = face value)

# Optimization
MARGIN_RATE = 0.30                   # gross margin on incremental revenue
CAMPAIGN_BUDGET = 50_000.0           # $ total coupon budget for a wave
MAX_OFFERS_PER_CUSTOMER_30D = 2      # frequency cap
MIN_UPLIFT_THRESHOLD = 0.0          # only target customers with positive expected uplift

# COMMAND ----------

def ensure_schema(spark):
    """Create catalog.schema if they do not exist. Safe to call repeatedly."""
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    print(f"Using {CATALOG}.{SCHEMA}")

# COMMAND ----------

print("Config loaded.")
print(f"  Catalog/Schema : {CATALOG}.{SCHEMA}")
print(f"  Treatment/Holdout: {int(TREATMENT_FRACTION*100)}/{int(HOLDOUT_FRACTION*100)}")
print(f"  Registered model : {REGISTERED_MODEL_NAME}")
