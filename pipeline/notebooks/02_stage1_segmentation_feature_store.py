# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Stage 1 — Segmentation & Feature Store
# MAGIC
# MAGIC Implements the Stage-1 flowchart:
# MAGIC
# MAGIC `All customers at checkout` → routed into
# MAGIC 1. **Known customers** with full history → segmentation
# MAGIC 2. **Sparse customers** (little history) → segmentation with the features they have
# MAGIC 3. **Unknown customers** (no data) → fall back to the current basket-driven coupon process
# MAGIC
# MAGIC Known + sparse customers flow through: **Segmentation (clustering) → Profile each segment →
# MAGIC Feature engineering → Feature Store.**
# MAGIC
# MAGIC **Outputs:** `customer_features`, `customer_segments`, and a managed **Feature Store** table.

# COMMAND ----------

# MAGIC %run ../src/config

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.window import Window

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REF = F.to_timestamp(F.lit(REFERENCE_DATE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Feature engineering (RFM + behavioural)
# MAGIC We build one row per customer from the transaction history. These are the features used both
# MAGIC for clustering (Stage 1) and as model inputs (Stage 2/3), so we compute them once and publish
# MAGIC them to the Feature Store.

# COMMAND ----------

txns = spark.table(TBL_TRANSACTIONS)

cust_features = (
    txns.groupBy("customer_id")
    .agg(
        F.count("*").alias("txn_count"),
        F.sum("amount").alias("total_spend"),
        F.avg("amount").alias("avg_basket_value"),
        F.max("amount").alias("max_basket_value"),
        F.avg("n_items").alias("avg_items"),
        F.countDistinct("category").alias("category_diversity"),
        F.max("txn_date").alias("last_txn_date"),
        F.min("txn_date").alias("first_txn_date"),
    )
    # Recency / tenure / frequency
    .withColumn("recency_days", F.datediff(REF, F.col("last_txn_date")))
    .withColumn("tenure_days", F.datediff(REF, F.col("first_txn_date")))
    .withColumn("frequency_per_month",
                F.col("txn_count") / F.greatest(F.col("tenure_days") / 30.0, F.lit(1.0)))
    .withColumn("monetary_per_month",
                F.col("total_spend") / F.greatest(F.col("tenure_days") / 30.0, F.lit(1.0)))
)

# Join static customer attributes
cust_master = spark.table(TBL_CUSTOMERS).select(
    "customer_id", "age_group", "city_tier", "channel_pref"
)
cust_features = cust_features.join(cust_master, "customer_id", "inner")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Route customers — known vs sparse vs unknown
# MAGIC Customers absent from the transaction table are **unknown** (cold start) and are handled by the
# MAGIC legacy basket-driven process, not by the model. Among customers we *do* know, those with at
# MAGIC least `MIN_HISTORY_TXNS` transactions get the full feature treatment; the rest are flagged
# MAGIC `sparse` and still clustered, but their thinner history is noted for downstream confidence.

# COMMAND ----------

cust_features = cust_features.withColumn(
    "data_tier",
    F.when(F.col("txn_count") >= MIN_HISTORY_TXNS, F.lit("known_full"))
     .otherwise(F.lit("known_sparse"))
)

# Unknown customers = in master but no transactions at all
unknown = (cust_master.join(cust_features.select("customer_id"), "customer_id", "left_anti")
           .withColumn("data_tier", F.lit("unknown")))
print("Routing counts:")
cust_features.groupBy("data_tier").count().show()
print(f"unknown (no data): {unknown.count():,}")

# Persist the engineered feature table (offline source of truth)
feature_ts = F.current_timestamp()
(cust_features.withColumn(FS_TIMESTAMP_KEY, feature_ts)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(TBL_CUSTOMER_FEATS))
print("Saved engineered features →", TBL_CUSTOMER_FEATS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Segmentation (clustering)
# MAGIC We cluster the **known** customers (full + sparse) on scaled RFM-style features. The flowchart
# MAGIC lists K-means, DBSCAN and k-prototypes as candidates; we use **K-means** here and validate the
# MAGIC number of clusters with the silhouette score. (DBSCAN/k-prototypes can be swapped in — the
# MAGIC interface downstream only needs a `segment_id` per customer.)

# COMMAND ----------

CLUSTER_FEATURES = [
    "recency_days", "frequency_per_month", "monetary_per_month",
    "avg_basket_value", "category_diversity", "avg_items",
]

pdf = (spark.table(TBL_CUSTOMER_FEATS)
       .select("customer_id", *CLUSTER_FEATURES)
       .toPandas())

X = pdf[CLUSTER_FEATURES].fillna(0).copy()
# Log-transform skewed monetary features
for c in ["monetary_per_month", "avg_basket_value"]:
    X[c] = np.log1p(X[c])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Validate K by silhouette (small sample for speed)
sample_idx = np.random.RandomState(RANDOM_STATE).choice(
    len(X_scaled), size=min(5000, len(X_scaled)), replace=False)
sil = {}
for k in range(3, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled[sample_idx])
    sil[k] = silhouette_score(X_scaled[sample_idx], labels)
print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})

best_k = max(sil, key=sil.get) if sil else N_SEGMENTS
print("Chosen k:", best_k)

km = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
pdf["segment_id"] = km.fit_predict(X_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Profile each segment → High / Medium / Low value
# MAGIC We rank clusters by monetary value and map them to business-readable value tiers, exactly as
# MAGIC the flowchart's "Profile each segment" box requires.

# COMMAND ----------

seg_profile = (pdf.groupby("segment_id")
               .agg(size=("customer_id", "count"),
                    recency=("recency_days", "mean"),
                    freq=("frequency_per_month", "mean"),
                    monetary=("monetary_per_month", "mean"),
                    basket=("avg_basket_value", "mean"))
               .reset_index())

# Rank by monetary → value tier label
seg_profile = seg_profile.sort_values("monetary", ascending=False).reset_index(drop=True)
tier_labels = ["High Value", "Medium-High Value", "Medium Value", "Low Value",
               "Low Value", "Low Value"]
seg_profile["value_tier"] = [tier_labels[i] for i in range(len(seg_profile))]
display_cols = ["segment_id", "value_tier", "size", "recency", "freq", "monetary", "basket"]
print(seg_profile[display_cols].round(2).to_string(index=False))

tier_map = dict(zip(seg_profile["segment_id"], seg_profile["value_tier"]))
pdf["value_tier"] = pdf["segment_id"].map(tier_map)

# Persist segment assignment
seg_sdf = spark.createDataFrame(
    pdf[["customer_id", "segment_id", "value_tier"]]
)
(seg_sdf.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(TBL_SEGMENTS))
print("Saved segment assignments →", TBL_SEGMENTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Publish to the Feature Store
# MAGIC The engineered features + segment become a **Feature Store** table keyed by `customer_id` with a
# MAGIC timestamp key for point-in-time correctness. Stage 2 training and Stage 3 scoring both read
# MAGIC features from here, guaranteeing train/serve consistency.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

fs_source = (spark.table(TBL_CUSTOMER_FEATS)
             .join(spark.table(TBL_SEGMENTS), "customer_id", "left")
             .select(
                 "customer_id", FS_TIMESTAMP_KEY,
                 "recency_days", "tenure_days", "frequency_per_month", "monetary_per_month",
                 "txn_count", "total_spend", "avg_basket_value", "max_basket_value",
                 "avg_items", "category_diversity",
                 "age_group", "city_tier", "channel_pref",
                 "segment_id", "value_tier",
             ))

# Create the FS table once; thereafter write/merge into it.
try:
    fe.create_table(
        name=FS_FEATURE_TABLE,
        primary_keys=[FS_PRIMARY_KEY],
        timestamp_keys=[FS_TIMESTAMP_KEY],
        df=fs_source,
        description="Customer RFM + behavioural features and segment for coupon uplift modelling.",
    )
    print("Created Feature Store table:", FS_FEATURE_TABLE)
except Exception as e:
    # Table already exists → merge the latest snapshot
    print("create_table skipped (", type(e).__name__, ") — writing instead.")
    fe.write_table(name=FS_FEATURE_TABLE, df=fs_source, mode="merge")
    print("Merged snapshot into:", FS_FEATURE_TABLE)

# COMMAND ----------

print("Stage 1 complete: segments profiled and features published to the Feature Store.")
