# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Synthetic Data Generation
# MAGIC Creates realistic retail transaction + customer data so the whole pipeline is runnable
# MAGIC end-to-end in any Databricks workspace. In production you would replace this notebook
# MAGIC with reads from your real POS / CRM source tables.
# MAGIC
# MAGIC **Outputs:** `transactions`, `customers` Delta tables.

# COMMAND ----------

# MAGIC %run ../src/config

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ensure_schema(spark)
rng = np.random.default_rng(RANDOM_STATE)

N_CUSTOMERS = 20_000
REF = datetime.fromisoformat(REFERENCE_DATE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer master
# MAGIC A mix of long-tenured loyal shoppers, occasional buyers and brand-new sign-ups, so that
# MAGIC Stage 1 can split them into "full feature", "sparse" and "unknown" groups.

# COMMAND ----------

segments_latent = rng.choice(
    ["loyal", "regular", "occasional", "new"],
    size=N_CUSTOMERS,
    p=[0.20, 0.35, 0.30, 0.15],
)

signup_days_ago = np.where(
    segments_latent == "new",
    rng.integers(1, 60, N_CUSTOMERS),
    rng.integers(60, 1095, N_CUSTOMERS),
)

customers = pd.DataFrame({
    "customer_id": [f"CUST{i:07d}" for i in range(1, N_CUSTOMERS + 1)],
    "signup_date": [REF - timedelta(days=int(d)) for d in signup_days_ago],
    "age_group": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"],
                            N_CUSTOMERS, p=[0.15, 0.30, 0.25, 0.18, 0.12]),
    "city_tier": rng.choice(["Tier1", "Tier2", "Tier3"], N_CUSTOMERS, p=[0.45, 0.35, 0.20]),
    "channel_pref": rng.choice(["app", "web", "store"], N_CUSTOMERS, p=[0.4, 0.35, 0.25]),
    "_latent": segments_latent,
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transactions
# MAGIC Order frequency, basket size and an unobserved **coupon-responsiveness** trait vary by latent
# MAGIC segment. The responsiveness trait is what the uplift model will eventually try to recover.

# COMMAND ----------

freq_map  = {"loyal": 40, "regular": 18, "occasional": 6, "new": 2}
basket_mu = {"loyal": 65, "regular": 45, "occasional": 38, "new": 30}

# Latent individual responsiveness to coupons (used later to simulate campaign outcomes)
resp_base = {"loyal": 0.04, "regular": 0.12, "occasional": 0.20, "new": 0.15}
customers["_responsiveness"] = [
    np.clip(rng.normal(resp_base[s], 0.05), -0.05, 0.45) for s in customers["_latent"]
]

rows = []
for cust in customers.itertuples(index=False):
    n_tx = max(0, int(rng.poisson(freq_map[cust._latent])))
    for _ in range(n_tx):
        days_ago = rng.integers(1, max(2, (REF - cust.signup_date).days))
        amount = max(5.0, rng.normal(basket_mu[cust._latent], 12))
        rows.append((
            cust.customer_id,
            (REF - timedelta(days=int(days_ago))),
            round(float(amount), 2),
            int(rng.integers(1, 8)),                      # n_items
            rng.choice(["grocery", "household", "beauty", "snacks", "beverages"]),
        ))

transactions = pd.DataFrame(
    rows, columns=["customer_id", "txn_date", "amount", "n_items", "category"]
)
print(f"customers={len(customers):,}  transactions={len(transactions):,}")

# COMMAND ----------

# Persist to Delta. Keep the latent columns on customers for outcome simulation in notebook 04,
# but they are NOT used as model features (that would be leakage).
(spark.createDataFrame(customers)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(TBL_CUSTOMERS))

(spark.createDataFrame(transactions)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(TBL_TRANSACTIONS))

print("Saved:", TBL_CUSTOMERS, "and", TBL_TRANSACTIONS)
