# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Stage 2c — Train Uplift Model & Register in MLflow
# MAGIC
# MAGIC Flowchart boxes: **`Train uplift model (on treatment-vs-holdout group)`** →
# MAGIC **`Execute learning wave → benchmark against holdout → register best model in MLflow`**.
# MAGIC
# MAGIC ## What "uplift" means
# MAGIC A normal churn/propensity model predicts **P(purchase)**. That is the wrong target here: some
# MAGIC customers would have bought *anyway* (giving them a coupon just burns margin), and a few might
# MAGIC even be put off. What we actually want is the **incremental** effect of the coupon for each
# MAGIC customer — the *uplift*, or **Individual Treatment Effect (ITE)**:
# MAGIC
# MAGIC > uplift(x) = P(purchase | x, treated) − P(purchase | x, not treated)
# MAGIC
# MAGIC Because notebook 03 randomized treatment vs holdout, both conditional probabilities are
# MAGIC estimable from data, and their difference is causal.
# MAGIC
# MAGIC ## Approach: the T-learner (two-model uplift)
# MAGIC We fit **two** classifiers and subtract them:
# MAGIC 1. `model_T` — P(purchase) trained on the **treatment** group.
# MAGIC 2. `model_C` — P(purchase) trained on the **holdout (control)** group.
# MAGIC 3. uplift(x) = `model_T.predict_proba(x)` − `model_C.predict_proba(x)`.
# MAGIC
# MAGIC We wrap both into a single MLflow `pyfunc` model so Stage 3 can score uplift with one call, then
# MAGIC **benchmark against the holdout** with the Qini/uplift-curve and register the winner in Unity
# MAGIC Catalog.

# COMMAND ----------

# MAGIC %run ../src/config

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

import pyspark.sql.functions as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

mlflow.set_registry_uri("databricks-uc")          # register models in Unity Catalog
mlflow.set_experiment(MLFLOW_EXPERIMENT)
CAMPAIGN_ID = "WAVE_01_LEARNING"

# NumPy 2.x renamed trapz -> trapezoid; support both.
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build the training frame from the Feature Store
# MAGIC We read features **from the Feature Store** (train/serve consistency) and join the campaign
# MAGIC outcome (`purchased`) and the `group` label. Each customer contributes one row; the `group`
# MAGIC column decides which sub-model it trains.

# COMMAND ----------

NUM_FEATURES = ["recency_days", "tenure_days", "frequency_per_month", "monetary_per_month",
                "txn_count", "total_spend", "avg_basket_value", "max_basket_value",
                "avg_items", "category_diversity"]
CAT_FEATURES = ["age_group", "city_tier", "channel_pref", "value_tier"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
LABEL = "purchased"

features = spark.table(FS_FEATURE_TABLE).select("customer_id", *ALL_FEATURES)
outcomes = (spark.table(TBL_CAMPAIGN_OUTCOMES)
            .filter(F.col("campaign_id") == CAMPAIGN_ID)
            .select("customer_id", "group", LABEL))

train_sdf = outcomes.join(features, "customer_id", "inner")
pdf = train_sdf.toPandas()
print("Training rows:", len(pdf))
print(pdf["group"].value_counts().to_dict())

# Hold back 20% for honest uplift evaluation (stratified by group)
from sklearn.model_selection import train_test_split
tr, te = train_test_split(pdf, test_size=0.2, random_state=RANDOM_STATE, stratify=pdf["group"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define a reusable classifier pipeline
# MAGIC One preprocessing + gradient-boosting pipeline definition, instantiated twice (treatment and
# MAGIC control). Keeping them identical means the subtraction is apples-to-apples.

# COMMAND ----------

def make_pipeline():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])

tr_treat = tr[tr["group"] == "treatment"]
tr_ctrl  = tr[tr["group"] == "holdout"]

model_T = make_pipeline().fit(tr_treat[ALL_FEATURES], tr_treat[LABEL])
model_C = make_pipeline().fit(tr_ctrl[ALL_FEATURES],  tr_ctrl[LABEL])
print("Fitted treatment model on", len(tr_treat), "rows; control model on", len(tr_ctrl), "rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Wrap the two models as one MLflow pyfunc
# MAGIC The custom `UpliftModel` returns the **uplift score** directly, so Stage 3 never has to know
# MAGIC there are two underlying estimators.

# COMMAND ----------

class UpliftModel(mlflow.pyfunc.PythonModel):
    """T-learner uplift: P(purchase|treated) - P(purchase|control)."""
    def __init__(self, model_t, model_c, feature_cols):
        self.model_t = model_t
        self.model_c = model_c
        self.feature_cols = feature_cols

    def predict(self, context, model_input):
        X = model_input[self.feature_cols]
        p_t = self.model_t.predict_proba(X)[:, 1]
        p_c = self.model_c.predict_proba(X)[:, 1]
        return pd.DataFrame({"uplift_score": p_t - p_c})

uplift_model = UpliftModel(model_T, model_C, ALL_FEATURES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Benchmark against the holdout — Qini / uplift curve
# MAGIC The right metric for uplift is **not** accuracy or AUC. We use the **Qini coefficient**: rank the
# MAGIC evaluation customers by predicted uplift, then walk down the list checking whether the *treated*
# MAGIC purchasers really do outrun the *control* purchasers in that order. A good model concentrates the
# MAGIC real incremental buyers at the top of the ranking.

# COMMAND ----------

def qini_curve(df, score_col, treat_col, outcome_col):
    """Return (x fraction targeted, y cumulative incremental purchases) and Qini coefficient."""
    d = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    d["is_t"] = (d[treat_col] == "treatment").astype(int)
    d["is_c"] = (d[treat_col] == "holdout").astype(int)
    d["t_cum"] = (d["is_t"] * d[outcome_col]).cumsum()
    d["c_cum"] = (d["is_c"] * d[outcome_col]).cumsum()
    d["nt_cum"] = d["is_t"].cumsum().clip(lower=1)
    d["nc_cum"] = d["is_c"].cumsum().clip(lower=1)
    # incremental gain = treated responders - control responders scaled to treated population
    d["qini"] = d["t_cum"] - d["c_cum"] * (d["nt_cum"] / d["nc_cum"])
    x = np.arange(1, len(d) + 1) / len(d)
    y = d["qini"].values
    # Qini coefficient = area between model curve and the random diagonal
    rand = np.linspace(0, y[-1], len(y))
    qini_coef = float(_trapz(y - rand, x))
    return x, y, qini_coef

te_scored = te.copy()
te_scored["uplift_score"] = uplift_model.predict(None, te)["uplift_score"].values
x, y, qini_coef = qini_curve(te_scored, "uplift_score", "group", LABEL)
print(f"Qini coefficient (holdout benchmark): {qini_coef:.2f}")

# Also report decile lift: top-decile uplift should beat bottom-decile
te_scored["uplift_decile"] = pd.qcut(te_scored["uplift_score"].rank(method="first"),
                                     10, labels=False)
top = te_scored[te_scored.uplift_decile == 9]
bot = te_scored[te_scored.uplift_decile == 0]
def grp_rate(d, g): 
    s = d[d.group == g]
    return s[LABEL].mean() if len(s) else 0.0
top_lift = grp_rate(top, "treatment") - grp_rate(top, "holdout")
bot_lift = grp_rate(bot, "treatment") - grp_rate(bot, "holdout")
print(f"Top-decile observed uplift:    {top_lift:+.4f}")
print(f"Bottom-decile observed uplift: {bot_lift:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log & register in MLflow
# MAGIC We log params, the Qini metric and the curve, then register the pyfunc model in Unity Catalog.
# MAGIC "Register the best model" = compare this run's Qini against the current registered champion and
# MAGIC only promote if it wins (guard shown below).

# COMMAND ----------

signature = infer_signature(te[ALL_FEATURES],
                            pd.DataFrame({"uplift_score": [0.0] * len(te)}))

with mlflow.start_run(run_name=f"uplift_{CAMPAIGN_ID}") as run:
    mlflow.log_params({
        "approach": "T-learner",
        "base_estimator": "GradientBoostingClassifier",
        "campaign_id": CAMPAIGN_ID,
        "n_train": len(tr), "n_eval": len(te),
        "treatment_fraction": TREATMENT_FRACTION,
    })
    mlflow.log_metrics({
        "qini_coef": qini_coef,
        "top_decile_uplift": top_lift,
        "bottom_decile_uplift": bot_lift,
    })

    # Save the Qini curve as an artifact
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, label=f"Model (Qini={qini_coef:.1f})")
    ax.plot(x, np.linspace(0, y[-1], len(y)), "--", color="grey", label="Random")
    ax.set_xlabel("Fraction of customers targeted (by predicted uplift)")
    ax.set_ylabel("Cumulative incremental purchases")
    ax.set_title("Qini curve — holdout benchmark")
    ax.legend()
    mlflow.log_figure(fig, "qini_curve.png")
    plt.close(fig)

    model_info = mlflow.pyfunc.log_model(
        artifact_path="uplift_model",
        python_model=uplift_model,
        signature=signature,
        input_example=te[ALL_FEATURES].head(3),
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    run_id = run.info.run_id
    print("Logged + registered:", model_info.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Champion/challenger promotion guard
# MAGIC Promote the new version to the **@champion** alias only if its Qini beats the incumbent. This is
# MAGIC the gate behind "register the best model".

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

# newest version we just created
versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
new_version = max(int(v.version) for v in versions)

def qini_of(version):
    v = client.get_model_version(REGISTERED_MODEL_NAME, str(version))
    r = client.get_run(v.run_id)
    return r.data.metrics.get("qini_coef", -1e9)

promote = True
try:
    champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    promote = qini_of(new_version) > qini_of(champ.version)
    print(f"Incumbent champion v{champ.version} Qini={qini_of(champ.version):.2f} | "
          f"new v{new_version} Qini={qini_of(new_version):.2f}")
except Exception:
    print("No champion yet — promoting first model.")

if promote:
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", new_version)
    print(f"Promoted v{new_version} → @champion")
else:
    print(f"Kept existing champion; v{new_version} stays as challenger.")

print("Stage 2 complete: uplift model trained, benchmarked and registered.")
