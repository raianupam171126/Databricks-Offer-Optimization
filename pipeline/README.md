# Pipeline — Databricks Notebooks

The data + modelling backbone of the project. Eight notebooks executed in order produce everything the [app](../app/) renders.

## Notebooks

| File | Stage | Role |
|------|-------|------|
| `src/config.py` | — | Shared config: tables, Feature Store, MLflow, business parameters |
| `notebooks/01_data_generation.py` | — | Synthetic POS + customer data (swap for real sources in production) |
| `notebooks/02_stage1_segmentation_feature_store.py` | 1 | Routing, clustering, profiling, Feature Store publish |
| `notebooks/03_stage2_campaign_design.py` | 2a | Select segment → randomized treatment/holdout split |
| `notebooks/04_stage2_run_campaign.py` | 2b | Run campaign, collect redemption + purchase outcomes |
| `notebooks/05_stage2_uplift_model.py` | 2c | **T-learner uplift model**, Qini benchmark, MLflow register |
| `notebooks/06_stage3_scoring_optimization.py` | 3a | Batch score + budget-constrained optimization |
| `notebooks/07_stage3_delivery_lift_retrain.py` | 3b | Delivery, incremental lift vs holdout, retrain feedback |
| `notebooks/08_orchestration.py` | — | Runs all notebooks in order |

## Running

1. Import this folder into Databricks (Repos or Workspace).
2. Edit `src/config.py` — set `CATALOG` and `SCHEMA` to a Unity Catalog schema you can write to.
3. Attach a Databricks Runtime ML cluster.
4. Run `notebooks/08_orchestration.py` end-to-end, or run 01→07 manually in order.

For production: wire the notebooks into a multi-task Databricks Job with task dependencies. A typical cadence:
- **One-off onboarding:** 01 → 02 → 03 → 04 → 05 (build features, run learning wave, train first model)
- **Recurring (weekly):** 02 → 05 → 06 → 07 (refresh features, retrain, score, deliver, measure)

See [`../docs/Implementation_Guide.docx`](../docs/Implementation_Guide.docx) for the full methodology rationale.
