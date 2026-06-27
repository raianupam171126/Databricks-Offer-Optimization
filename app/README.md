# App — Streamlit on Databricks Apps

A five-screen Streamlit walkthrough of the [pipeline's](../pipeline/) outputs, deployed as a Databricks App.

## Files

- `app.py` — single-file Streamlit app with the five screens (Overview, Segments, Learning Wave, Uplift Model, Optimization)
- `app.yaml` — Databricks App manifest (command + env vars)
- `requirements.txt` — Python dependencies

## How it reads data

- **Auth:** OAuth machine-to-machine using the app's service principal (`DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET`, both injected automatically by Databricks Apps). Falls back to a personal access token (`DATABRICKS_TOKEN`) for local development.
- **Compute:** queries are routed to a SQL Warehouse via the `databricks-sql-connector`. No Spark inside the app.
- **MLflow:** the Uplift Model screen reads the registered champion model (`models:/main.coupon_uplift.coupon_uplift_model@champion`) when needed; the rest of the screens read derived tables that the pipeline already produced.

## Deployment

1. **Compute → Apps → Create app → Custom**, point at this folder.
2. **Resources** → attach your SQL Warehouse.
3. **Environment** → set `DATABRICKS_WAREHOUSE_HTTP_PATH = /sql/1.0/warehouses/<your-warehouse-id>`. (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` are injected automatically.) If your tables are not in `main.coupon_uplift`, also override `CATALOG` and `SCHEMA`.
4. **Grant the app's service principal access** to your tables (run in SQL Editor):
   ```sql
   GRANT USE CATALOG ON CATALOG <catalog> TO `<service-principal-uuid>`;
   GRANT USE SCHEMA ON SCHEMA <catalog>.<schema> TO `<service-principal-uuid>`;
   GRANT SELECT ON ALL TABLES IN SCHEMA <catalog>.<schema> TO `<service-principal-uuid>`;
   ```
   The service principal UUID is the app's `DATABRICKS_CLIENT_ID`, visible in the app's Environment tab.
5. **Deploy.** First start takes ~30s; subsequent loads are fast.

## Local development

```bash
export DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
export DATABRICKS_TOKEN=<personal access token>
export DATABRICKS_WAREHOUSE_HTTP_PATH=/sql/1.0/warehouses/<id>
export CATALOG=main
export SCHEMA=coupon_uplift
pip install -r requirements.txt
streamlit run app.py
```
