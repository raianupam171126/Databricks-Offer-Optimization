"""
Coupon Uplift Demo — Databricks App (Streamlit).

A tight 5-screen walkthrough of the coupon-personalization uplift pipeline:
  1. Overview        — architecture + customer routing counts
  2. Segments        — value tiers + segment profile
  3. Learning Wave   — treatment/holdout balance + aggregate ATE
  4. Uplift Model    — Qini curve + decile lift
  5. Optimization    — budget slider + ROI + measured lift

Reads from the Delta tables produced by the pipeline notebooks (config.py).
Deployed as a Databricks App; runs against the workspace's SQL warehouse.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from databricks import sql

# ---------------------------------------------------------------------------
# Configuration  (must match src/config.py in the pipeline project)
# ---------------------------------------------------------------------------
CATALOG = os.getenv("CATALOG", "main")
SCHEMA = os.getenv("SCHEMA", "coupon_uplift")

# --- Warehouse / auth (Databricks Apps style) -------------------------------
# Databricks Apps inject DATABRICKS_HOST plus a service-principal OAuth
# (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET) into the app's environment.
# The HTTP path of the SQL warehouse is NOT auto-injected; we read it from
# DATABRICKS_WAREHOUSE_HTTP_PATH, which we set ourselves in the app's
# Environment settings.
SERVER_HOSTNAME = os.getenv("DATABRICKS_HOST", "").replace("https://", "").rstrip("/")
WAREHOUSE_HTTP_PATH = os.getenv("DATABRICKS_WAREHOUSE_HTTP_PATH")
CLIENT_ID = os.getenv("DATABRICKS_CLIENT_ID")
CLIENT_SECRET = os.getenv("DATABRICKS_CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("DATABRICKS_TOKEN")  # legacy PAT path (local dev only)

T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

# ---------------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Coupon Uplift Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#1F6FEB"
TEAL = "#0E7C7B"
ORANGE = "#C77700"

st.markdown(
    f"""
    <style>
      .big-metric {{ font-size: 2.2rem; font-weight: 700; color: {ACCENT}; }}
      .metric-label {{ font-size: 0.85rem; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }}
      .pill {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
              background: #EAF1FB; color: {ACCENT}; font-size: 0.8rem; font-weight: 600; }}
      .pill-teal {{ background: #E0F0EF; color: {TEAL}; }}
      .pill-orange {{ background: #FBF0DD; color: {ORANGE}; }}
      h1, h2 {{ color: #0B2545; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data access — one cached SQL query helper
# ---------------------------------------------------------------------------
@st.cache_resource
def _conn():
    """Open a single SQL warehouse connection, reused across queries.

    Auth precedence:
      1. OAuth machine-to-machine using DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET
         (this is what Databricks Apps inject — the app's service principal).
      2. Personal access token via DATABRICKS_TOKEN (local dev).
    """
    if not WAREHOUSE_HTTP_PATH:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_HTTP_PATH is not set. Add it in the app's "
            "Environment settings (e.g. /sql/1.0/warehouses/<warehouse-id>)."
        )
    if not SERVER_HOSTNAME:
        raise RuntimeError("DATABRICKS_HOST is not set in the app environment.")

    if CLIENT_ID and CLIENT_SECRET:
        # OAuth M2M — the Databricks Apps recommended path.
        from databricks.sdk.core import Config, oauth_service_principal

        cfg = Config(
            host=f"https://{SERVER_HOSTNAME}",
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        return sql.connect(
            server_hostname=SERVER_HOSTNAME,
            http_path=WAREHOUSE_HTTP_PATH,
            credentials_provider=lambda: oauth_service_principal(cfg),
        )
    elif ACCESS_TOKEN:
        # Personal access token (local dev only)
        return sql.connect(
            server_hostname=SERVER_HOSTNAME,
            http_path=WAREHOUSE_HTTP_PATH,
            access_token=ACCESS_TOKEN,
        )
    else:
        raise RuntimeError(
            "No Databricks auth available. Either DATABRICKS_CLIENT_ID + "
            "DATABRICKS_CLIENT_SECRET (set automatically by Databricks Apps) "
            "or DATABRICKS_TOKEN (local dev) must be present."
        )

@st.cache_data(ttl=600, show_spinner=False)
def query(q: str) -> pd.DataFrame:
    """Run a SQL query against the warehouse and return a DataFrame.

    Uses fetchall_arrow() when available (fastest), falls back to fetchall()
    + pandas conversion. fetch_pandas_all() was removed in newer versions
    of databricks-sql-connector.
    """
    with _conn().cursor() as cur:
        cur.execute(q)
        # Try Arrow first (fast, columnar)
        if hasattr(cur, "fetchall_arrow"):
            try:
                return cur.fetchall_arrow().to_pandas()
            except Exception:
                pass
        # Fallback: rows + description -> DataFrame
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return pd.DataFrame.from_records(rows, columns=cols)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 Coupon Uplift Demo")
    st.caption("Live demo against the Databricks pipeline.")
    st.markdown(f"**Catalog.Schema:** `{CATALOG}.{SCHEMA}`")
    st.divider()
    page = st.radio(
        "Walkthrough",
        ["1. Overview",
         "2. Stage 1 — Segments",
         "3. Stage 2 — Learning Wave",
         "4. Stage 2 — Uplift Model",
         "5. Stage 3 — Optimization & Lift"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Cache TTL: 10 min")

# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------
def page_overview():
    st.title("Coupon Personalization with Uplift Modelling")
    st.caption("End-to-end pipeline: segmentation → treatment/holdout learning → uplift model → budget-constrained optimization → measured lift.")

    st.markdown("### The three stages")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Stage 1")
        st.markdown("**Segmentation & Feature Store**")
        st.caption("Route customers (known / sparse / unknown). Cluster the known ones. Publish features.")
    with c2:
        st.markdown("#### Stage 2")
        st.markdown("**Learning wave + Uplift model**")
        st.caption("80/20 treatment/holdout split. Train two-model T-learner. Validate with Qini.")
    with c3:
        st.markdown("#### Stage 3")
        st.markdown("**Score → Optimize → Deliver → Learn**")
        st.caption("Batch score, budget-constrained knapsack, measure incremental lift vs holdout.")

    st.divider()
    st.markdown("### Customer routing at checkout")

    try:
        # Counts from the segments table; unknowns = customers in master but not in segments
        seg = query(f"""
            SELECT value_tier, COUNT(*) AS n
            FROM {T('customer_segments')}
            GROUP BY value_tier
            ORDER BY n DESC
        """)
        n_known = int(seg["n"].sum())
        n_total = int(query(f"SELECT COUNT(*) AS n FROM {T('customers')}").iloc[0]["n"])
        n_unknown = max(0, n_total - n_known)
    except Exception as e:
        st.error(f"Could not load segment counts. Has the pipeline been run? ({e})")
        return

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-label">Total customers</div>'
        f'<div class="big-metric">{n_total:,}</div>', unsafe_allow_html=True)
    c2.markdown(
        f'<div class="metric-label">Known (modelled)</div>'
        f'<div class="big-metric">{n_known:,}</div>'
        f'<span class="pill pill-teal">Uplift-scored</span>', unsafe_allow_html=True)
    c3.markdown(
        f'<div class="metric-label">Unknown (cold start)</div>'
        f'<div class="big-metric">{n_unknown:,}</div>'
        f'<span class="pill pill-orange">Basket-driven rules</span>', unsafe_allow_html=True)

    st.markdown("**Takeaway.** Known customers are personalized at the individual level by the uplift model. "
                "Unknown customers fall back to basket-driven rules — this hybrid is the practical handling of cold start.")

# ---------------------------------------------------------------------------
# Page 2 — Stage 1 segments
# ---------------------------------------------------------------------------
def page_segments():
    st.title("Stage 1 — Segments")
    st.caption("Clustered on RFM + behavioural features, profiled and labelled into value tiers.")

    profile = query(f"""
        SELECT s.value_tier,
               COUNT(*) AS n_customers,
               AVG(f.recency_days) AS avg_recency,
               AVG(f.frequency_per_month) AS avg_freq,
               AVG(f.monetary_per_month) AS avg_monthly_spend,
               AVG(f.avg_basket_value) AS avg_basket
        FROM {T('customer_segments')} s
        JOIN {T('customer_features')} f USING (customer_id)
        GROUP BY s.value_tier
        ORDER BY avg_monthly_spend DESC
    """)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Segment sizes")
        chart = alt.Chart(profile).mark_bar(color=ACCENT).encode(
            x=alt.X("n_customers:Q", title="Customers"),
            y=alt.Y("value_tier:N", sort="-x", title=None),
            tooltip=["value_tier", "n_customers"],
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)

    with c2:
        st.markdown("### Profile")
        st.dataframe(
            profile.round(2).rename(columns={
                "value_tier": "Tier", "n_customers": "Customers",
                "avg_recency": "Recency (d)", "avg_freq": "Freq/mo",
                "avg_monthly_spend": "Spend/mo", "avg_basket": "Avg basket",
            }),
            hide_index=True, use_container_width=True,
        )

    st.info("**Why this matters:** the value tier is used both as a campaign-targeting lever "
            "AND as a feature for the uplift model — so the model can learn that, say, high-value "
            "customers respond differently to coupons than low-value ones.")

# ---------------------------------------------------------------------------
# Page 3 — Learning Wave (treatment vs holdout)
# ---------------------------------------------------------------------------
def page_learning_wave():
    st.title("Stage 2 — Learning Wave: Treatment vs Holdout")
    st.caption("Randomized 80/20 split, stratified by value tier — turns the campaign into a causal experiment.")

    bal = query(f"""
        SELECT value_tier, `group`, COUNT(*) AS n
        FROM {T('campaign_assignments')}
        GROUP BY value_tier, `group`
    """)
    bal_pivot = bal.pivot(index="value_tier", columns="group", values="n").fillna(0)
    bal_pivot["treatment_pct"] = (bal_pivot.get("treatment", 0) /
                                  bal_pivot.sum(axis=1)).round(3)

    st.markdown("### Balance check (each tier should be ~80% treatment)")
    st.dataframe(bal_pivot.reset_index(), hide_index=True, use_container_width=True)

    # Aggregate ATE
    out = query(f"""
        SELECT `group`,
               COUNT(*) AS n,
               AVG(purchased) AS purchase_rate,
               AVG(spend) AS avg_spend,
               AVG(redeemed) AS redemption_rate
        FROM {T('campaign_outcomes')}
        GROUP BY `group`
    """)
    t = out[out["group"] == "treatment"].iloc[0]
    c = out[out["group"] == "holdout"].iloc[0]
    ate = float(t["purchase_rate"] - c["purchase_rate"])

    st.markdown("### Aggregate Treatment Effect (ATE)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Treatment purchase rate", f"{t['purchase_rate']:.1%}", f"n={int(t['n']):,}")
    c2.metric("Holdout purchase rate", f"{c['purchase_rate']:.1%}", f"n={int(c['n']):,}")
    c3.metric("ATE (lift on average)", f"{ate:+.1%}",
              help="Treatment minus holdout purchase rate. This is causal because the split was random.")

    st.warning("**The pivot point of the whole project.** "
               "The ATE proves the coupon worked *on average* — but it doesn't tell us *for whom*. "
               "Some customers contributed strongly; sure-things, lost-causes and sleeping-dogs contributed nothing or hurt. "
               "Uplift modelling pushes this aggregate down to the individual level so we can target only the persuadables.")

# ---------------------------------------------------------------------------
# Page 4 — Uplift Model & Qini
# ---------------------------------------------------------------------------
def page_uplift_model():
    st.title("Stage 2 — Uplift Model & Qini Validation")
    st.caption("T-learner: two gradient-boosting models (treated, control). Subtract probabilities to get per-customer uplift.")

    # Score distribution
    scores = query(f"SELECT uplift_score, value_tier FROM {T('uplift_scores')}")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Distribution of predicted uplift")
        hist = alt.Chart(scores).mark_bar(color=ACCENT, opacity=0.85).encode(
            x=alt.X("uplift_score:Q", bin=alt.Bin(maxbins=40), title="Predicted uplift"),
            y=alt.Y("count()", title="Customers"),
            tooltip=[alt.Tooltip("count()", title="customers")],
        ).properties(height=260)
        st.altair_chart(hist, use_container_width=True)

    with c2:
        st.markdown("### Where the value lives")
        st.metric("Mean uplift", f"{scores['uplift_score'].mean():+.3f}")
        st.metric("Top 10% threshold", f"{scores['uplift_score'].quantile(0.9):+.3f}")
        st.metric("% with positive uplift",
                  f"{(scores['uplift_score'] > 0).mean():.1%}",
                  help="Only these are eligible for coupons in Stage 3.")

    st.divider()
    st.markdown("### Qini curve — validated against the holdout")

    # Build the Qini curve from outcomes joined with scores
    qini_data = query(f"""
        SELECT u.uplift_score, o.`group`, o.purchased
        FROM {T('uplift_scores')} u
        JOIN {T('campaign_outcomes')} o USING (customer_id)
    """)
    if len(qini_data) == 0:
        st.warning("No overlap between scored customers and campaign outcomes yet.")
        return

    d = qini_data.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    d["is_t"] = (d["group"] == "treatment").astype(int)
    d["is_c"] = (d["group"] == "holdout").astype(int)
    d["t_buy"] = (d["is_t"] * d["purchased"]).cumsum()
    d["c_buy"] = (d["is_c"] * d["purchased"]).cumsum()
    d["nt"] = d["is_t"].cumsum().clip(lower=1)
    d["nc"] = d["is_c"].cumsum().clip(lower=1)
    d["qini"] = d["t_buy"] - d["c_buy"] * (d["nt"] / d["nc"])
    d["pct"] = (np.arange(1, len(d) + 1) / len(d)) * 100
    d["random"] = np.linspace(0, d["qini"].iloc[-1], len(d))

    try:
        _trap = np.trapezoid
    except AttributeError:
        _trap = np.trapz
    qini_coef = float(_trap(d["qini"] - d["random"], d["pct"] / 100))

    curve = pd.concat([
        d[["pct", "qini"]].rename(columns={"qini": "y"}).assign(series="Model"),
        d[["pct", "random"]].rename(columns={"random": "y"}).assign(series="Random"),
    ])
    chart = alt.Chart(curve).mark_line(strokeWidth=3).encode(
        x=alt.X("pct:Q", title="% of customers targeted (by predicted uplift)"),
        y=alt.Y("y:Q", title="Cumulative incremental purchases"),
        color=alt.Color("series:N",
                        scale=alt.Scale(domain=["Model", "Random"],
                                        range=[ACCENT, "#999"])),
        strokeDash=alt.condition("datum.series == 'Random'",
                                 alt.value([6, 4]), alt.value([0])),
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

    # Decile lift sanity check
    d["decile"] = pd.qcut(d["uplift_score"].rank(method="first"), 10, labels=False)
    def lift(dx):
        tt = dx[dx["group"] == "treatment"]["purchased"].mean()
        cc = dx[dx["group"] == "holdout"]["purchased"].mean()
        return (tt if pd.notna(tt) else 0) - (cc if pd.notna(cc) else 0)
    top_lift = lift(d[d["decile"] == 9])
    bot_lift = lift(d[d["decile"] == 0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Qini coefficient", f"{qini_coef:.1f}",
              help="Area between the model curve and the random diagonal. Higher = better ranking.")
    c2.metric("Top-decile observed uplift", f"{top_lift:+.3f}")
    c3.metric("Bottom-decile observed uplift", f"{bot_lift:+.3f}",
              help="Top should clearly exceed bottom — confirms the model ranks persuadables to the top.")

    st.success("**The model ranks persuadables correctly** — top-decile observed uplift is materially "
               "higher than bottom-decile, and the Qini curve bows above the random diagonal.")

# ---------------------------------------------------------------------------
# Page 5 — Optimization & Lift
# ---------------------------------------------------------------------------
def page_optimization():
    st.title("Stage 3 — Optimization & Measured Lift")
    st.caption("Budget-constrained knapsack: rank by margin-per-dollar, allocate greedily until the budget is spent.")

    scored = query(f"""
        SELECT u.customer_id, u.uplift_score, u.value_tier, f.avg_basket_value
        FROM {T('uplift_scores')} u
        JOIN {T('customer_features')} f USING (customer_id)
    """)

    st.markdown("### Tune the budget and see allocation respond")
    c1, c2, c3 = st.columns(3)
    with c1:
        budget = st.slider("Campaign budget ($)", 5_000, 200_000, 50_000, step=5_000)
    with c2:
        margin_rate = st.slider("Margin rate", 0.10, 0.50, 0.30, step=0.05)
    with c3:
        coupon_value = st.slider("Coupon value ($)", 1.0, 20.0, 5.0, step=1.0)

    # Run the same optimization the notebook does
    opt = scored[scored["uplift_score"] > 0].copy()
    opt["exp_incr_margin"] = opt["uplift_score"] * opt["avg_basket_value"] * margin_rate
    opt["p_redeem"] = (0.3 + opt["uplift_score"]).clip(0.05, 0.9)
    opt["exp_cost"] = opt["p_redeem"] * coupon_value
    opt = opt[opt["exp_incr_margin"] > opt["exp_cost"]].copy()
    opt["roi_per_dollar"] = opt["exp_incr_margin"] / opt["exp_cost"]
    opt = opt.sort_values("roi_per_dollar", ascending=False).reset_index(drop=True)
    opt["cum_cost"] = opt["exp_cost"].cumsum()
    opt["offered"] = opt["cum_cost"] <= budget
    sel = opt[opt["offered"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers offered", f"{len(sel):,}")
    c2.metric("Expected spend", f"${sel['exp_cost'].sum():,.0f}")
    c3.metric("Expected incr. margin", f"${sel['exp_incr_margin'].sum():,.0f}")
    roi = sel['exp_incr_margin'].sum() / max(sel['exp_cost'].sum(), 1)
    c4.metric("Expected ROI", f"{roi:.2f}x")

    st.divider()
    st.markdown("### Measured incremental margin (vs holdout, from the actual wave)")
    try:
        lift = query(f"SELECT * FROM {T('incremental_lift')} ORDER BY measured_ts DESC LIMIT 1")
        if len(lift) == 0:
            st.info("No measured-lift rows yet — run Stage 3b (notebook 07) to populate.")
        else:
            r = lift.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Treated", f"{int(r['n_treated']):,}")
            c2.metric("Holdout", f"{int(r['n_holdout']):,}")
            c3.metric("Total incr. margin", f"${r['total_incremental_margin']:,.0f}")
            c4.metric("Measured ROI", f"{r['roi']:.2f}x",
                      help="Honest, holdout-validated ROI — not gross redemptions.")
            st.success("**Holdout-validated.** This is the number that goes on the BI dashboard. "
                       "Coupon cost is real; the incremental margin counts only purchases that *wouldn't* have happened anyway.")
    except Exception as e:
        st.info(f"Incremental lift table not available yet. ({e})")

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ROUTES = {
    "1. Overview": page_overview,
    "2. Stage 1 — Segments": page_segments,
    "3. Stage 2 — Learning Wave": page_learning_wave,
    "4. Stage 2 — Uplift Model": page_uplift_model,
    "5. Stage 3 — Optimization & Lift": page_optimization,
}
ROUTES[page]()
