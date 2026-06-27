# Coupon Personalization with Uplift Modelling on Databricks

> End-to-end proof of concept: segment customers, run a controlled treatment/holdout campaign, train an uplift model, allocate offers under a budget constraint, and measure incremental lift — all on Databricks.

A working POC built end-to-end on Databricks — pipeline + Feature Store + MLflow + a live Databricks App for visualisation. The aim was not just a working model but a **defensible methodology**: every step grounded in causal inference, every metric tied to a business decision.

## Demo screenshots

The app is a five-screen Streamlit walkthrough hosted as a Databricks App. It reads live from the Delta tables and MLflow model the pipeline produces.

### 1\. Overview — customer routing at checkout

!\[Overview](docs/screenshots/01\_overview.png)

20,000 customers routed three ways: **known** (modelled by uplift), **sparse** (basket-driven for now), **unknown** (cold start, basket-driven rules). The hybrid handling is the practical answer to cold start — not every customer needs the model.

### 2\. Stage 1 — Segments

!\[Segments](docs/screenshots/02\_segments.png)

Known customers clustered on RFM + behavioural features, profiled and labelled into value tiers. The tier is both a targeting lever **and** a feature for the uplift model — so the model can learn that high-value customers respond differently than low-value ones.

### 3\. Stage 2 — Learning Wave: Treatment vs Holdout

!\[Learning Wave](docs/screenshots/03\_learning\_wave.png)

The first wave is the **learning wave**: 80/20 randomized split stratified by value tier. The balance check confirms \~80% treatment in every tier (0.78–0.80). The aggregate treatment effect (ATE) of **+13.3%** proves the coupon works *on average* — but doesn't tell us *for whom*. That's the pivot to uplift modelling.

### 4\. Stage 2 — Uplift Model

!\[Uplift Model](docs/screenshots/04\_uplift\_model.png)

T-learner: two gradient-boosting models (one on treated, one on holdout). Subtract their probabilities for each customer to get per-customer predicted uplift. **91.4%** of customers have positive uplift; the top-10% threshold is +0.239 — that's how much extra probability a coupon adds for the most persuadable group.

### 5\. Qini curve — validated against the holdout

!\[Qini Curve](docs/screenshots/05\_qini\_curve.png)

The model curve bows clearly above the random diagonal, peaking around 80–85% targeting depth. **Qini measures incremental purchases captured as you target down the ranking** — not accuracy, not AUC. It's the right metric for uplift because it directly rewards ranking the genuinely persuadable customers to the top.

### 6\. Stage 3 — Optimization \& Measured Lift

!\[Optimization](docs/screenshots/06\_optimization.png)

Live budget slider that re-runs the optimizer in real time. Per-customer **expected incremental margin = uplift × basket × margin\_rate**; **expected cost = P(redeem) × coupon\_value**. Rank by margin-per-dollar, cumulative-sum cost, cut at the budget — a budget-constrained knapsack solved greedily (provably near-optimal when items are small relative to budget). The measured incremental margin at the bottom is holdout-validated — the honest after-the-fact number.

> \*\*Note on the synthetic-data measured ROI.\*\* The "measured ROI" tile can show small positive or negative values on this synthetic dataset because the experimental wave has only a few thousand customers — well within statistical noise. The point of the screen is to demonstrate the \*structure\*: an honest treated-vs-holdout comparison that produces a defensible number regardless of which way it falls. On real data with hundreds of thousands of customers per wave, the noise floor drops and the expected/measured gap closes.

```

## Caveats

* **Data is synthetic.** The pipeline generates realistic-looking POS and customer data so it runs anywhere. In production you'd swap `01\_data\_generation.py` for source reads (POS, CRM, Marketo). The engineering and modelling logic is unchanged.
* **Optimizer is greedy.** Greedy knapsack is provably near-optimal at this problem's scale (items small relative to budget) and is fully transparent. For interdependent constraints (per-segment envelopes, channel caps, coupon-value optimization), swap to OR-Tools / PuLP — the framework supports it.
* **POC scope.** Real-time scoring at POS is sketched as a fallback in Stage 3 but not implemented as a serving endpoint. The architecture supports it (Databricks Model Serving + Feature Store online lookup); the productionisation is out of scope for this POC.

## License

MIT — see [LICENSE](LICENSE). Use freely for portfolio reference, learning, or as a starting point.

\---

Built by [@raianupam171126](https://github.com/raianupam171126).

