# Research log — reconstructed VANTA decision history

**Reconstructed November 22, 2026** from prior project discussions, code and saved outputs. It is not a contemporaneous notebook. Early dates are not independently established; chronology is reliable at the phase level. See [experiment registry](experiment_registry.md) for all numerical figures and [findings](findings.md) for confidence levels.

## EX01 — initial VANTA-R, feature interactions and Diamonds

The starting interest was an explicit, compact signed interaction operator for tabular regression: one token and scalar value per original feature, independent Q/K projections, no softmax, linear readout, and gated residual. The associative computation was recognized as based on existing linear-attention work and not an independent invention. The early Diamonds result near 97.4% R² was promising; stronger MLP and tree scores motivated separating parameter efficiency from predictive superiority.

## EX02 — initialization-seed reliability

Five repetitions on the *same* Diamonds split produced low initialization variance (VANTA-R R² 0.97431 ± 0.00008, RMSE 632.12 ± 0.97) but a stronger matched MLP (R² 0.97966 ± 0.00033). The original impression of universal stability was explicitly narrowed to fixed-split initialization behavior.

## EX03 — cross-dataset generalization shock

Concrete, California Housing, Ames and NYC Airbnb were added, with Ridge, MLP, XGBoost and CatBoost. With seeds changing both split and initialization, R no longer consistently competed with the baselines. Ames seed 43 yielded R² -1.81892; two houses accounted for about 97.2% of total squared error, with one predicted at $2,059,981 against $183,850 actual. The team considered exploding weights but correctly distinguished **exploding predictions** from evidence of diverging parameter values.

## EX04 — mathematically bounded Q/K

The next design changed only Q/K norm scaling and the aggregation divisor, producing VANTA-RN:

$$
\tilde q_i=q_i/\sqrt{1+\|q_i\|^2}, \quad
\tilde k_i=k_i/\sqrt{1+\|k_i\|^2}, \quad
h=\tilde Q(\tilde K^\top v)/\sqrt F.
$$

The key insight was a Cauchy–Schwarz score bound preserving signed asymmetry and the O(Fd) associative computation. Ames mean R² improved from -0.04129 ± 1.54028 (R) to 0.85366 ± 0.02222 (RN), California improved, and Concrete initially worsened. The new questions became: has normalization made learning harder, and are the worst inputs amplifying the *value* path?

## RN interlude — gamma, diagnostics, training budget

Airbnb RN seed 42 learned gamma ~1.95 at R² ~0.0915; seed 43 learned ~0.796 at R² ~0.1256. This **did not prove** that suppressing attention improves generalization because splits, training durations and effective readout scaling varied.

On Concrete, increased training budget and a validation-selected batch size 128 improved three-seed mean RN R² from 0.64887 to 0.85103, changing the understanding of its original underperformance. Ames tuned RN improved more modestly, and tree baselines continued to lead. The separate tuned standoff on California gave RN mean R² 0.77989 and matched MLP 0.78209, with XGBoost/CatBoost near 0.853.

## Attribution and residual pathway studies

Diagnostics showed raw Q/K projections and values can be large while *normalized scores remain bounded*. Ames extreme standardized coordinates included Misc Val +33.63 and Lot Area +32.28. The tuned seed-43 prediction for row 2180 was ~$562k versus $184k actual, split into ~$286k direct plus ~$277k attention. The source-feature Misc Val contributed +$110k to the signed attention term. California sometimes showed very large opposing direct/attention terms and learned gamma above 3.

A post-hoc ablation varied the *trained* gate multiplier. Ames full mean RMSE ~$29.9k, direct-only ~$47.3k; California full ~$54.1k, direct-only ~$208.8k. Ames gate ×0.75 happened to produce mean test RMSE ~$29.845k, near the full ~$29.926k, but no gate should be chosen from examined test results. These are interventions on jointly trained weights, not retrained structural ablations.

## Scientific project practices established

Create a permanent architecture specification, mathematical notes, registry of all experiments, findings by evidence level and a dated research log. Keep both successful and failed studies; version numerical results and code separately. Treat the LaTeX manuscript as a conservative summary and leave citations in the GitHub README as well as creating a parallel BibTeX database. Optional MLflow may support experiment tracking, but its introduction should not displace readable Markdown findings and immutable result manifests.

## Intended next experiments

1. Complete the **robust numerical preprocessing** study while keeping VANTA-RN unchanged, using train-only-fit transforms and validation selection.
2. Unless preprocessing reveals a major unexpected mechanism, proceed to EX05 and examine true nonlinear feature interaction kernels.
3. For EX05, distinguish nonlinear separable vector maps that can retain O(Fd) from arbitrary scalar g(qᵀk), which generally cannot. Include a rigorously trained identity-kernel baseline and measure runtime, map width, and predictive performance.

## Historical hypotheses revised or rejected

- Universal advantage inferred from Diamonds: rejected.
- Universal stability inferred from fixed-split seeds: narrowed.
- Exploding *weights* concluded from a large Ames prediction: not established.
- Bound on Q/K scores assumed to bound predictions: mathematically false.
- Gamma value assumed to determine attention importance: unsupported.
- Feature-wise gate assumed to automatically increase expressivity: generally false for unconstrained query rows.
- Concrete early performance interpreted as expressivity limit: premature after longer-budget gain.
- Gate ×0.75 selected using test-set sweep: prohibited for confirmatory evaluation.
- O(Fd) necessarily lost with all nonlinear kernels: false in general.
