# Experimental chronology

This document summarizes completed experiments in chronological order.
Numerical details appear in [experiment_registry.md](experiment_registry.md),
[benchmark15_results.md](benchmark15_results.md) and
[ex05_preprocessing_freeze.md](ex05_preprocessing_freeze.md). Experimental
descriptions are retrospective; original run timestamps and complete per-row
artifacts are not available for every study.

## EX01 — Initial formulation and Diamonds evaluation

VANTA-R was implemented as a signed, non-softmax feature-interaction regression
model with separate query/key projections, a scalar-gated residual pathway,
and a linear readout. The associative aggregation implements an established
linear-attention factorization. On the initial Diamonds evaluation, test
R² was approximately 0.974 with approximately 1,450 trainable parameters.
Matched MLP and tree-ensemble baselines attained higher R².

Independent Diamonds ablations of attention removal, frozen query/key
projections and fixed gamma are retained as committed JSON records.

## EX02 — Fixed-split initialization study

Five initializations on a fixed Diamonds partition produced VANTA-R
R² = 0.97431 ± 0.00008 and RMSE = 632.12 ± 0.97.
The matched MLP produced R² = 0.97966 ± 0.00033 and RMSE = 562.52 ± 4.64.
The study measures initialization variability on one partition only.

## EX03 — Multi-dataset evaluation

Concrete, California Housing, Ames Housing and NYC Airbnb were evaluated
against Ridge, parameter-matched MLP, XGBoost and CatBoost. Three seeds
changed the data partition and model initialization together. The original
model showed high error variance on Ames: seed 43 produced R² = -1.81892
and a prediction of $2,059,981 against an actual value of $183,850.
No observations were removed from the reported test metrics.

## EX04 — Smooth projection normalization

VANTA-RN introduced smooth L2 normalization of query and key feature
vectors and replaced the original sqrt(d) attention divisor with sqrt(F).
These modifications bound individual attention coefficients but not the
regression output. The original-budget Ames R² changed from
-0.04129 ± 1.54028 (R) to 0.85366 ± 0.02222 (RN). California improved;
Concrete worsened under the initial budget.

## RN interlude — Optimization and model comparison

A longer neural training budget and validation-selected minibatch size
changed Concrete RN mean test R² from 0.64887 to 0.85103. On California,
a separately tuned comparison produced RN R² = 0.779891 and matched MLP
R² = 0.782091, with XGBoost and CatBoost around 0.853.
Ames RN mean RMSE was $29,925.805 in the tuned comparison.

Inspection of saved model outputs identified extreme predictions and large
direct/interaction contributions on selected Ames and California cases.
A post-hoc residual-gate sweep found strong dependence on jointly trained
direct and interaction branches. These results do not estimate the
performance of independently retrained attention-free models.

## Numeric input-preprocessing study — Ames

Standard preprocessing, z-score clipping, training-fitted winsorization,
and signed asinh compression were compared using validation seeds 301
and 302. Asinh was selected for confirmation on seeds 303–305.
Mean confirmation test RMSE was $24,550.434 (asinh) versus
$27,411.838 (standard). The three paired RMSE changes were
-$7,927.496, +$374.796 and -$1,031.513. The intervention
changes numerical query/key inputs, attention values, and direct values
simultaneously.

## EX05-A — Single-block nonlinear feature maps

The RNX model implemented separate query/key vector maps and a
numeric attention-value transformation with unchanged associative
aggregation. Validation selected a componentwise tanh Q/K map and
a two-unit GELU value mapping on Ames. Confirmation across split/init
seeds 403–405 produced the following mean test RMSE:

| Model | RMSE | R² |
|---|---:|---:|
| RN reference | 30,393.431 | 0.820277 |
| Tanh Q/K | 30,332.108 | 0.821290 |
| GELU2 attention values | 29,595.707 | 0.834308 |
| Combined tanh/GELU2 | 29,566.968 | 0.834786 |

The confirmation standard deviations were large. These results do not
demonstrate a general benefit across datasets.

## Benchmark-15 — Heterogeneous tabular regression

A subsequent study evaluated RN, a parameter-matched MLP, XGBoost and
CatBoost across 15 tabular regression datasets and three combined
split/initialization seeds. All 180 model-dataset-seed runs are present
in the supplied per-seed results. RN exceeded MLP mean R² on eight
tasks and the higher of both tree-ensemble means on two tasks
(King County and Graduate Admissions). XGBoost and CatBoost maintained
superior average cross-dataset ranks.

Detailed results, fixed protocols and limitations are documented in
[benchmark15_results.md](benchmark15_results.md).
