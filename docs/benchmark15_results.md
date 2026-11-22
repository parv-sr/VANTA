# Benchmark-15: tabular regression results

## Scope and protocol

Fifteen public tabular regression datasets were evaluated with VANTA-RN,
a parameter-matched embedding MLP, XGBoost and CatBoost.
For each task and model, three seeds (42, 43 and 44) determined both
data partition and model initialization where applicable.
Data splits were approximately 80% training, 10% validation and 10% test,
except the host-grouped NYC Airbnb partition. Numerical imputers,
scalers, categorical vocabularies and target standardization were fitted
using training rows exclusively. Neural training used a maximum of
650 epochs, batch size 128 and AdamW; tree learners used a maximum
of 2,000 iterations and validation early stopping.

The source for this table is the 180-row per-seed result export provided
for the completed benchmark. This is a **retrospective exploratory
summary**; the benchmark was not rerun in the documentation change.
The source CSV includes dataset file SHA-256 values, training-set
descriptors, selected epochs, optimizer updates, runtime, parameters,
validation/test metrics and worst-row squared-error shares. The original
per-seed CSV should be retained alongside the experiment artifacts.

## Dataset-level results

Values are **mean test R² over seeds 42–44**. Raw RMSE across different
targets is not comparable; use target-standardized NRMSE for cross-task
analyses. The feature count is number of original columns, before
categorical one-hot expansion.

| Dataset | Rows | Numeric | Categorical | RN | Matched MLP | XGBoost | CatBoost |
|---|---:|---:|---:|---:|---:|---:|---:|
| NYC Airbnb | 48,884 | 7 | 3 | 0.1100 | 0.0613 | 0.1236 | 0.1174 |
| Ames | 2,930 | 35 | 42 | 0.8676 | 0.8761 | 0.9088 | 0.9085 |
| Auto MPG | 398 | 5 | 2 | 0.8803 | 0.8791 | 0.8723 | 0.8806 |
| Avocado | 18,249 | 9 | 2 | 0.7728 | 0.7777 | 0.8696 | 0.8255 |
| Bike hourly | 17,379 | 4 | 8 | 0.9230 | 0.9510 | 0.9445 | 0.9401 |
| California Housing | 20,640 | 8 | 1 | 0.7792 | 0.7746 | 0.8514 | 0.8391 |
| Concrete | 1,030 | 8 | 0 | 0.8536 | 0.8567 | 0.9169 | 0.9071 |
| Diamonds | 53,917 | 6 | 3 | 0.9786 | 0.9814 | 0.9824 | 0.9821 |
| Energy efficiency | 768 | 8 | 0 | 0.9939 | 0.9599 | 0.9991 | 0.9984 |
| Graduate admission | 500 | 7 | 0 | 0.8027 | 0.8059 | 0.7709 | 0.7761 |
| Insurance | 1,338 | 3 | 3 | 0.8100 | 0.7860 | 0.8161 | 0.8210 |
| King County houses | 21,613 | 17 | 1 | 0.9055 | 0.8959 | 0.9039 | 0.9014 |
| Student math | 395 | 13 | 17 | 0.0853 | 0.0131 | 0.2114 | 0.1606 |
| Wine, red | 1,599 | 11 | 0 | 0.4399 | 0.4275 | 0.4535 | 0.4579 |
| Wine, white | 4,898 | 11 | 0 | 0.3734 | 0.3837 | 0.5393 | 0.4522 |

Averaging each dataset's model ranks (higher R² is better) gives
XGBoost 1.667, CatBoost 2.000, RN 3.133 and matched MLP 3.200.
RN's mean R² exceeded MLP on 8/15 tasks and both tree ensembles on
2/15 tasks (King County and Graduate Admissions). On matched
dataset-seed pairs, RN had lower RMSE than MLP on 26/45 evaluations.
The limited task set and previously explored datasets preclude
generalization claims beyond these specific comparisons.

## Wine datasets

Both tasks have 11 numerical input variables, no categorical inputs,
and use latent width d=24. RN has exactly 1,069 trainable parameters;
the matched MLP has 1,067.

| Dataset/model | Test R², mean | Test NRMSE, mean |
|---|---:|---:|
| Red wine / RN | 0.4399 | 0.7605 |
| Red wine / MLP | 0.4275 | 0.7696 |
| Red wine / XGBoost | 0.4535 | 0.7514 |
| Red wine / CatBoost | 0.4579 | 0.7475 |
| White wine / RN | 0.3734 | 0.8026 |
| White wine / MLP | 0.3837 | 0.7958 |
| White wine / XGBoost | 0.5393 | 0.6885 |
| White wine / CatBoost | 0.4522 | 0.7507 |

The paired white-wine XGBoost-minus-RN test R² differences for seeds
42, 43 and 44 were +0.238439, +0.128728 and +0.130583.
The white-wine gap was therefore present in all three partitions.
Neither model was tuned independently for this dataset.
The mean worst-one-row squared-error concentration on white wine
was low, relative to the Ames outlier-dominated cases.

The white-wine model comparison does not identify whether the gap is
caused by feature dependence, nonlinear interaction order, latent
dimension, optimization, duplicate-row overlap or other dataset
properties. A separate feature-space audit is necessary.

## Limitations and comparative fairness

- The seed changes both initialization and partition; sources of
  variation are not separable in this benchmark.
- Original numeric features undergo train-only median imputation and
  standardization. CatBoost uses its native categorical representation,
  whereas XGBoost uses one-hot encoded categories and RN/MLP use embeddings.
- The MLP is approximately parameter matched but has a different
  inductive bias. Tree-model complexity and training budgets are not
  parameter matched to neural models.
- Existing code reports training time, but GPU execution details, launch
  overhead and per-model stopping conditions differ.
- Wine duplicate feature vectors can cross random training and test
  partitions. Measured overlap on these exact file versions is not
  yet available; no causal interpretation of this potential leakage
  is justified by the benchmark alone.
- For Bike Sharing and Avocado, random-split performance is not evidence
  of future-time forecasting accuracy.
- Model-development decisions have involved previously inspected
  partitions; these results are not independent confirmation benchmarks.

**Source files:** the original benchmark code
([benchmark.py](../experiments/rn_interlude/benchmark.py),
[dataset registry](../experiments/rn_interlude/benchmark_datasets.py))
and the archived per-seed results generated by that code.
