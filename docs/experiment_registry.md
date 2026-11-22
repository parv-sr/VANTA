# Experiment registry: numerical results and provenance

Recorded through 2026-11-22 from project console output, supplied CSVs and repository ablation JSON. These figures were **not independently re-run** for this PR. With the exception of EX02, seeds change both random initialization and data partition. Three-split standard deviations are sample SD. Reused test splits are **exploratory** after repeated investigation.

## EX01: Diamonds baseline

About 54k filtered records, nine features; VANTA-R with ~1,450 parameters obtained test R² ~0.97404, RMSE ~$635. Matched MLP and XGBoost were near R² 0.9793; CatBoost near 0.9823. These are exploratory single-split figures.

### Repository ablations (one Diamonds split; independently trained configurations)

| Model variant | Trainable params | Test R² | Test RMSE | Test MAE | Best epoch |
|---|---:|---:|---:|---:|---:|
| No attention | 33 | 0.9244535 | 1084.0918 | 717.7577 | 94 |
| Frozen Q/K | 34 | 0.9292696 | 1048.9672 | 693.9459 | 77 |
| Fixed gamma=1 | 1449 | 0.9739366 | 636.7575 | 375.9024 | 118 |

Sources: [no attention](../VANTA_R/ablation/no_attention/metrics.json), [frozen Q/K](../VANTA_R/ablation/frozen_qk/metrics.json), [fixed gate](../VANTA_R/ablation/fixed_gamma/metrics.json). Parameter budgets differ.

## EX02: Diamonds, fixed data split, five initialization seeds

| Model | Test R² mean ± SD | Test RMSE mean ± SD |
|---|---:|---:|
| VANTA-R | 0.97431 ± 0.00008 | 632.12 ± 0.97 |
| Matched MLP | 0.97966 ± 0.00033 | 562.52 ± 4.64 |

This is initialization sensitivity, **not** changing-data-split generalization.

## EX03: cross-dataset generalization, three split/init seeds

All statistics refer to test means ± sample SD; original baseline budget.

| Dataset | Model | R² | RMSE |
|---|---|---:|---:|
| Concrete | VANTA-R | 0.69951 ± 0.04963 | 8.655 ± 0.710 |
| Concrete | MLP | 0.76638 ± 0.02231 | 7.643 ± 0.322 |
| Concrete | Ridge | 0.58230 ± 0.00570 | 10.229 ± 0.089 |
| Concrete | XGBoost | 0.91915 ± 0.03122 | 4.440 ± 0.851 |
| Concrete | CatBoost | 0.93196 ± 0.02311 | 4.088 ± 0.657 |
| California | VANTA-R | 0.69360 ± 0.05927 | 63,649.905 ± 7,108.728 |
| California | MLP | 0.75336 ± 0.01699 | 57,225.170 ± 2,442.856 |
| California | Ridge | 0.64200 ± 0.02409 | 68,949.872 ± 3,044.895 |
| California | XGBoost | 0.84827 ± 0.01314 | 44,878.542 ± 2,392.854 |
| California | CatBoost | 0.84540 ± 0.01668 | 45,282.803 ± 2,803.402 |
| Ames | VANTA-R | -0.04129 ± 1.54028 | 63,083.641 ± 49,219.663 |
| Ames | MLP | 0.88302 ± 0.02554 | 28,865.955 ± 6,799.867 |
| Ames | Ridge | 0.81421 ± 0.03484 | 35,800.737 ± 2,881.473 |
| Ames | XGBoost | 0.91016 ± 0.02185 | 25,249.969 ± 6,015.278 |
| Ames | CatBoost | 0.91380 ± 0.01070 | 24,703.106 ± 4,487.788 |
| Airbnb | VANTA-R | 0.08953 ± 0.01633 | 236.796 ± 42.949 |
| Airbnb | MLP | 0.09229 ± 0.04175 | 236.298 ± 42.194 |
| Airbnb | Ridge | 0.09934 ± 0.02405 | 235.696 ± 44.463 |
| Airbnb | XGBoost | 0.12554 ± 0.03126 | 232.304 ± 44.531 |
| Airbnb | CatBoost | 0.11746 ± 0.02428 | 233.282 ± 43.728 |

**Failure case:** Ames VANTA-R seed 43 test R² -1.81892, RMSE ~$119,394; index 2180 $183,850 actual versus $2,059,981 predicted, index 956 $375,000 actual versus -$359,992 predicted. The first error accounts for 84.27% of squared error; two errors roughly 97.2%. Do not delete them to improve metrics. Airbnb uses host-grouped train/validation/test splits.

## EX04: normalized VANTA-RN, original budget

| Dataset | Test R² mean ± SD | Test RMSE mean ± SD | Test MAE mean |
|---|---:|---:|---:|
| Concrete | 0.64887 ± 0.02253 | 9.376 ± 0.323 | 7.602 |
| California | 0.75710 ± 0.02467 | 56,762.149 ± 3,208.877 | 40,001.428 |
| Ames | 0.85366 ± 0.02222 | 31,867.379 ± 2,887.322 | 14,999.674 |
| Airbnb | 0.11350 ± 0.01908 | 233.771 ± 43.449 | 71.954 |

Largest observed attention-score magnitudes: Ames 0.09554 versus bound 0.11396; California 0.29064 versus 0.33333; Airbnb 0.27589 versus 0.31623; Concrete 0.30360 versus 0.35355. On Airbnb seeds 42/43/44, RN gamma ~1.9487/0.7962/0.7064 and R² ~0.09150/0.12555/0.12345; splits and stopping epochs differ.

## RN interlude: budget tuning and standoff

Concrete validation seed 101, up to 600 epochs: batch128 RMSE 5.3827, higher LR 5.5499, width48 6.0576, gamma_init0.5 6.3057, long reference 6.7545. On seeds 42–44, tuned RN test R² 0.85103 ± 0.02053, RMSE 6.098 ± 0.398 versus original RN R² 0.64887 ± 0.02253, RMSE 9.376 ± 0.323.

Ames validation seed 101: batch128 RMSE 20,295.71, higher LR 20,321.32, gamma_init0.5 20,328.98, width48 20,486.82, long reference 20,499.37. Confirmation tuned RN R² 0.86757 ± 0.04599, RMSE 29,925.805 ± 5,036.103 versus original RN 0.85366 ± 0.02222, 31,867.379 ± 2,887.322. Optimization updates and batch-size effects are confounded.

### Tuned model standoff: mean ± sample SD, three explored splits

| Dataset | Model | Test R² | Test RMSE | Test MAE |
|---|---|---:|---:|---:|
| California | RN | 0.779891 ± 0.017385 | 54,051.684 ± 2,482.114 | 37,551.774 ± 1,379.268 |
| California | MLP | 0.782091 ± 0.016520 | 53,782.661 ± 2,375.507 | 36,901.953 ± 917.280 |
| California | XGBoost | 0.853650 ± 0.011777 | 44,074.454 ± 2,082.602 | 29,193.149 ± 738.132 |
| California | CatBoost | 0.852719 ± 0.013121 | 44,210.399 ± 2,307.643 | 29,298.524 ± 715.605 |
| Ames | RN | 0.867574 ± 0.045988 | 29,925.805 ± 5,036.103 | 14,766.999 ± 397.206 |
| Ames | MLP | 0.881393 ± 0.017998 | 28,736.668 ± 3,413.545 | 15,189.058 ± 1,500.190 |
| Ames | XGBoost | 0.908992 ± 0.019258 | 25,396.301 ± 5,595.831 | 14,812.578 ± 1,770.177 |
| Ames | CatBoost | 0.912599 ± 0.011593 | 24,864.380 ± 4,525.885 | 14,609.170 ± 1,524.663 |

California search-selected RN: batch 128, LR 0.002, d 24, max epochs 450, patience 120, validation seeds 101 and 102. California RN minus MLP paired test RMSE gaps on seeds 42/43/44: +319.78, +340.81 and +146.47. MLP narrowly won all three.

## Inference diagnostics and post-hoc residual ablation

Ames seed 43 tuned checkpoint: row 2180 actual $183,850, predicted $562,133, with direct $285,526 and attention +$276,607; 56.51% of test squared error. Extremely large standardized inputs include Misc Val +33.63 on row 2180 and Lot Area +32.28 on row 956. Row 2180 source attention: Misc Val +$110,388, Mas Vnr Area +$35,630, Gr Liv Area +$26,724; receiver effects include Neighborhood -$80,170, Condition 2 +$53,328, Land Contour -$43,819. These are signed algebraic terms, **not causal effects**.

California tuned seed 43 gamma ~3.621; row 14805 actual $187,500 and prediction ~$592,784; reference-replacement longitude change approximately -$344,086. Parameter magnitudes did not establish exploding trained weights.

Inference-only **post-hoc** residual sweep (co-trained checkpoint, not retrained model):

| Dataset test variant | R² mean ± SD | RMSE mean ± SD |
|---|---:|---:|
| Ames direct-only | 0.6865 ± 0.0689 | 47,258.558 ± 11,250.490 |
| Ames gate ×0.50 | 0.8436 ± 0.0228 | 33,300.696 ± 6,443.710 |
| Ames gate ×0.75 | 0.8722 ± 0.0243 | 29,845.218 ± 4,611.943 |
| Ames trained full ×1 | 0.8676 ± 0.0460 | 29,925.805 ± 5,036.102 |
| Ames gate ×1.25 | 0.8297 ± 0.0729 | 33,618.742 ± 6,599.139 |
| California direct-only | -2.3070 ± 0.7100 | 208,802.564 ± 23,547.833 |
| California gate ×0.50 | 0.0189 ± 0.1850 | 113,832.138 ± 11,307.688 |
| California gate ×0.75 | 0.5950 ± 0.0565 | 73,235.234 ± 5,471.922 |
| California trained full ×1 | 0.7799 ± 0.0174 | 54,051.685 ± 2,482.114 |
| California gate ×1.25 | 0.5735 ± 0.0676 | 75,122.169 ± 6,322.679 |

Ames gate ×0.75 slightly improved mean *explored test* RMSE, but this is **not a valid reason to change the gate**. Validation evidence and future independent evaluation are required. California exhibits much greater dependency on the co-trained interaction branch.
