# Ames numerical preprocessing experiment

**Reference model:** VANTA-RN v0.1, source code at
dfa7ff55dc980cfc1093c1a0a6de831f53e51703.
Architecture and categorical handling were unchanged.

A training-only standardized numerical input vector was transformed by
one of four predefined policies: identity, z-score clipping to ±5,
training-fitted 1%/99% winsorization, or elementwise asinh.
The modification affected numeric Q/K tokens and both scalar-value
pathways.

## Validation search

Two split/initialization seeds, 301 and 302; selection metric:
mean validation RMSE.

| Transformation | Mean validation RMSE |
|---|---:|
| asinh | 21,040.058 |
| clip_z5 | 21,079.503 |
| standard | 21,127.174 |
| winsor_01_99 | 21,717.873 |

## Confirmation

Three disjoint split/initialization seeds, 303–305.

| Variant | Test R² mean ± SD | Test RMSE mean ± SD | Test MAE mean ± SD |
|---|---:|---:|---:|
| asinh | 0.915762 ± 0.010408 | 24,550.434 ± 4,467.074 | 14,433.241 ± 1,348.173 |
| standard | 0.891797 ± 0.028313 | 27,411.838 ± 3,080.412 | 16,888.708 ± 3,856.665 |

Paired RMSE (asinh minus standard): seed 303, -7,927.496;
seed 304, +374.796; seed 305, -1,031.513. The mean reduction
is dominated by one partition. No improvement was established
uniformly across the three splits.

The trial does not isolate input changes to query/key routing from
changes to scalar value aggregation or direct regression readout.
