# EX05-A: Nonlinear Q/K feature maps and scalar-value maps

**Status:** experimental code; benchmark results pending. VANTA-RN v0.1 is frozen at main commit dfa7ff55dc980cfc1093c1a0a6de831f53e51703.

The implementation in [VANTA_RNX/model.py](../../VANTA_RNX/model.py) extends RN using two *separate* vector-map functions on Q and K (after smooth normalization), and an independent scalar mapping on the **numeric values entering attention**. The direct residual retains its original v; categorical scalar embeddings are not changed.

For maps phi, psi and numeric value transform u:

$$
h=\frac{\Phi(\tilde Q)\big(\Psi(\tilde K)^\top u\big)}{\sqrt F},\qquad
\hat y=w^\top(v+\gamma h)+b.
$$

The aggregation is **exactly associative O(Fd)** for d-dimensional maps; no F×F score matrix is materialized in forward(). Mapped Q/K norms are rescaled to the original RN norm, retaining the RN score bound. Explicit attention_matrix() is available for inspection only.

## Factorial isolation protocol

- Baseline: identity Q/K, identity values. Must match the exact frozen RN model.
- Q/K only: fixed componentwise tanh, learned residual cubic. Cubic has separate learned query/key parameters and is initialized as exact identity.
- Values only: fixed asinh, learnable identity-initialized asinh mixture, or a two-hidden-unit GELU residual transform per numeric feature.
- Combination: apply independently validation-selected Q/K map and validation-selected value map. Compare with separate single-factor models.
- All use EX03 standard numeric inputs; **not** input-wide asinh preprocessing (which also changes the Q/K and direct path).
- Search on split/init seeds 401 and 402 using validation RMSE only. Confirm on disjoint seeds 403–405 without tuning on tests.
- Frozen training recipe: latent_dim 24, batch 128, AdamW, weight decay 0.0001, gradient clipping 1.0, max epochs 650 for Ames and 450 for California, patience 120, LR 0.001 for Ames and 0.002 for California. Report parameter counts and optimizer updates; the variants are not exactly parameter matched.
- Test metrics and per-row predictions are saved **only** in confirmation.

Run from repository root:

~~~bash
uv run python -m unittest discover -s tests -v
uv run python experiments/ex05/run_feature_maps.py --dataset ames --stage search
uv run python experiments/ex05/run_feature_maps.py --dataset ames --stage confirm
~~~

Repeat on California as warranted. Complete artifacts in experiments/ex05/results/ are excluded from Git and must be backed up.

## Subsequent, separate studies

EX05-B: stack 2–3 shallow identity-kernel VANTA blocks (first static Q/K routing, then dynamic) with carefully initialized per-feature gates; record state norms and gradients. Cost O(LFd). Do **not** introduce deep stacking into EX05-A.

EX05-C: combine only independently successful mappings and depth; compare at equal training budgets, report accuracy/parameters/latency and standard deviations.

EX05-D: compare alternative efficient separable nonlinear kernel families; explicit scalar g(A) is not proposed as a deployed operator since it generally loses the existing factorization. Nonlinear maps can be evaluated without any F² matrix.
