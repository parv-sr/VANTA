# Findings: supported conclusions, limitations, rejected hypotheses

**Status:** exploratory evidence through 2026-11-22. For all measured values and protocols see [experiment_registry.md](experiment_registry.md); for theory see [mathematical_notes.md](mathematical_notes.md). These are *not* general performance claims on untouched external data.

## Established mathematical properties

1. VANTA-R and RN aggregate signed and generally asymmetric rank-≤d scores associatively in O(Fd) arithmetic, using a known linear-attention technique.
2. VANTA-RN's smooth L2 normalization gives |A_ij| < 1/sqrt(F) for finite Q/K. This **does not** imply bounded h, gamma, readout, or prediction.
3. Separate query and key projections permit directional source-versus-receiver feature contribution accounting; algebraic contributions are not causal feature importances.
4. Finite-dimensional separable nonlinear feature maps can preserve O(Fm) aggregation; O(Fd) is possible when feature-map dimension and evaluation scale as O(d).

## Observed empirical findings

- **Diamonds success is dataset-specific.** The initial R² around 0.974 was below stronger MLP/tree baselines and did not establish broad superiority.
- **Initialization consistency is not cross-split consistency.** On a fixed Diamonds split five-seed VANTA-R R² SD was ~0.00008; changing split and initialization together in EX03 produced larger variation and catastrophic Ames failure.
- **Normalization mitigated, not eliminated, extreme Ames extrapolation.** VANTA-R on Ames seed 43 predicted ~$2.06M for a ~$184k house, yielding R² -1.81892 on that run. RN's attention scores remained within bound, but an RN tuned checkpoint still predicted ~$562k on that house.
- **Optimization budget is material.** Concrete RN R² improved from 0.64887 to 0.85103 (three-seed means) under extended training and batch-size 128, but optimizer-update count, noise and wall-clock budget were not isolated independently.
- **Matched MLP remains competitive.** California tuned RN R² 0.77989 versus MLP 0.78209, with RN slightly worse per-seed RMSE on all three explored splits; Ames RN 0.86757 versus MLP 0.88139 in R², although RN mean MAE ~$14,767 versus ~$15,189.
- **Boosted trees remain stronger on overall R²/RMSE** in the studied tuned California and Ames comparisons; this should remain visible in any paper.
- **Rare rows dominate some squared-error metrics.** Ames tuned seed 43 row 2180 contributes 56.51% of squared error; this does not show up equally in MAE.
- **The fitted interaction branch matters.** Removing it *without retraining* raises Ames mean RMSE ~$29.9k to ~$47.3k and California ~$54.1k to ~$208.8k. These do not measure retrained no-attention architectures.
- **Gamma alone is not an explanatory variable.** Its effect is gamma × readout × aggregation, and across seeds both data partition and stopping epoch vary.

## Rejected, narrowed, or unsupported hypotheses

| Earlier conjecture | Disposition |
|---|---|
| High Diamonds R² implies generalization superiority | **Rejected as a broad claim** by EX03 and standoff |
| Small fixed-split seed SD implies universal stability | **Narrowed:** split variability is distinct |
| Ames catastrophic error proves exploding learned weights | **Unsupported:** fitted parameter magnitudes are finite/moderate; large inputs and residual contributions remain |
| Bounded attention scores imply bounded predictions | **Mathematically false** |
| Lower gamma is the direct cause of Airbnb's higher R² | **Unsupported:** splits/stopping confound interpretation |
| Per-feature gate automatically expands function family with unbounded Q | **Generally false** under query-row rescaling |
| Poor original-budget RN Concrete score proves insufficient representation | **Premature**, contradicted by substantial optimization gains |
| Best observed test gate multiplier should replace trained gamma | **Rejected:** test sweep is diagnostic, not model selection |
| True nonlinear attention must be quadratic in F | **False in general** with suitable separable feature maps |
| Immediately add further attention normalization or kernels | **Deferred** pending robust-input study |

## Open questions

1. Can a **training-only-fitted** robust numeric transformation reduce high-leverage extrapolation while retaining typical accuracy?
2. Is Q/K projection saturation a *cause* of optimization limitations or a consequence of learned representations?
3. What is the expressivity cost of pairwise-only interactions, versus an ordinary MLP?
4. Which exact/approximate nonlinear kernels preserve signed scores and linear-in-feature-count complexity with manageable feature-map width?
5. How do results change on strictly untouched tasks and when varying initialization separately from the data partition?

## Evidence requirements before manuscript claims

Independent held-out datasets/splits, paired uncertainty, documented compute budgets, stable code/config manifests, no test-set selection, and direct ablation under matched retraining budgets.
