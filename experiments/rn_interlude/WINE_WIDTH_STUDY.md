# Wine feature-space audit and latent-width sensitivity

## Scope

These are reproducible analysis scripts for the original
VANTA-RN v0.1 formulation. Neither script modifies the model.
**No new wine audit or latent-width results are claimed in this
document.** Both scripts write ignored, run-level outputs and require
actual dataset access for execution.

## Dataset audit

Run from the repository root:

~~~bash
uv run python -m unittest discover -s tests -v
uv run python experiments/rn_interlude/wine_audit.py
~~~

The audit loads the exact dataset versions available on the host,
records SHA-256, and writes results under
\`experiments/rn_interlude/results/wine_audit/\`.

For each wine dataset it computes:

- Original numerical/categorical feature counts and sample count;
- Distribution of wine-quality targets, using training rows;
- Fraction of repeated predictor vectors and groups with inconsistent targets;
- Train/test overlap of identical predictor vectors under original
  random-row splitting versus predictor-group-disjoint splitting;
- Training-only numerical missingness, skew, extreme z-scores,
  pairwise Pearson correlations, PCA effective rank and the number of
  components needed to explain 90% of standardized training variance.

Grouping is based on **predictor values only**, never on the label.
The group-disjoint comparison uses two sequential GroupShuffleSplit
partitions. Duplicate predictor groups cannot occur across partitions,
but target-label noise within a feature group is retained. Group sizes
are unequal, so exact 80/10/10 row counts are not guaranteed.

These measurements assess possible duplicate-row leakage; they do not
establish a causal effect on predictor accuracy until models are
evaluated under the new split.

## Latent-width experiment

Validation search: \`d = 4, 8, 16, 24, 32, 48, 64\`.
The search uses partition seeds 501/502 crossed with initialization
seeds 601/602 (four independent partition-initialization combinations
per width and dataset). Only validation data are used for width
selection. The predetermined rule selects the **smallest width whose
mean validation NRMSE is within 1% of the best width**.

Confirmation: partition seeds 503/504/505 crossed with initialization
seeds 603/604/605. The two configurations are the frozen baseline
\`d=24\` and the validation-selected width. Each receives nine
paired, independently initialized evaluations. Neither test
performance nor the observed outcome of these confirmation
partitions can change the selected width.

The initial analysis covers red and white wine with **feature-vector
group-disjoint** partitions. Additional non-wine datasets may be
evaluated with \`--split-mode row_random\` in separate output directories.

Common optimizer: AdamW, learning rate 0.001, batch size 128,
maximum 650 epochs, weight decay 0.0001, max gradient norm 1,
minimum validation-MSE checkpoint with patience 120, evaluated
every five epochs. These fixed training settings isolate changes
in latent width, but they are not necessarily individually optimal.

The sweep records parameter count, validation/test R², RMSE, MAE,
normalized RMSE, worst-row SSE shares, best epoch, optimizer
update count, learned gamma, training time, Q/K norm quantiles and
effective attention rank on training observations. Explicit attention
matrices are computed only for a small training diagnostic sample;
model training uses O(Fd) aggregation.

### Commands (Ubuntu / uv)

~~~bash
# Inspect duplicate structure and training feature space
uv run python experiments/rn_interlude/wine_audit.py

# Search on wine datasets, GPU if available
uv run python experiments/rn_interlude/latent_dimension_sweep.py \
  --stage search --device auto

# Confirm selected width vs frozen d=24
uv run python experiments/rn_interlude/latent_dimension_sweep.py \
  --stage confirm --device auto
~~~

A smaller preflight may use \`--datasets wine_red --widths 8 24 48\`.
Use a separate \`--output\` directory when changing width grids or
other experiment settings. Search and confirmation must use the
same configuration to ensure reproducibility.

Optional replication on energy, California, Ames or other registered
datasets uses \`--split-mode row_random\` and a new output directory.
The broad 15-dataset benchmark remains a separate exploratory study.

Run outputs are under
\`experiments/rn_interlude/results/latent_dimension/\`. Search
produces \`search_metrics.csv\`, \`validation_width_ranking.csv\`
and \`selection.json\`. Confirmation produces paired results and
\`confirmation_summary.csv\`. Full model checkpoints and test
predictions are optional. Save checkpoints for downstream
representation analyses if storage permits.

## Interpretation limits

At the original wine feature count F=11, increasing d above 11
cannot increase the theoretical upper bound on attention-matrix
rank, although it changes the representation and optimization
properties of feature-specific projections. Comparing the
validation-selected width with d=24 does not directly identify
whether poor white-wine accuracy results from interaction order,
optimization, duplicate-row overlap or measurement noise.

If all seven widths perform similarly on a feature-group-disjoint
partition, increasing latent width alone is unlikely to resolve
the original performance gap under these training conditions.
