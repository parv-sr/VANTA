# Experiment 6: cross-dataset width and numerical-value ablations

**Status:** implemented protocol; no Experiment 6 benchmark results are reported.
The original R, RN and RNX source files and their regression heads are unchanged.
This study evaluates RN and the existing RNX configuration with
`kernel_map="identity", value_map="gelu2"`. No deep-block, Q/K-kernel, tree-depth
or regression-head ablation is included.

## A. Dataset-dependent latent width

Evaluate `d = 4, 8, 16, 24, 32, 48, 64` using the same training recipe per task.
Primary selection is the **minimum mean validation NRMSE**, independently for
each dataset and model. Exact ties favor the smaller width. The smallest width
within 1% of the minimum is also recorded as a descriptive compactness choice;
it does not override accuracy-first selection. No global optimal d is assumed.

## B. Numerical attention-value mapping

RNX-GELU2 receives the same width grid, initialization seeds, partition seeds
and optimization budget as RN. Confirmation evaluates the union of:

1. RN at the historical reference width d=24;
2. RN at its validation-selected width;
3. RNX-GELU2 at the RN-selected width;
4. RNX-GELU2 at its independently selected width.

Identical configurations are fitted once and reused across comparisons.
`A_width` compares selected RN with RN-24. `B_fixed_width` isolates the value
mapping at fixed latent width. `B_tuned_families` gives both families equal
opportunity to select a width. GELU2 adds six parameters per numerical feature;
the fixed-width ablation is not an exactly parameter-matched comparison.

Shared inherited parameters and minibatch order are identical at a given width
and initialization seed. GELU2 initially adds a zero-valued residual. Only numeric
values entering attention are transformed; categorical values, direct values,
query/key maps and the affine regression readout remain unchanged.

## Dataset panel and partitions

Default tasks: **Diamonds, Ames, California Housing, White Wine, Bike Sharing,
and Concrete**. This covers different sample sizes, feature counts, categorical
mixes, target behavior and observed failure modes. `--datasets all` runs the
same procedure on all fifteen registered tasks; explicit task lists are supported.
The panel is developmental, not an untouched external benchmark.

Search crosses partition seeds 701/702 with initialization seeds 801/802.
Confirmation crosses partition seeds 703/704/705 with initialization seeds
803/804. Search and confirmation seeds are disjoint within each seed type.
Search must be complete and balanced before any confirmation evaluation.
Selections are frozen in the manifest; test scores never choose widths.

Wine uses exact predictor-vector group-disjoint splits, constructed without
labels. Airbnb retains its existing host-grouped policy; other tasks retain
random-row splits. Group sizes may prevent exact 80/10/10 row proportions.
Numeric imputation and standardization, categorical vocabularies and target
normalization are fitted on training rows only. No input-wide asinh transform
is added. Wine results are not directly comparable to historical random-row
wine results. Temporal extrapolation is not established by random splitting.

## Training and measurements

Defaults: AdamW, learning rate 0.001, weight decay 0.0001, batch size 128,
maximum 650 epochs, gradient norm clipping at 1, and 120-epoch early-stopping
patience. Train and validation evaluation occurs at epoch 1, every five epochs,
and the final epoch. The minimum validation normalized MSE chooses the checkpoint
with an improvement threshold of 1e-7. All final metrics use that checkpoint.
FP32 is retained; no AMP, compilation or modified model initialization is added.
Width also changes parameter count and norm geometry; this is not a pure rank-only
intervention. Fixed settings do not establish individually optimal optimization.

Recorded metrics include train/validation learning curves, confirmation-only test
MSE/RMSE/MAE/R2/NRMSE, worst-one/worst-five squared-error shares, parameter counts,
best/stopped epochs, optimizer updates, gamma, training-only dataset descriptors,
Q/K norm quantiles, and attention effective rank on a small training sample.
Effective rank is a spectral concentration statistic, not exact matrix rank.

Fit time includes learning-curve evaluations and in-memory checkpoint snapshots.
Forward timing uses a device-resident training batch of at most 512 observations,
three warmups, ten timed calls, and CUDA synchronization. It excludes preprocessing
and transfers and is not a deployment-latency measurement. Peak allocated CUDA
memory is hardware-dependent. Explicit attention matrices are created only for
16 diagnostic training observations, not in the model training forward pass.

These reliability and cost measurements are reporting controls, not a third
architecture experiment.

## Exactly four result files

```text
experiments/ex06/results/
    manifest.json
    results.csv
    learning_curves.csv
    summary.csv
```

| File | Contents |
|---|---|
| `manifest.json` | Immutable settings, source/lockfile hashes, environment, Git revision, source identities, exact split indices, train-only descriptors, frozen selections, errors |
| `results.csv` | One row per completed fit across all datasets, widths, models, seeds and phases |
| `learning_curves.csv` | Evaluation-epoch train/validation curves for all completed fits, joined by run_id |
| `summary.csv` | Model/width summaries, paired comparisons and partition-level paired means |

There are no per-dataset/per-seed directories, prediction dumps, figures or saved
model checkpoints. Best weights are retained in memory for evaluation and then
released; later checkpoint-level analysis requires reproducible retraining.
Historical outputs are neither modified nor deleted.

The summary distinguishes pooled descriptive SD from SD across initialization-
averaged partition means. Repeated initializations are not additional independent
partitions. Paired deltas are candidate minus reference: negative is favorable
for RMSE/MAE/NRMSE; positive is favorable for R2. Pairwise and partition-level win
counts are retained. Reused configurations can produce zero differences; these
are equivalent selection policies, not independent evidence. Raw RMSE is not
averaged across targets with different units.

Writes use atomic replacement and an Ubuntu-compatible directory lock. Rerunning
resumes completed fits; an interrupted fit restarts. Orphaned learning-curve rows
are removed on resume. Changed settings, executable hashes, data, dependencies or
device are rejected instead of silently mixing experiments. Failures stop execution
and are recorded in the manifest. Use a different output directory for a new study.

## Commands (Ubuntu; repository root)

```bash
uv run python -m unittest discover -s tests -p 'test_ex06.py' -v
uv run python experiments/ex06/run_experiment.py --dry-run
uv run python experiments/ex06/run_experiment.py --stage all --device cuda --threads 8
```

The default six-task panel has **336 search fits** and at most **144 confirmation
fits**. Confirmation usually requires fewer fits because selected configurations
overlap. These are fit counts, not runtime estimates.

The two stages can be invoked separately with otherwise identical options:

```bash
uv run python experiments/ex06/run_experiment.py --stage search --device cuda --threads 8
uv run python experiments/ex06/run_experiment.py --stage confirm --device cuda --threads 8
```

For a separately registered fifteen-task study:

```bash
uv run python experiments/ex06/run_experiment.py --datasets all --stage all \
  --device cuda --threads 8 --output experiments/ex06/results_all15
```

`--device auto` selects CUDA when available; explicit CUDA fails if unavailable.
Deterministic PyTorch operations are requested within the recorded environment;
bitwise reproducibility across hardware/software versions is not asserted.
Unsupported deterministic operations raise an error instead of silently changing
execution. `--data-root` uses the existing local-data registry layout.

## Verification boundary

The included network-free tests use synthetic data and the existing RN/RNX model
classes. They check complete validation grids, accuracy versus compact selection,
paired configurations, duplicate grouping, unchanged shared initialization/head,
search test isolation, four-file output and end-to-end resumption. This addition
contains no real-dataset training results or GPU performance claims.
