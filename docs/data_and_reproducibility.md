# Data handling, reproducibility and artifact policy

## Dependency and execution baseline

Repository requires Python 3.13+ and uses uv. Keep pyproject.toml and uv.lock versioned. Reproduce with uv sync and uv run. Some benchmarks download data from KaggleHub; record **dataset slug, resolved version, source file checksum and license** before publication. Dataset access and content can change over time.

## Splits and leakage precautions

EX03 code fits numeric median imputation, StandardScaler, categorical vocabularies, target mean/std and classical one-hot encoder on the **train partition only**. The Airbnb dataset uses host_id grouped splits; others use random splits. Each historical seed 42–44 changes **both** split assignment and initialization. A study of optimizer robustness should hold split seed fixed while varying initialization seed; a study of data robustness should vary splits with fixed or repeated initializations.

A preprocessing candidate must fit medians, standardization/quantile bounds or clipping thresholds **only on training data**. Never inspect validation or test targets when fitting transformations. Validation can select configurations; test data must not choose them. Exploration of previously tested splits is not independent confirmation.

## Reproducible run manifest (example schema; not a claimed historical run)

Save alongside each checkpoint and output bundle:

~~~yaml
experiment_id: "rn-interlude-ames-2026-11-22"
model: "VANTA-RN-v0.1"
git_sha: "actual-commit-sha"
environment:
  python: "3.13"
  uv_lock_sha256: "computed-lockfile-hash"
dataset:
  slug: "shashanknecrothapa/ames-housing-dataset"
  file_sha256: "computed-dataset-hash"
splits:
  partition_seed: 43
  initialization_seed: 43
  split_indices_sha256: "computed-split-hash"
preprocessing:
  method: "train-only-standard"
training:
  latent_dim: 24
  batch_size: 128
  lr: 0.001
  weight_decay: 0.0001
  max_epochs: 650
  checkpoint_selection: "minimum-validation-mse"
artifacts:
  checkpoint: "checkpoint.pt"
  metrics: "metrics.json"
  predictions: "test_predictions.csv"
~~~

Use actual values from the run; never fill a manifest with plausible guesses. Track best epoch and stopped epoch separately, as well as optimizer updates.

## Version control and archive boundaries

**Commit:** model code, immutable experiment configs, small CSV numerical summaries, mathematical derivations, test source, selected figures, research log, paper source and bibliography, dependency lock.

**Archive outside Git:** checkpoints, full per-row predictions, full learning curves, run-level logs, downloaded datasets. The current .gitignore intentionally ignores results/ and *.pt; this is **not** a backup. Keep raw artifacts in a read-only run directory and back up to another disk or a versioned object store. Do not remove a run when its final summary is surprising.

Recommended run naming: experiment/dataset/variant/split-seed/init-seed, plus a unique timestamp or content hash. Archive the Git commit and complete inputs with the bundle.

## Optional MLflow

MLflow is an *optional future integration*; the repository does not assume it is installed or running. For a small local study, a file-backed tracking URI with artifact uploads may be sufficient. Log run name, seed(s), git SHA, data hashes, hyperparameters, per-split metrics, best checkpoint, feature preprocessing metadata and diagnostic figures. Avoid putting large datasets or credentials in Git/MLflow accidentally. For more rigorous collaboration, use a properly backed database and object storage. MLflow must complement—not replace—the readable experiment registry and research notes.

## Research release gate

A model version is publication-ready only when: its formulation and reference tests are frozen; exact dependencies/data versions are archived; baselines are tuned fairly; untouched independent tasks are evaluated without post-hoc selection; paired uncertainty and compute costs are reported; failure cases are retained; and paper statements link to result manifests.
