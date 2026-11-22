# VANTA research documentation

This directory records model definitions, experimental methods, numerical results,
and limitations. The source implementations, dataset-processing code, dependency
lockfile, and run configurations are required to reproduce measurements.

| Document | Content |
|---|---|
| [architecture.md](architecture.md) | Implemented mathematical formulations |
| [mathematical_notes.md](mathematical_notes.md) | Established algebraic properties and computational complexity |
| [experiment_registry.md](experiment_registry.md) | Historical experiments, numerical tables and provenance |
| [benchmark15_results.md](benchmark15_results.md) | Completed 15-task benchmark and performance stratification |
| [findings.md](findings.md) | Empirical findings and their evidential limits |
| [research_log.md](research_log.md) | Chronological record of completed experiments |
| [data_and_reproducibility.md](data_and_reproducibility.md) | Evaluation, artifacts and reproducibility procedures |
| [ex05_preprocessing_freeze.md](ex05_preprocessing_freeze.md) | Completed input-transformation ablation |

Exploratory studies reuse datasets and partitions across model development. Reported
test results must not be treated as independent confirmatory estimates. Where
available, archive per-seed results, exact dataset source checksums, code versions,
checkpoint-selection rules and full model configurations.

The repository does not currently include a configured MLflow tracking service.
