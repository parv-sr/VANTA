# Research documentation

This directory is the evidence and decision record for VANTA. See the root README for an introduction, and paper/main.tex for the research manuscript.

| File | Purpose |
|---|---|
| [architecture.md](architecture.md) | Exact formulations and model versioning |
| [mathematical_notes.md](mathematical_notes.md) | Derivations, bounds, complexity, expressive limitations |
| [experiment_registry.md](experiment_registry.md) | Protocols and numerical results for each study |
| [findings.md](findings.md) | Supported findings, unsupported claims, limitations |
| [research_log.md](research_log.md) | Chronology of questions, decisions, rejected hypotheses |
| [data_and_reproducibility.md](data_and_reproducibility.md) | Reproducibility contract, artifacts and optional tracking |
| [metrics/reported_results.csv](metrics/reported_results.csv) | Curated aggregate numerical snapshot |

**Evidence status:** OBSERVED = documented in archived local investigator logs, user-supplied CSVs, or committed metrics; INFERRED = interpretation; OPEN = untested hypothesis; NOT SUPPORTED = evidence does not warrant claim. This documentation PR does not rerun every original experiment, and the historical splits have already informed development.

## Commit practice

- Hypothesis and protocol before run; recorded code revision and dependency lock.
- Train-fitted preprocessing and validation-based model selection; test once.
- Save split and initialization seeds *separately*, dataset releases, parameters, best checkpoint and per-seed predictions.
- Preserve failed studies and negative results; never silently overwrite a versioned architecture.
- Version small figures/summary tables in Git and archive bulky artifacts separately.
- Use paired per-split comparisons and report uncertainty, not just best-score anecdotes.

An optional MLflow workflow is described in the reproducibility guide; MLflow is not yet configured as a dependency.
