# VANTA experiments

See [numerical registry](../docs/experiment_registry.md), [reproducibility guide](../docs/data_and_reproducibility.md), and [research log](../docs/research_log.md).

| Study | Source entrypoint | Scientific question |
|---|---|---|
| EX01 | baselines_ex01/run_baselines.py | Diamonds and baseline performance |
| EX02 | seeds_ex02/run_seeds.py | Initialization sensitivity on fixed data partition |
| EX03 | datasets_ex03/run_exp.py | Four-dataset generalization and baselines |
| EX04 | ../VANTA_RN/train.py | Smooth projection norm and feature-count scaling |
| RN tuning | rn_interlude/tune.py | Neural optimization and budget |
| RN training-budget comparison | rn_interlude/training_budget_test.py | Tuned matched MLP/XGBoost/CatBoost comparison |
| RN diagnostics | rn_interlude/diagnose.py, rn_interlude/trace_explosions.py | Prediction decomposition and input effects |
| RN residual ablation | rn_interlude/residual_ablation.py | Post-hoc gate interventions |
| RN preprocessing | rn_interlude/preprocessing_experiment.py | Train-only numeric-tail robustness |

Runs create artifacts under ignored results/ paths. Preserve immutable archived copies externally and document the code SHA, data version, seeds and selection protocol. Historical test splits have been inspected and are exploratory, not untouched final benchmarks.
