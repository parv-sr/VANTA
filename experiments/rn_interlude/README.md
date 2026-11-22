# VANTA-RN interlude: optimization, diagnostics, and input robustness

The source of numerical findings is [docs/experiment_registry.md](../../docs/experiment_registry.md).

Existing code: diagnose.py, tune.py, training_budget_test.py (validation-selected model comparison), trace_explosions.py, residual_ablation.py, common.py.

The new preprocessing_experiment.py keeps VANTA-RN v0.1 unchanged, retaining identical categorical inputs, target normalization, splits, loss and optimizer while modifying only the numeric feature values **after EX03 train-fitted standardization**. Candidates: standard baseline, fixed ±5 z-score clip, training-fitted 1%-99% winsorization, and signed asinh compression.

Search uses seeds 301 and 302 on validation only; confirmation uses disjoint seeds 303–305. It retrains the **validation-selected variant versus standard baseline**, rather than selecting from explored test results.

Run from root:

~~~bash
uv run python3 experiments/rn_interlude/preprocessing_experiment.py --dataset ames --stage search
uv run python3 experiments/rn_interlude/preprocessing_experiment.py --dataset ames --stage confirm
~~~

Optional California replication: replace ames with california. The script saves per-run configurations, fitted policy, checkpoint, metrics and test predictions into ignored results/preprocessing. Confirm preprocessing choices on independent data before publication.

Training-budget comparison:

~~~bash
uv run python3 experiments/rn_interlude/training_budget_test.py --dataset california --stage search
uv run python3 experiments/rn_interlude/training_budget_test.py --dataset california --stage confirm
~~~

Historical output remains under `results/standoff/` for compatibility with existing checkpoints and attribution scripts; only source-file naming is standardized.

The post-hoc residual gate sweep is **not** an independently trained architecture ablation and its test numbers cannot select the gate.
