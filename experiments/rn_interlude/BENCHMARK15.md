# Benchmark-15 — heterogeneous regression generalization

**Location:** experiments/rn_interlude/benchmark.py and benchmark_datasets.py. This is an exploratory suite and **does not change VANTA-RN**. Future variants are built with the same model constructor/forward interface, or loaded through a trusted model plugin.

The benchmark has **15 tasks** chosen to vary numerical/categorical mix, feature count, sample size, heavy tails, target noise, nonlinear physical phenomena, and geography. Each source is a KaggleHub dataset slug. Source SHA-256 and train-only dataset profiles are captured in run records. Some mirrors may alter filenames/schema in future: use --inspect before an expensive run; files are **not** bundled in Git and this PR has **not** executed all remote datasets end-to-end.

## Datasets

| ID | Kaggle slug | Target | Characteristic |
|---|---|---|---|
| concrete | prathamtripathi/regression-with-neural-networking | Strength | Small, all numeric, materials |
| california | camnugent/california-housing-prices | median_house_value | Geographic features, mixed |
| ames | shashanknecrothapa/ames-housing-dataset | SalePrice | Wide mixed features, outliers |
| airbnb | dgomonov/new-york-city-airbnb-open-data | price | Heavy-tailed listings, host-grouped splits |
| diamonds | shivam2503/diamonds | price | Large retail, numeric/categorical and geometry |
| wine_red | uciml/red-wine-quality-cortez-et-al-2009 | quality | Small all-numeric, ordinal labels treated as regression |
| wine_white | kashnitsky/mlcourse-ai | quality | Medium all-numeric chemistry, ordinal regression (large mirrored download) |
| insurance | mirichoi0218/insurance | charges | Nonlinear categorical effects, small |
| energy | elikplim/eergy-efficiency-dataset | Y1 | Small physical energy response; Y2 excluded |
| bike_hourly | lakshmi25npathi/bike-sharing-dataset | cnt | Hourly count, seasonality; casual/registered excluded |
| avocado | neuromusic/avocado-prices | AveragePrice | Regional market data, time-correlated |
| king_county | harlfoxem/housesalesprediction | price | Geographic, large housing |
| auto_mpg | uciml/autompg-dataset | mpg | Tiny tabular mixed, missing horsepower |
| graduate | mohansacharya/graduate-admissions | Chance of Admit | Small numeric, bounded response |
| student_math | dskagglemt/student-performance-data-set | G3 | Small, many categorical; prior grades G1/G2 excluded |

**Leakage controls:** Bike-sharing `casual + registered = cnt`, so both must be excluded, along with row/date IDs. Student G1/G2 are strong near-target grade proxies and intentionally excluded. Energy Y2 is a second target and excluded. King County row ID and date, graduate serial ID, and avocado Date are removed. The four original EX03 datasets retain the original dataset loader and processing rules (including host-grouped Airbnb splits). This does not eliminate all possible domain leakage: geographic neighborhoods, duplicated records, temporal dependence and repeated entities warrant additional external validations.

**Licensing:** Kaggle dataset pages may contain different sharing conditions. Before redistribution or publication, confirm each task's current license and cite the original creators. Some sources are mirrors; record versions and provenance, do not commit datasets.

## Commands

List available tasks, then **inspect/download without training**:

~~~bash
uv run python experiments/rn_interlude/benchmark.py --list-datasets
uv run python experiments/rn_interlude/benchmark.py --datasets all --inspect --continue-on-error
~~~

**Default benchmark**: one frozen RN model, three seeds, no baseline costs:

~~~bash
uv run python experiments/rn_interlude/benchmark.py
~~~

**Full comparison on the CAI server**, with all three optional competitors:

~~~bash
uv run python experiments/rn_interlude/benchmark.py --datasets all \
  --seeds 42 43 44 --models rn \
  --baselines mlp xgboost catboost --device cuda --tree-gpu \
  --epochs 450 --batch-size 128 --lr 0.001 --trees 2000 \
  --threads 8 --save-checkpoints --continue-on-error
~~~

The `--device cuda` switch moves PyTorch models to GPU, while `--tree-gpu` independently enables GPU XGBoost and CatBoost. Both require compatible CUDA drivers, package builds and free device memory. Without `--tree-gpu`, trees run on CPU even when PyTorch uses CUDA. On an unsupported tree GPU backend, rerun without `--tree-gpu` in a **fresh output folder** to avoid mixing execution settings.

**Five seeds and a subset**:

~~~bash
uv run python experiments/rn_interlude/benchmark.py \
  --datasets ames king_county energy insurance student_math \
  --seeds 42 43 44 45 46 --device auto
~~~

**Compare future variants under identical data partitions**:

~~~bash
uv run python experiments/rn_interlude/benchmark.py \
  --models rn rnx --rnx-kernel cubic --rnx-value gelu2 \
  --baselines mlp catboost --mlp-match rn \
  --output experiments/rn_interlude/results/benchmark15_rnx
~~~

Additional source models may be supplied with `--plugin /absolute/path/model.py:ClassName --plugin-name vanta-v2 --models rn vanta-v2`. The class must implement RN-compatible constructor kwargs and `forward(x_numeric,x_categorical)`. Only load trusted local Python model plugins. `--plugin-kwargs` accepts JSON kwargs. The MLP approximately matches the **selected reference model** using EX03's embedding-width formula; its actual parameter count and matching ratio are reported. Models are trained using the same early-stopping and optimizer recipe; tree models use their own early-stopped boosting recipe.

**Local data on CAI:** `--data-root /path/to/data` expects `<root>/<task-id>/<filename>` or `<root>/<filename>`. Original EX03 preprocessing conventions still apply. This is helpful after prefetching Kaggle sources.

## Reproduction and results

By default 3 seeds: 42, 43, 44. Override with `--seeds` for 3–5 **distinct** seeds. All models in a seed share the same 80/10/10 train/validation/test partition; source, preprocessing and numerical target standardization are fitted to train only. Validation chooses the best epoch or boosting round. No test-target-driven hyperparameter optimization is performed in this runner.

Artifacts in ignored `results/benchmark15/`:

- `per_seed.csv`: metrics, runtime, actual parameter count, selected epoch, and top-1/top-5 test squared-error shares;
- `summary.csv`: three–five-seed means and sample SD for R², RMSE, MAE, NRMSE, and worst-row squared-error concentration;
- `paired.csv`: per-split paired normalized-RMSE advantage against the first selected VANTA model;
- `reference_strengths.csv`: RN (or selected reference) dataset regimes ordered by NRMSE;
- `regime_summary.csv`: exploratory grouped performance by sample size, feature mix, target skew, input-tail frequency and domain;
- `ranks.csv`: per-dataset NRMSE ranks and mean ranks when competitors are enabled;
- `run_settings.json`, source SHA-256, per-run `result.json` and optional checkpoints/predictions;
- `failures.txt` when `--continue-on-error` is used.

**Across datasets, raw RMSE cannot be compared**; use NRMSE or R², and keep target-scale RMSE for each dataset. Domain/regime correlations on only fifteen tasks are exploratory; avoid general claims of causation or universal superiority.

### Caveats

The suite is *random-split tabular regression*. Bike sharing and avocado market data contain temporal dependence, and this does not establish future-time forecasting performance. Similarly, geographic housing splits are not leave-region-out evaluations. Repeating the same 15 tasks during EX05 development turns them into development data: reserve new external benchmarks for final claims.

A single fixed optimizer budget across differently sized tasks is a **controlled comparison**, not proof each model is individually optimal. Report training cost, selected epochs, and parameter-match ratios. Treat repeated model searches and comparisons as exploratory unless independent holdouts are used.


**Model support:** Built-in identifiers are `rn`, `r`, `g`, and `rnx`. The first `--models` identifier is the reference for paired differences and data-regime reports. RN remains the default. The approximately matched MLP records its actual parameter count and ratio to the reference budget.

**Output discipline:** Run configurations are fingerprinted. Changing models, GPU options, hyperparameters or feature maps requires a new `--output` directory, rather than mixing incompatible results. With `--continue-on-error`, errors are written per task/seed/model to `failures.txt`; partial summary statistics must not be presented as a complete 15-task benchmark.
