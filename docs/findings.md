# Findings and limitations

Findings are restricted to implemented model properties and observed
experiments. Refer to [the numerical registry](experiment_registry.md),
[Benchmark-15](benchmark15_results.md), and [preprocessing results](ex05_preprocessing_freeze.md)
for supporting values. Repeatedly evaluated datasets are exploratory
rather than independent confirmation sets.

## Mathematically established properties

- VANTA-R and RN compute their interaction aggregations associatively
  without materializing the feature-by-feature attention matrix.
  Aggregation arithmetic is O(Fd), independent of dataset row count.
- RN's smooth projection normalization yields
  |A_ij| < 1/sqrt(F) for finite vectors. It does not bound the final
  prediction because values, gamma and readout weights remain unrestricted.
- The score matrix has rank at most min(F,d). Increasing latent width
  beyond F cannot increase this upper bound, although it changes the
  parameterized query/key functions and optimization geometry.
- The implemented RNX maps preserve the separable attention factorization.
  This gives O(Fd) aggregation for width-d feature maps.

## Empirical observations

- A fixed-split five-seed Diamonds study had low initialization
  variability; studies changing data partitions had greater variability.
- The original unnormalized R formulation produced severe extrapolation
  errors on some Ames observations. RN reduced score magnitudes and
  improved the historical mean test R², but extreme predictions persisted.
- Increasing training budget and selecting minibatch size on validation
  substantially improved RN's Concrete results. The study does not
  separately identify the effect of epoch count and batch size.
- On Ames, validation-selected numeric asinh preprocessing reduced mean
  confirmation RMSE from $27,411.838 to $24,550.434 across three seeds.
  The paired effect was not consistently favorable.
- The Ames RNX confirmation showed a modest average improvement from
  a two-unit numerical value mapping, with little additional gain from
  combining it with Q/K tanh. The three-split uncertainty was large.
- In Benchmark-15 (15 datasets, three seeds each), RN exceeded the
  matched MLP's mean test R² on eight datasets. Tree ensembles had better
  cross-dataset rank averages and were usually stronger.
- White Wine showed a repeatable XGBoost advantage over RN on seeds
  42, 43 and 44. Red Wine was much closer. The wine datasets have the
  same 11 numerical features but differ in size and data distribution.
  This comparison does not establish a mechanism for the performance gap.
- On Ames, high concentration of test squared error in a small number
  of observations remained a significant diagnostic concern.

## Claims not established by these experiments

- General superiority to XGBoost, CatBoost or carefully tuned MLPs.
- Output boundedness from normalized query/key scores alone.
- A causal link between any individual feature or learned gate and a
  prediction error, inferred solely from algebraic contributions.
- Performance of an independently retrained attention-free architecture
  from post-hoc branch ablation.
- Improved robustness or generalization from RNX's value mapping
  outside the examined Ames confirmation partitions.
- Inference latency or energy superiority from the O(Fd) aggregation
  complexity expression alone.

## Evaluation limitations

Datasets were repeatedly inspected during model development.
Model optimization budgets and tree tuning differ. Existing seeds
often vary both partition and initialization simultaneously; the new
width-sweep protocol separates these variables. Time-correlated
datasets in Benchmark-15 use random rather than forward-in-time splits.
Duplicate feature vectors may cross random train/test partitions in
wine datasets. These design choices limit external generalization claims.
