# VANTA
### Value-Aggregated Non-Softmax Token Attention

**VANTA** is an experimental family of lightweight neural architectures for **tabular regression**, built around learned, potentially asymmetric feature interactions.

The central idea is to represent each original feature as a token, construct separate query and key projections, and aggregate **scalar feature values** through signed dot-product interactions—without softmax, a dense value-projection matrix, or a Transformer feed-forward block.

VANTA is an ongoing research project investigating the expressivity, computational properties, numerical stability, and limitations of this interaction operator. It is **not** presented as a state-of-the-art predictor or production-ready library.

## Motivation

Tabular prediction frequently depends on interactions between features. Linear regression does not model them without explicit feature engineering, while general nonlinear models do not necessarily expose a structured interaction operator.

VANTA explores a compact alternative:

1. Preserve **one token and one scalar value per original input feature**.
2. Learn separate, feature-specific **query and key** representations.
3. Aggregate scalar values using **signed, non-softmax interactions**.
4. Preserve a **direct residual pathway** alongside the interaction pathway.
5. Produce a prediction through a **linear readout**.

The associative aggregation used by VANTA follows established ideas from **linear attention**. It is not claimed as a new discovery; connections to the kernel view of attention also inform this project.

## Mathematical formulation

Let there be $F$ features, with query/key dimension $d$. Each feature contributes a token $t_i\in\mathbb R^d$ and a scalar value $v_i$. Numerical features use their standardized scalar for the value and repeated token coordinates; categorical features use learned token and scalar-value embeddings.

Separate affine projections produce

$$
q_i=t_i\odot w_i^Q+b_i^Q,\qquad
k_i=t_i\odot w_i^K+b_i^K.
$$

Stacking query and key vectors gives $Q,K\in\mathbb R^{F\times d}$, and collecting scalar values gives $v\in\mathbb R^F$.

### VANTA-R — Scalar residual gate

The original formulation uses an unnormalized dot-product interaction matrix:

$$
A=\frac{QK^\top}{\sqrt d},\qquad
h=Av,\qquad
\boxed{\hat y=w^\top(v+\gamma h)+b.}
$$

The learnable gate $\gamma$ scales the interaction branch. In general, $A$ is asymmetric and its rank is at most $d$. Associativity gives

$$
h=\frac{Q(K^\top v)}{\sqrt d},
$$

avoiding an explicit $F\times F$ matrix. The **aggregation** requires $\mathcal O(Fd)$ arithmetic; this is a known linear-attention-style factorization.

### VANTA-G — Feature-wise residual gates

VANTA-G replaces the scalar gate with a learnable vector $\boldsymbol\gamma\in\mathbb R^F$:

$$
\boxed{\hat y=w^\top(v+\boldsymbol\gamma\odot h)+b.}
$$

With unconstrained affine query projections, these gates can be absorbed into query parameters, so the two residual parameterizations do not differ in their theoretical function families. They may still behave differently during optimization.

### VANTA-RN — Normalized query/key interactions

VANTA-RN introduces smooth L2 normalization to control the magnitude of query/key dot products:

$$
\tilde q_i=\frac{q_i}{\sqrt{1+\lVert q_i\rVert_2^2}},\qquad
\tilde k_i=\frac{k_i}{\sqrt{1+\lVert k_i\rVert_2^2}}.
$$

It then uses feature-count scaling:

$$
\tilde A=\frac{\tilde Q\tilde K^\top}{\sqrt F},\qquad
\boxed{\hat y=w^\top(v+\gamma\tilde A v)+b.}
$$

The resulting score magnitude is strictly below $1/\sqrt F$. Signed interactions, asymmetry, low-rank structure, and associative aggregation are preserved. **The final predictions are not bounded**: values, the residual gate, and the readout can still contribute large magnitudes.

The implemented variants currently retain the **identity score kernel**. Nonlinear score kernels are a future research direction rather than an existing feature.

## Repository layout

```text
VANTA/
├── VANTA_R/
│   ├── model.py               # Original residual architecture
│   ├── train.py               # Diamonds regression pipeline
│   ├── mlr_baseline.py        # Linear regression baseline
│   └── ablation/              # Gate and interaction ablations
├── VANTA_G/
│   ├── model.py               # Feature-wise gating
│   └── train.py
├── VANTA_RN/
│   ├── model.py               # Smoothly normalized Q/K
│   └── train.py               # Cross-dataset normalization experiments
├── experiments/
│   ├── baselines_ex01/        # Classical, neural, and tree comparisons
│   ├── seeds_ex02/            # Training-seed sensitivity
│   └── datasets_ex03/         # Cross-dataset regression benchmark
├── paper/
│   └── main.tex              # Preliminary research manuscript
├── pyproject.toml
└── uv.lock
```

Trained checkpoints, downloaded datasets, and result tables are generated locally and generally excluded from version control.

## Getting started

**Requirements:** Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/parv-sr/VANTA.git
cd VANTA
uv sync
```

### Train the original architecture

```bash
uv run python3 VANTA_R/train.py
```

The Diamonds training script obtains its dataset through KaggleHub if a local CSV is unavailable.

### Compare against baselines

After the original VANTA-R checkpoint has been created:

```bash
uv run python3 experiments/baselines_ex01/run_baselines.py
```

The baseline experiment includes linear regression, polynomial Ridge regression, a parameter-matched embedding MLP, XGBoost, and CatBoost.

### Cross-dataset evaluation

```bash
uv run python3 experiments/datasets_ex03/run_exp.py
```

This benchmark includes Concrete Compressive Strength, California Housing, Ames Housing, and NYC Airbnb listings. It uses train-only preprocessing, validation-based model selection, and multiple split/initialization seeds.

To compare the normalized model on the same splits, after completing the baseline experiment:

```bash
uv run python3 VANTA_RN/train.py \
  --datasets concrete california ames airbnb \
  --seeds 42 43 44
```

Some scripts depend on artifacts generated by earlier experiments. See each script's docstring for its defaults and required inputs.

### Experiment 6: width and numerical attention values

[Experiment 6](experiments/ex06/README.md) compares dataset-dependent latent widths and RN versus RNX-GELU2, both at fixed and independently selected widths. Validation chooses configurations; confirmation uses separate partition and initialization seeds. All results are aggregated into four files. The original architectures and regression heads remain unchanged.

## Research status

The project is exploratory, and current results should not be interpreted as established benchmark superiority. The experiments investigate:

- Whether learnable query/key interactions improve upon direct linear readouts.
- Whether scalar and feature-wise gates differ in optimization behavior.
- How the architecture compares with classical, nonlinear neural, and boosted-tree baselines.
- How initialization, dataset splits, batch size, and training budget influence performance.
- Whether bounded query/key scores mitigate catastrophic extrapolation on unusual observations.

Preliminary evidence indicates that learned interactions can be useful, but modern tabular baselines often remain stronger. Normalization improves numerical behavior in some settings while creating tradeoffs in optimization and representational flexibility. The datasets have also informed ongoing model development; independent evaluation is required before making strong generalization claims.

## Future directions

- **Nonlinear interaction kernels:** Investigate signed score mappings $g(\tilde q_i^\top\tilde k_j)$, expressive power, and whether useful factorization or linear-in-feature-count computation can be retained.
- **Numerical robustness:** Study extreme standardized inputs, query/key norm saturation, readout cancellation, and robust preprocessing.
- **Optimization:** Compare learning rates, batch sizes, optimizer-update budgets, latent dimension, and residual-gate dynamics.
- **Generalization:** Extend to independent real-world datasets, different training-set sizes, and more rigorous evaluation protocols.
- **Theory:** Characterize interaction order, low-rank constraints, identifiability, and connections to established kernel and feature-interaction models.

## Research documentation and reproducibility

The project keeps an evidence-based research record separate from this introductory README:

- [Architecture and model versions](docs/architecture.md)
- [Mathematical notes and nonlinear-kernel complexity](docs/mathematical_notes.md)
- [Experiment registry and measured numerical results](docs/experiment_registry.md)
- [Supported findings and rejected hypotheses](docs/findings.md)
- [Research log](docs/research_log.md)
- [Data and reproducibility protocol](docs/data_and_reproducibility.md)
- [Experiment entrypoints](experiments/README.md)

The foundational works cited below are also available as [BibTeX](paper/references.bib); citations remain here for accessibility.

## Research foundations and references

VANTA builds on previous research, particularly the associative computation associated with **linear attention** and the **kernel interpretation of attention**. These connections are acknowledged directly rather than presented as original discoveries.

1. **Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020).** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 119, 5156–5165. [Paper](https://proceedings.mlr.press/v119/katharopoulos20a.html) · [arXiv](https://arxiv.org/abs/2006.16236).
2. **Tsai, Y.-H. H., Bai, S., Yamada, M., Morency, L.-P., & Salakhutdinov, R. (2019).** *Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel.* EMNLP 2019. [arXiv](https://arxiv.org/abs/1908.11775) · [PDF](https://arxiv.org/pdf/1908.11775).

## Acknowledgements

ChatGPT and Codex were used to assist with writing and revising the code.

---

**Authored by: Parv Sharma, FLAME University**
