# Architecture and versioned formulations

Reference implementations: [VANTA-R](../VANTA_R/model.py), [VANTA-G](../VANTA_G/model.py), [VANTA-RN](../VANTA_RN/model.py). These models do scalar **tabular regression**; they are not sequence Transformers.

## One token and one value per feature

Let $F$ be the number of original features and $d$ the query/key width. A numerical column is imputed and standardized using **training data only**, then represented as token $t_i=x_i\mathbf{1}_d$ and scalar value $v_i=x_i$. A categorical column is represented using a trainable $d$-vector token embedding and a separate trainable **scalar value embedding**; ID zero is reserved for unknown categories.

Feature-specific componentwise affine projections:

$$
q_i=t_i\odot W_{q,i}+b_{q,i}, \qquad
k_i=t_i\odot W_{k,i}+b_{k,i}.
$$

Thus $Q,K\in\mathbb R^{F\times d}$ and $v\in\mathbb R^F$ for one observation. Scores are generally asymmetric and signed. The readout is one linear projection, without softmax, learned dense value projection or Transformer FFN.

## VANTA-R — original scalar-gated residual attention

$$
A_R=\frac{QK^\top}{\sqrt d}, \qquad
h_R=A_Rv=\frac{Q(K^\top v)}{\sqrt d}, \qquad
\hat y_R=w^\top(v+\gamma h_R)+b.
$$

$\gamma$ is one trainable scalar. The interaction matrix rank is at most $d$; aggregation is $O(Fd)$ when computed associatively. This says nothing about the cost of dataset preprocessing, projection creation or training. Original Q/K and values can grow without an operator-level bound.

## VANTA-G — feature-wise residual gates

$$
\hat y_G=w^\top(v+\boldsymbol\gamma\odot h_R)+b,
\qquad \boldsymbol\gamma\in\mathbb R^F.
$$

The gates differ in optimization parameterization. With otherwise unconstrained Q rows, a per-row scalar gate can be absorbed into the query parameters; do not infer greater theoretical expressivity merely from extra gates.

## VANTA-RN v0.1 — normalized Q/K

Smooth L2 normalization follows the original affine projection:

$$
\tilde q_i=\frac{q_i}{\sqrt{1+\|q_i\|_2^2}}, \qquad
\tilde k_i=\frac{k_i}{\sqrt{1+\|k_i\|_2^2}}.
$$

$$
A_{RN}=\frac{\tilde Q\tilde K^\top}{\sqrt F}, \qquad
h_{RN}=\frac{\tilde Q(\tilde K^\top v)}{\sqrt F}, \qquad
\boxed{\hat y_{RN}=w^\top(v+\gamma h_{RN})+b.}
$$

RN adds no trainable parameters to R; its source implementation subclasses R. By Cauchy–Schwarz, each normalized attention score satisfies $|A_{ij}|<1/\sqrt F$ for finite vectors. The score can be negative; the matrix remains asymmetric and has rank at most $d$. Associative aggregation still costs $O(Fd)$.

**Crucial limit:** bounded individual scores do **not** bound $h$, $w^\top v$, $\gamma$ or the final prediction. Nor is dividing by $\sqrt F$ the same as Transformer scaled dot-product attention's $\sqrt d$ factor: one scales aggregation count, the other dot-product dimension.

The smooth norm is already a nonlinear transformation of input vectors; the **score kernel** remains the identity $g(u)=u$. Nonlinear score kernels are a future experiment.

## Versioning rule

Mathematical changes (normalization, scaling, nonlinearity, feature-value definition, gate shape) require a new model version and distinct checkpoint metadata. Changes in optimizer, dataset, split or training budget are **experiment configurations**, not silent architecture revisions.

See [mathematical notes](mathematical_notes.md) for complexity and kernel theory and [experiments](experiment_registry.md) for observed performance.


## VANTA-RNX — implemented separable feature-map variant

The experimental RNX model is implemented in
[VANTA_RNX/model.py](../VANTA_RNX/model.py). It preserves the RN
normalization and introduces independently configurable query/key
vector maps and numeric scalar value maps:

$$
h = \Phi(\tilde Q)\left(\Psi(\tilde K)^\top u\right)/\sqrt F,
\qquad
\hat y=w^\top(v+\gamma h)+b.
$$

The direct residual remains $v$; only the numeric values entering
attention change to $u=\psi_V(v)$. The implemented map configurations
are identity, componentwise tanh, and identity-initialized learned
cubic mappings for Q/K; identity, asinh, identity-initialized learned
asinh and two-hidden-unit GELU for attention values. Q/K mappings
are rescaled to preserve original RN token norm. Aggregation uses
the same associative O(Fd) computation.

The completed Ames RNX experiment and its uncertainty are reported
in [experiment_registry.md](experiment_registry.md).
