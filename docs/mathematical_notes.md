# Mathematical properties and complexity

## Associativity and cost

For one observation, $Q,K\in\mathbb R^{F\times d}$ and $v\in\mathbb R^F$:

$$
(QK^\top)v=Q(K^\top v).
$$

Computing the right side costs $O(Fd)$ arithmetic and avoids constructing an $F\times F$ matrix. An explicit attention diagnostic incurs $O(F^2d)$ arithmetic. This is a known **linear-attention factorization**, not an original algebraic invention. Inference and training also include tokenization, feature projections and optimizer costs.

## The normalized-score bound

Define $N(q)=q/\sqrt{1+\|q\|_2^2}$. Then

$$
\|N(q)\|_2^2=\frac{\|q\|_2^2}{1+\|q\|_2^2}<1
$$

and Cauchy–Schwarz yields

$$
|N(q)^\top N(k)|\le\|N(q)\|_2\|N(k)\|_2<1.
$$

Consequently $|A_{ij}|<1/\sqrt F$. By the triangle inequality:

$$
|h_i|\le \frac{1}{\sqrt F}\sum_j|v_j|,
\qquad
|\gamma w^\top Av|\le\frac{|\gamma|}{\sqrt F}
\left(\sum_i|w_i|\right)\left(\sum_j|v_j|\right).
$$

**There is no fixed finite prediction bound** as long as the values, readout and gate are unconstrained.

The normalization Jacobian is

$$
J_N(q)=\frac{I}{\sqrt{1+\|q\|^2}}
-\frac{qq^\top}{(1+\|q\|^2)^{3/2}}.
$$

Large norms progressively suppress radial sensitivity. Gamma/readout compensation is an *unproven* optimization hypothesis, not a diagnosis of exploding weights.

## Complexity of nonlinear separable kernels

Nonlinearity does not, by itself, imply quadratic dependence on $F$. The important property is an exact or approximate separable feature map

$$
K(q,k)=\phi(q)^\top\psi(k), \qquad \phi,\psi\in\mathbb R^m.
$$

Then

$$
h_i=\sum_j K(q_i,k_j)v_j
=\phi(q_i)^\top\left(\sum_j\psi(k_j)v_j\right),
$$

so aggregation costs $O(Fm)$ plus computing $\phi$ and $\psi$.

1. **Exact $O(Fd)$ with nonlinear vector feature maps:** for example, $K(q,k)=\tanh(q)^\top\tanh(k)$ with componentwise tanh. This is genuinely nonlinear in $q$ and $k$, can preserve signed scores, and has $m=d$. **It is not a scalar nonlinearity $g(q^\top k)$.**
2. **Exact nonlinear scalar dot-product kernel:** $g(u)=u+\alpha u^3$ factors using linear and cubic tensor features with map width $m=d+\binom{d+2}{3}$. At $d=24$, $m=2,624$. It stays linear in $F$ but not generally in $d$; tensor feature construction has nontrivial cost.
3. **Approximate feature maps:** random or trained finite maps may approximate richer kernels in $O(Fm)$ with an approximation-accuracy tradeoff. Any efficiency claim must report $m$, errors, map-construction cost and hardware.
4. **Direct arbitrary elementwise $g$:** applying $g$ to every entry of an explicit $F\times F$ matrix generally forfeits simple associative $O(Fd)$ evaluation.

Scalar dot-product nonlinearities and separable nonlinear vector maps define different function families. Exact finite-dimensional factorizations determine whether associative aggregation remains available. With existing RN L2 normalization, additional tanh maps introduce another saturation mechanism, so compare against identically trained identity-kernel RN.

## Pairwise versus three-way features

Each $q_i$ depends on one original column, each $k_j$ on another and $v_j$ on column $j$. A single layer with scalar score $g(q_i^\top k_j)$ gives a sum of functions of at most pairs $(x_i,x_j)$, even if individual terms have high polynomial degree (e.g. $x_ix_j^2$). This does not automatically represent a genuine three-feature product $x_1x_2x_3$. This limitation may matter for nonlinear benchmarks; it is not established as the cause of current accuracy gaps.

## Interpretation cautions

- $\gamma$ alone does not measure attention importance: the actual contribution is $\gamma w^\top Av$.
- Source features supply values $v_j$; receiving features determine readout weight $w_i$. Both perspectives are useful, neither is causal attribution.
- Post-hoc removal of a branch from jointly trained parameters does not measure the accuracy of a separately retrained ablated model.
- Test-set gate sweeps remain exploratory and cannot be used to choose the gate.
