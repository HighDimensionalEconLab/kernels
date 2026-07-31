# Fixed-center recursive growth kernel

`extra/neoclassical_growth_kernel_recursive_fixed.py` is the nonadaptive
kernel counterpart to `neoclassical_growth_nn_recursive.py`. Both solvers keep
technology and capital as separate economic states. The kernel uses benchmark
solutions only to choose a fixed set of training states; those states never
move during optimization.

## Economic problem

Production is homogeneous,

\[
Y(z,k)=z\left(\frac{k}{z}\right)^a,
\]

and the state equations are kept in levels:

\[
\dot z=gz,
\qquad
\dot k=i_\theta(z,k)-\delta k,
\qquad
c_\theta(z,k)=Y(z,k)-i_\theta(z,k).
\]

For CRRA coefficient \(\sigma\), the Euler residual is

\[
R_\theta(z,k)
=\frac{d}{dt}\log c_\theta(z,k)
-\frac{Y_k(z,k)-\delta-\rho}{\sigma}.
\]

The implementation obtains the total derivative with a two-state JVP along

\[
(\dot{\log z},\dot{\log k})
=\left(g,\frac{i_\theta(z,k)}{k}-\delta\right).
\]

Thus the explicit TFP-growth contribution is included in the residual; the
economic problem is not replaced by a detrended one-state system. The only
exception is the independent `solve_bvp` benchmark, which solves an equivalent
normalized boundary-value problem internally and reconstructs the level paths
\(z(t),k(t),c(t)\) before returning them. Validation enforces both the requested
\(g<\rho\) restriction and the effective-discount condition
\(\rho+(\sigma-1)g>0\).

## Shared homothetic policy representation

Homothetic structure is used only in the approximation architecture. Both the
NN and kernel receive

\[
q(z,k)=\log k-\log z
\]

as their learned-policy input and return an investment share:

\[
i_\theta(z,k)=Y(z,k)\,\operatorname{sigmoid}(h_\theta(q)).
\]

This guarantees positive investment and consumption without introducing an
initial-state reference level. For the fixed kernel,

\[
h_\theta(q)=i_{\mathrm{const}}
+\sum_{j=1}^{n}\alpha_j K(q,q_j),
\qquad
K(q,q_j)=\sigma_K^2
\exp\left[-\frac{(q-q_j)^2}{2\ell^2}\right].
\]

The learned `i_const` is the unpenalized kernel mean. Its initialization and
the NN output bias both imply an investment share of 0.2; this is only an
initial parameter value, not an anchor or a reference-level normalization.

## Fixed exact-path distribution

For every supplied training pair \((z_{0m},k_{0m})\), the code evaluates the
benchmark at

\[
t_j=\frac{j}{P-1}T_{\mathrm{train}},
\qquad j=0,\ldots,P-1.
\]

It retains the separate benchmark states \((z_{mj},k_{mj})\) for the economic
residual and places kernel centers only on
\(q_{mj}=\log k_{mj}-\log z_{mj}\). The defaults \(P=65\) and
\(T_{\mathrm{train}}=40\) exactly match the recursive NN's 65 residual dates,
with spacing 0.625. The NN's fixed RK4 rollout uses 32 internal steps over the
same horizon. With \(M\) training trajectories, the residual contains all
\(MP\) trajectory-date observations, flattened and scaled by
\(1/\sqrt{MP}\).

The centers are constructed once and are never adapted to optimizer iterates.
Benchmark investment and consumption do not enter the objective; they are
retained only for ex-post diagnostics. `collocation_source="exact_path"` is
the explicit toggle boundary for a later alternative fixed-state experiment.

Uniform time sampling puts many centers near balanced growth. The RBF Gram
matrix can therefore be nearly rank deficient. Policy evaluation uses the
unshifted kernel, while the optimizer metric factors \(K+\epsilon I\). Ridge
LM solves

\[
\min_{\alpha,i_{\mathrm{const}}}
\frac{1}{MP}\sum_{m=1}^{M}\sum_{j=1}^{P}
R_\theta(z_{mj},k_{mj})^2
+\lambda\alpha^\mathsf{T}(K+\epsilon I)\alpha.
\]

Only the coefficient block is penalized. The learned constant is a trailing
free block. Reports retain raw and shifted Gram diagnostics, train/test Euler
errors, a scale-invariant transversality certificate relative to its initial
log value, optimizer time, total fitting time, and held-out benchmark path
errors. This distinction matters: a short lengthscale can fit the Euler
equation on the fixed states while generating the wrong transition away from
those states, so local convergence alone is not acceptance.

The benchmark states and Gram matrix are a frozen experimental design. JAX AD
therefore differentiates the residual and fitted coefficients conditional on
that design. A total derivative that regenerates the exact-path distribution
when structural parameters or initial conditions change would require a
differentiable (or custom-JVP) benchmark-state generator; the reported metadata
labels the present derivative semantics explicitly.

## Training, testing, and saved output

Training initial conditions may be scalars or equal-length lists. Both the NN
and kernel use the same deterministic held-out generator. A separate
`data_seed` draws independent centered perturbations of \(\log z_0\) and
\(\log k_0\), each with the requested exact population variance. The network
seed changes only NN parameters, so all NN seeds and the deterministic kernel
face identical test trajectories.

The test rollout uses the configured fixed-step solver, `test_T`, and `dt`;
`test_T` must be an integer multiple of `dt`.
Long-form Parquet tables retain every `(trajectory, t)` row, including the
realized initial pair, separate state paths, policy quantities, benchmark
paths, and signed and absolute relative errors. JSON files contain settings
and aggregate accuracy/timing statistics.

Run the kernel alone with

```bash
uv run python -m extra.neoclassical_growth_kernel_recursive_fixed
```

Run the apples-to-apples saved comparison with

```bash
uv run python -m extra.replication.neoclassical_growth_recursive_comparison \
  --output_dir .results_extra
```

The comparison performs one discarded NN warm-up, retains warm NN timings,
and compares them with the kernel's five-run warm median. Its common validation
gate requires solver convergence, the same train-Euler tolerance, a finite
positive test path, transversality, and the same benchmark-error tolerances for
both methods. The method table is stored in Parquet; the comparison JSON is a
single wide row with method-prefixed values and speed/error ratios.

Run the focused tests with

```bash
uv run python -m unittest \
  extra.tests.test_neoclassical_growth_balanced \
  extra.tests.test_recursive_growth_comparison \
  extra.tests.test_recursive_kernel_fixed -v
```
