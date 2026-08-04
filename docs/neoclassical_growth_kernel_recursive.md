# Fixed-center recursive growth kernel

`neoclassical_growth_kernel_recursive.py` is the nonadaptive
kernel counterpart to `neoclassical_growth_nn_recursive.py`. Both solvers keep
technology and capital as separate economic states. The kernel uses benchmark
solutions only to choose a fixed set of training states; those states never
move during optimization.

## Economic problem

Production is Cobb-Douglas,

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
\(z(t),k(t),c(t)\) before returning them. Interface validation rejects only
ill-posed structural parameters — \(g<\rho\), the effective-discount condition
\(\rho+(\sigma-1)g>0\), and positive effective depreciation \(\delta+g>0\) —
and never computes steady states or balanced-growth quantities.

## Policy representation

Both the NN and kernel receive

\[
q(z,k)=\log k-\log z
\]

as their learned-policy input — the only place homothetic structure enters
either solver — and share one output map:

\[
i_\theta(z,k)=\delta\,k\,U(h_\theta(q)),
\qquad U(0)=U'(0)=1,
\]

with \(U\) the unit-anchored softplus. Because
\(\dot k/k=\delta\,(U(h_\theta)-1)\), a zero head holds capital exactly at
rest in levels (under \(g>0\) detrended \(\kappa=k/z\) still falls at rate
\(g\) at that anchor), investment is positive, and consumption
\(c_\theta=Y-i_\theta\) is unclamped: an infeasible iterate produces a loud
NaN residual that the optimizer rejects rather than a silent projection. Only
\(\delta\) and the state enter the map — no production function, growth rate,
or steady state.

The two solvers differ only in the representation of \(h_\theta\). The NN uses
an MLP whose head initializes with zero bias and small weights; the fixed
kernel uses

\[
h_\theta(q)=h_{\mathrm{const}}
+\sum_{j=1}^{n}\alpha_j K(q,q_j),
\qquad
K(q,q_j)=\sigma_K^2
\exp\left[-\frac{(q-q_j)^2}{2\ell^2}\right],
\]

with \(\theta_0=0\), so both start at the capital-at-rest anchor. The
unpenalized constant \(h_{\mathrm{const}}\) can be removed entirely with
`StateKernelSettings.free_constant=False`, leaving the pure RKHS expansion.

## Fixed exact-path distribution

For every drawn training pair \((z_{0m},k_{0m})\), the code evaluates the
benchmark at

\[
t_j=\frac{j}{P-1}T_{\mathrm{train}},
\qquad j=0,\ldots,P-1.
\]

It retains the separate benchmark states \((z_{mj},k_{mj})\) for the economic
residual and places kernel centers only on
\(q_{mj}=\log k_{mj}-\log z_{mj}\). The defaults \(P=65\) and
\(T_{\mathrm{train}}=100\) exactly match the recursive NN's 65 residual dates,
with spacing 1.5625. The NN's fixed RK4 rollout uses 64 internal steps over
the same horizon. With \(M\) training trajectories, the residual contains all
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
\min_{\alpha,h_{\mathrm{const}}}
\frac{1}{MP}\sum_{m=1}^{M}\sum_{j=1}^{P}
R_\theta(z_{mj},k_{mj})^2
+\lambda\alpha^\mathsf{T}(K+\epsilon I)\alpha.
\]

Only the coefficient block is penalized; the learned constant, when enabled,
is a trailing free block.

Acceptance is solver convergence alone. Everything else — train and held-out
maximum Euler residuals, finiteness and positivity of the test paths, the
scale-invariant transversality certificate relative to its initial log value,
and the benchmark path errors — is computed identically in both solvers and
reported as diagnostics, never used to reject a solution. The point is to
measure the generalization error of a converged algorithm: a short
lengthscale that fits the Euler equation on the fixed states while generating
the wrong transition away from them shows up as a large held-out Euler
residual and benchmark error in the reports, not as a hidden rejection.

The benchmark states and Gram matrix are a frozen experimental design. JAX AD
therefore differentiates the residual and fitted coefficients conditional on
that design. A total derivative that regenerates the exact-path distribution
when structural parameters or initial conditions change would require a
differentiable (or custom-JVP) benchmark-state generator; the reported metadata
labels the present derivative semantics explicitly.

## Training, testing, and saved output

Initial conditions are drawn, not supplied as lists. A single `data_seed`
splits into independent train and test streams; each stream draws
`num_train_trajectories` (or `num_test_trajectories`) initial pairs centered
on the scalar \((z_0,k_0)\), with independent centered perturbations of
\(\log z_0\) and \(\log k_0\) carrying exactly the requested population
variances (`train_log_z_variance`, `train_log_k_variance`,
`test_log_z_variance`, `test_log_k_variance`). Zero variances reproduce the
deterministic base point without consuming randomness, so the single-IC
configuration is the degenerate case of the same machinery. Setting
`test_z_0`/`test_k_0` recenters only the held-out draws for
out-of-distribution testing; unset, they inherit the training center.
The network seed
changes only NN parameters, so all NN seeds and the deterministic kernel face
identical train and test initial conditions.

The test rollout uses the configured fixed-step solver, `test_T`, and `dt`;
`test_T` must be an integer multiple of `dt`.
Long-form Parquet tables retain every `(trajectory, t)` row, including the
realized initial pair, separate state paths, policy quantities, benchmark
paths, and signed and absolute relative errors. JSON files contain settings
and aggregate accuracy/timing statistics.

Run the kernel alone with

```bash
uv run python neoclassical_growth_kernel_recursive.py
```

Run the apples-to-apples saved comparison with

```bash
uv run python -m extra.replication.neoclassical_growth_recursive_comparison \
  --output_dir .results_extra
```

The comparison performs one discarded NN warm-up, retains warm NN timings,
and compares them with the kernel's five-run warm median. Its common gate is
solver convergence (plus a well-formed collocation path); Euler residuals,
path checks, transversality, and benchmark errors are carried as reported
columns with one identical meaning for both methods.

Run the focused tests with

```bash
uv run python -m unittest \
  extra.tests.test_neoclassical_growth_balanced \
  extra.tests.test_recursive_growth_comparison \
  extra.tests.test_recursive_kernel -v
```
