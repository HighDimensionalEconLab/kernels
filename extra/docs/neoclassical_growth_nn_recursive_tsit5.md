# Adaptive Tsit5 collocation for the recursive growth NN

The canonical two-state solver in `neoclassical_growth_nn_recursive.py` can
train on either exact aligned knots from a fixed-step rollout or the accepted
internal states chosen by Tsit5. The economic state remains the separate pair

\[
  (\log z, \log k), \qquad
  \frac{d\log z}{dt}=g, \qquad
  \frac{d\log k}{dt}=\frac{i(z,k)}{k}-\delta.
\]

Only the policy representation uses the homothetic coordinate
`log(k) - log(z)`. Initial TFP, initial capital, the growth rate, and all other
structural quantities remain in the differentiable parameter pytree `p`.

At every accepted state, the training equation is the full two-state Euler
residual

\[
  r(z,k;\theta)
  = \nabla_{(\log z,\log k)}\log c(z,k;\theta)^\top
      \begin{bmatrix}g\\i(z,k;\theta)/k-\delta\end{bmatrix}
    - \frac{\operatorname{MPK}(z,k)-\delta-\widehat\rho}{\sigma}.
\]

For multiple initial conditions, independent solves are vmapped. A solve with
attempt budget \(B\) returns static storage of shape `(M, B + 1, 2)` and an
accepted-prefix mask. Padding contributes exact zeros, and the stacked
residual is divided by the square root of the total number of active states.
If any trajectory exhausts its budget before the training horizon, its
residual block becomes nonfinite so the optimizer cannot silently fit a path
prefix.

## Automatic differentiation contract

The integral controller's accept/reject tests and proposed time steps are
discrete decisions. Tinydiffeq applies `stop_gradient` to those controller
quantities, so the accepted mesh is locally frozen. Accepted states still
differentiate through every Tsit5 stage and support JVP, VJP, and the
reverse-over-forward derivatives used by geodesic LM. Derivatives therefore
describe a fixed local controller branch and can jump when a perturbation
changes that branch.

The fixed and adaptive fits use the same fixed-RK4 held-out, TVC, and settling
rollouts. This keeps reported policy errors separate from integrator choice.

## Horizons and calibrated defaults

The production comparison identifies the policy through
`train_T=100` and uses 65 fixed residual dates. Fixed RK4 takes 128 internal
steps, so its internal step is 0.78125 and each residual date lands on every
second internal node. The TVC check uses 256 steps through time 200. The
standard held-out trajectories end at time 40: for the baseline calibration,
this is more than 12 detrended transition half-lives, while the longer training
horizon tests the balanced-growth tail and Tsit5's large-step behavior.

The adaptive defaults are an attempt budget of 80, `dt_0=1`,
`rtol=3e-9`, `atol=3e-11`, and controller growth cap 3, with no maximum-step
ceiling. The budget is deliberately larger than the desired state count so
rejected attempts do not cause spurious path failure.

Those tolerances are resolved defaults for float64. Float32 instead resolves
to `rtol=1e-4`, `atol=1e-6`; forcing the float64 tolerances below float32
resolution can exhaust the attempt budget without advancing the path. Explicit
CLI values override the dtype defaults.

The mesh-only tolerance screen below used the seed-0 fixed-trained policy at
`T=100` and a larger screening capacity. Every row reached the horizon; the
selected and looser rows fit within the retained 80-attempt budget, while the
two tightest rows exhaust or exceed its capacity. Initial steps 0.25, 0.5, and
1.0 gave effectively the same counts, so `dt_0=1` avoids unnecessary startup
work.

| Relative/absolute tolerance | Accepted states | Status | Decision |
|---|---:|---|---|
| 1e-6 / 1e-8 | 26 | reached T=100 | rejected: too few states |
| 3e-7 / 3e-9 | 31 | reached T=100 | rejected: too few states |
| 1e-7 / 1e-9 | 37 | reached T=100 | rejected: too few states |
| 3e-8 / 3e-10 | 45 | reached T=100 | rejected: too few states |
| 1e-8 / 1e-10 | 54 | reached T=100 | rejected: below point target |
| 3e-9 / 3e-11 | 67 | reached T=100 | selected: matches 65-date design |
| 1e-9 / 1e-11 | 81 | reached T=100 | rejected: unnecessary work |
| 3e-10 / 3e-12 | 101 | capacity screen only | rejected: exceeds default capacity |

The retained three-seed comparison used one training path, 50 deterministic
held-out draws with log-initial-condition variance 0.0025, `g=0.02`, and
`T_test=40`. All runs converged and passed the collocation-horizon, positive
test-path, TVC, train-Euler, and benchmark-error gates.

| Method | States | Median LM steps | Warm fit (s) | Median capital error | Median consumption error | Test Euler mean abs. |
|---|---:|---:|---:|---:|---:|---:|
| fixed RK4 requested dates (historical interpolated-save path) | 65 | 41 | 1.058 | 1.21e-6 | 7.31e-7 | 1.57e-6 |
| Tsit5 accepted states | 67 | 32 | 1.833 | 8.28e-7 | 5.53e-7 | 6.62e-7 |

The fixed-row timing predates Tinydiffeq's exact-knot save path. Those residual
dates already coincided with every second RK4 knot, but the old implementation
still passed through its interpolated output machinery. Treat the fixed timing
and the adaptive-to-fixed speed ratio as historical until this comparison is
regenerated with `SaveAt(exact=True)`.

Tsit5's accepted step sizes ranged from about 0.158 to 13.87, with median
0.657, so this is genuinely adaptive rather than a disguised fixed grid. It is
about 1.73 times slower warm. First-invocation timing is strongly
persistent-cache dependent and is not interpreted as a clean compilation
benchmark. Tsit5 is more accurate in this held-out comparison: median capital
and consumption errors fall by factors of about 1.46 and 1.32, and the Euler
diagnostic falls by about 2.37. Thus Tsit5 is the accuracy-oriented option;
fixed RK4 remains the speed-oriented baseline.

Run one adaptive fit with:

```bash
uv run python neoclassical_growth_nn_recursive.py \
  --g 0.02 --data_set.ode_solver tsit5
```

Run the fixed-versus-adaptive saved comparison with:

```bash
uv run python -m \
  extra.replication.neoclassical_growth_nn_recursive_tsit5_comparison
```

The driver saves result, accepted training mesh, held-out trajectory, policy,
and per-trajectory summary Parquet files plus the full hyperparameter JSON.
