# Optional examples

This directory holds useful variations that are not part of the main paper
replication pipeline.

## Kernel variations

The asset-pricing and neoclassical-growth UNO implementations are main examples
with drivers under `replication/`. The human-capital and advertising UNO
implementations remain here as optional comparisons.

`multicountry_growth_kernel_dae.py` is the fully unreduced one-good
multicountry DAE. It keeps RKHS paths for country consumption and investment
and for the common interest rate, whereas the canonical top-level kernel and
NN use the reduced formulation.

`neoclassical_growth_kernel_recursive_adaptive.py` is the adaptive state-space
kernel experiment. It moves its RKHS centers to the accepted integrator states
and transfers the represented policy when that basis changes.

## NN variations

- `neoclassical_growth_nn_integrated.py`: an MLP approximates the time
  derivatives and integrates them into paths.
- `neoclassical_growth_nn_rff.py`: the same integrated-path construction with
  fixed random Fourier features and a trainable linear head.
- `neoclassical_growth_nn_recursive_adaptive.py`: the adaptive accepted-step
  recursive policy using Tsit5 or Rodas5P. The fixed-step RK4 version is the
  canonical top-level baseline.
- `random_fourier_features.py`: the RFF feature implementation shared by that
  example and its tests.

The MLP and RFF cases are separate modules rather than modes of one core file.

## Comparisons and sweeps

`extra/replication/kernel_vs_nn.py` and
`extra/replication/neoclassical_growth_method_comparison.py` preserve optional
comparison experiments. They are intentionally excluded from
`generate_all_results.sh`.

HPO material is under `extra/sweeps/`. Run optional modules from the repository
root, for example:

```bash
uv run python -m extra.replication.kernel_vs_nn --output_dir .results-extra
uv run python -m extra.multicountry_growth_kernel_dae --a_0 '[-0.5, 0.5]'
uv run python -m extra.replication.neoclassical_growth_method_comparison \
  --output_dir .results-extra
uv run python -m extra.replication.neoclassical_growth_nn_recursive_adaptive \
  --output_dir .results-extra
uv run python -m extra.replication.human_capital_kernel_uno \
  --output_dir .results-extra
```

Tests:

```bash
uv run python -m unittest discover -s extra/tests -v
```
