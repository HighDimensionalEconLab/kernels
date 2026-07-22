# Multicountry trade growth

This subproject contains the specialized full trade-growth experiments. The
generic one-good `multicountry_growth` example remains at the repository root.

## Layout

- `model.py`: static trade equilibrium and dynamic DAE shell.
- `kernel.py`: joint kernel DAE solver.
- `nn.py`: sequential NN solver for the same equations.
- `comparative_statics.py`: implicit-AD comparative statics.
- `preconditioners.py` and `whitened.py`: structured iterative and whitened
  linear-solver support.
- `scaling.py`: the iterative scaling ladder.
- `replication/`: data, comparison, comparative-statics, and figure drivers.
- `tests/`: model, implicit-AD, and preconditioner regression tests.

The default kernel solve uses the released dense Gram-Cholesky path. The
optional `gram_cg` path uses an adaptive time-block preconditioner and Krylov
recycling. Its optional endgame switches to whitened LSMR and uses the released
`ad_solver` interface for implicit differentiation.

## Running

Run modules from the repository root:

```bash
uv run python -m trade_growth.replication.kernel --output_dir .results-trade
uv run python -m trade_growth.replication.kernel_figures \
  --output_dir .results-trade
uv run python -m trade_growth.replication.nn --output_dir .results-trade
uv run python -m trade_growth.replication.comparison \
  --output_dir .results-trade
uv run python -m trade_growth.replication.comparative_statics \
  --output_dir .results-trade
uv run python -m trade_growth.scaling --country_counts '[4,10,25,50]'
```

Tests:

```bash
uv run python -m unittest discover -s trade_growth/tests -v
```
