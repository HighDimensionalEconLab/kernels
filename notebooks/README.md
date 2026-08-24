# Notebooks

Self-contained annotated versions of the paper's five canonical examples: each
`*_annotated.qmd` is the source, the `.ipynb` next to it is generated and committed so
it previews on GitHub, and each `*_minimal.py` is the same code without the prose.
Nothing here imports from `models/`.

## Regenerating

From the root of this repository:

```bash
uv run quarto render notebooks                                    # all
uv run quarto render notebooks/asset_pricing_kernel_annotated.qmd # one
```

`uv run` prepends `.venv/bin` to `PATH`, so quarto discovers the venv's `python3`
kernel. Execution is cached (`execute: cache: true` in `_quarto.yml`); delete
`notebooks/.jupyter_cache/` to force re-execution.

The minimal scripts run directly:

```bash
uv run python notebooks/asset_pricing_kernel_minimal.py
```
