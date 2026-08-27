import json
import pathlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct


@struct.dataclass
class NNXResidualArgs:
    batch: dict
    graphdef: nnx.GraphDef
    nondiff: nnx.State


# U(0) = U'(0) = 1, so a near-zero head starts at its declared output scale
# without shrinking the local Jacobian by softplus'(0).
def unit_softplus(raw):
    log_two = jnp.log(jnp.asarray(2.0, dtype=raw.dtype))
    return jax.nn.softplus(2.0 * log_two * raw) / log_two


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "softplus": jax.nn.softplus,
    "unit_softplus": unit_softplus,
    "identity": nnx.identity,
}


# unit_softplus on the head outputs flagged True; signed outputs pass through
# linearly. One flag per output.
@dataclass(frozen=True)  # value-hashable: shares DrawNNXModule/jit compilations
class UnitSoftplusWhere:
    positive: tuple[bool, ...]

    def __call__(self, raw):
        return jnp.where(jnp.asarray(self.positive), unit_softplus(raw), raw)


# MLP map t -> (..., out_features), batch-aware through the trailing feature
# axis: scalar t gives (out_features,), a t grid of shape (...) gives
# (..., out_features), so value_and_grad_fwd(model)(t_grid) yields per-sample
# derivatives without a vmap.
class TimeMLP(nnx.Module):
    def __init__(
        self,
        out_features,
        *,
        width,
        depth,
        activation,
        use_bias=True,
        use_output_bias=True,
        final_activation=None,
        head_kernel_std=1e-3,
        input_scale,
        output_scale=None,
        g=None,
        dtype,
        rngs,
    ):
        self.activation = nnx.static(activation)
        self.final_activation = nnx.static(final_activation)
        self.input_scale = nnx.static(input_scale)
        # Non-Param Variable: output scales are data (no recompilation when
        # they change) outside the differentiated Param state.
        self.output_scale = (
            None
            if output_scale is None
            else nnx.Variable(jnp.asarray(output_scale, dtype=dtype))
        )
        # Trend factor e^{g t}; None omits it from the graph entirely. Always
        # a Param when present -- learned vs frozen is the solve filter's call.
        self.g = None if g is None else nnx.Param(jnp.asarray(g, dtype=dtype))
        layers, in_features = [], 1
        for _ in range(depth):
            layers.append(
                nnx.Linear(
                    in_features,
                    width,
                    use_bias=use_bias,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
            in_features = width
        self.layers = nnx.List(layers)
        # Near-zero head: drawn paths start at the declared output scales;
        # initial conditions are imposed as residual rows.
        self.head = nnx.Linear(
            width,
            out_features,
            use_bias=use_output_bias,
            kernel_init=nnx.initializers.normal(head_kernel_std),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, t):
        t = jnp.asarray(t)
        h = (t / self.input_scale)[..., None]
        for layer in self.layers:
            h = self.activation(layer(h))
        out = self.head(h)
        if self.final_activation is not None:
            out = self.final_activation(out)
        if self.output_scale is not None:
            out = self.output_scale[...] * out
        if self.g is not None:
            out = jnp.exp(self.g[...] * t)[..., None] * out
        return out


# One .npz per trained module: the nnx.Param block and the non-differentiated
# Variable block as "params/..." and "nondiff/..." flat keys (nnx's supported
# State.to_pure_dict / replace_by_pure_dict conversion, list indices numeric),
# plus a JSON __meta__ block. RNG state is never written: a reload rebuilds
# the structural template from meta, takes the Param and Variable arrays from
# the file, and keeps the template's RNG state.
def save_model(path, module, meta):
    def flatten(namespace, node, out):
        if isinstance(node, dict):
            for key, value in node.items():
                flatten((*namespace, str(key)), value, out)
        else:
            out["/".join(namespace)] = np.asarray(node)
        return out

    _, params, _, nondiff = nnx.split(module, nnx.Param, nnx.RngState, ...)
    arrays = flatten(("params",), params.to_pure_dict(), {})
    flatten(("nondiff",), nondiff.to_pure_dict(), arrays)
    meta = {**meta, "dtype": str(next(iter(arrays.values())).dtype)}
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, __meta__=np.asarray(json.dumps(meta)), **arrays)


def load_model(path, make_module):
    data = np.load(path)
    meta = json.loads(str(data["__meta__"]))
    dtype = jnp.dtype(meta["dtype"])
    blocks = {"params": {}, "nondiff": {}}
    for name in data.files:
        if name == "__meta__":
            continue
        namespace, _, rest = name.partition("/")
        if namespace not in blocks:
            raise ValueError(
                f"{path}: unknown npz key namespace {namespace!r}; expected "
                "params/ or nondiff/ (an artifact from the flat pre-namespace "
                "format must be regenerated)"
            )
        node = blocks[namespace]
        *parents, leaf = rest.split("/")
        for part in parents:
            node = node.setdefault(int(part) if part.isdigit() else part, {})
        array = data[name]
        node[int(leaf) if leaf.isdigit() else leaf] = jnp.asarray(
            array, dtype=dtype if np.issubdtype(array.dtype, np.inexact) else None
        )
    graphdef, params, rngs, nondiff = nnx.split(
        make_module(meta, dtype), nnx.Param, nnx.RngState, ...
    )

    def leaf_paths(node, prefix=()):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from leaf_paths(value, (*prefix, str(key)))
        else:
            yield "/".join(prefix)

    # A missing Param subtree in replace_by_pure_dict is a silent no-op that
    # would keep the template's random initialization for those leaves.
    missing = set(leaf_paths(params.to_pure_dict())) - set(leaf_paths(blocks["params"]))
    if missing:
        raise ValueError(f"{path} is missing params arrays: {sorted(missing)}")
    params.replace_by_pure_dict(blocks["params"])
    if blocks["nondiff"]:
        nondiff.replace_by_pure_dict(blocks["nondiff"])
    return nnx.merge(graphdef, params, rngs, nondiff), meta
