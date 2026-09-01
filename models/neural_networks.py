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
def unit_softplus(raw, local_slope=1.0):
    """Positive softplus transform R -> (0, inf) anchored at U(0) = 1.

    U(r) = softplus(2 log(2) s r) / log(2) = log2(1 + 4^(s r)) for local
    slope s, so U(0) = 1, U'(0) = s, U''(0) = s^2 log(2), U'''(0) = 0,
    U -> 0 as r -> -inf and U' -> 2s as r -> inf. Equivalently
    2 s * torch.nn.Softplus(beta=2 s log(2)), or the parametric softplus
    alpha * log(1 + exp(beta x)) with alpha = 1/log(2), beta = 2 s log(2).
    """
    log_two = jnp.log(jnp.asarray(2.0, raw.dtype))
    return jax.nn.softplus(2.0 * log_two * local_slope * raw) / log_two


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "softplus": jax.nn.softplus,
    "unit_softplus": unit_softplus,
    "identity": nnx.identity,
}


# One local_slope entry per head output: a float applies unit_softplus with
# that slope (1.0 for the unit slope), None passes the signed output through
# linearly.
@dataclass(frozen=True)  # value-hashable: shares DrawNNXModule/jit compilations
class UnitSoftplusWhere:
    local_slope: tuple[float | None, ...]

    def __call__(self, raw):
        positive = jnp.asarray([s is not None for s in self.local_slope])
        slope = jnp.asarray(
            [1.0 if s is None else s for s in self.local_slope], raw.dtype
        )
        return jnp.where(positive, unit_softplus(raw, slope), raw)


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


def normalize_state(state, center, scale):
    return (state - center) / scale


# Per-input multiplier on the LeCun variance of the first-layer kernel rows:
# ones is plain LeCun, a smaller entry damps the draw's sensitivity to that
# input, zero starts the row at exactly zero (not frozen).
@dataclass(frozen=True)  # value-hashable: shares DrawNNXModule/jit compilations
class ScaledLecunInit:
    variance_scale: tuple[float, ...]

    def __call__(self, key, shape, dtype):
        return (
            nnx.initializers.lecun_normal()(key, shape, dtype)
            * jnp.sqrt(jnp.asarray(self.variance_scale, dtype=dtype))[:, None]
        )


# p-free MLP policy map state (..., in_features) -> (..., out_features):
# affine normalization, trunk, one Linear head per (out_width, kernel_std)
# entry (None kernel_std draws a zeros kernel) concatenated, the optional
# final_activation (unit_softplus, or UnitSoftplusWhere for mixed
# signed/positive heads), then the output_scale multiply. Positive states
# enter in logs, taken by the caller. A scalar state broadcasts against the
# (in_features,) normalization for in_features = 1. Every constructor kwarg
# is hashable so DrawNNXModule specs stay value-hashable.
class StateMLP(nnx.Module):
    def __init__(
        self,
        in_features,
        *,
        width,
        depth,
        activation,
        heads,
        final_activation=None,
        center=None,
        scale=None,
        output_scale,
        input_variance_scale=None,
        dtype,
        rngs,
    ):
        self.activation = nnx.static(activation)
        self.final_activation = nnx.static(final_activation)
        # Non-Param Variables: normalization and output scales are data
        # outside the differentiated Param state, restored from an artifact's
        # nondiff block on load_model.
        self.center = nnx.Variable(
            jnp.zeros(in_features, dtype=dtype)
            if center is None
            else jnp.asarray(center, dtype=dtype)
        )
        self.scale = nnx.Variable(
            jnp.ones(in_features, dtype=dtype)
            if scale is None
            else jnp.asarray(scale, dtype=dtype)
        )
        self.output_scale = nnx.Variable(jnp.asarray(output_scale, dtype=dtype))
        layers = [
            nnx.Linear(
                in_features,
                width,
                **(
                    {}
                    if input_variance_scale is None
                    else {"kernel_init": ScaledLecunInit(input_variance_scale)}
                ),
                dtype=dtype,
                param_dtype=dtype,
                rngs=rngs,
            )
        ]
        for _ in range(depth - 1):
            layers.append(
                nnx.Linear(width, width, dtype=dtype, param_dtype=dtype, rngs=rngs)
            )
        self.layers = nnx.List(layers)
        self.heads = nnx.List(
            [
                nnx.Linear(
                    width,
                    out_width,
                    kernel_init=(
                        nnx.initializers.zeros_init()
                        if kernel_std is None
                        else nnx.initializers.normal(kernel_std)
                    ),
                    bias_init=nnx.initializers.zeros_init(),
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
                for out_width, kernel_std in heads
            ]
        )

    def __call__(self, state):
        h = normalize_state(state, self.center[...], self.scale[...])
        for layer in self.layers:
            h = self.activation(layer(h))
        out = jnp.concatenate([head(h) for head in self.heads], axis=-1)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return self.output_scale[...] * out


# Homothetic neoclassical investment policy i = delta * k * U(h(log kappa)),
# kappa = k / z^{1/(1-a)} the detrended capital; a and delta are baked as
# non-Param Variables so the call is p-free like StateMLP's. Returns (1,).
class HomotheticPolicyMLP(nnx.Module):
    def __init__(
        self, *, width, depth, activation, head_kernel_std, a, delta, dtype, rngs
    ):
        self.activation = nnx.static(activation)
        self.a = nnx.Variable(jnp.asarray(a, dtype=dtype))
        self.delta = nnx.Variable(jnp.asarray(delta, dtype=dtype))
        layers = [nnx.Linear(1, width, dtype=dtype, param_dtype=dtype, rngs=rngs)]
        for _ in range(depth - 1):
            layers.append(
                nnx.Linear(width, width, dtype=dtype, param_dtype=dtype, rngs=rngs)
            )
        self.layers = nnx.List(layers)
        self.head = nnx.Linear(
            width,
            1,
            kernel_init=nnx.initializers.normal(head_kernel_std),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, log_state):
        log_z, log_k = log_state
        h = jnp.atleast_1d(log_k - log_z / (1.0 - self.a[...]))
        for layer in self.layers:
            h = self.activation(layer(h))
        return self.delta[...] * jnp.exp(log_k) * unit_softplus(self.head(h))


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

    # A missing subtree in replace_by_pure_dict is a silent no-op that would
    # keep the template's initialization for those leaves -- Params and
    # nondiff Variables alike.
    missing = set(leaf_paths(params.to_pure_dict())) - set(leaf_paths(blocks["params"]))
    missing |= set(leaf_paths(nondiff.to_pure_dict())) - set(
        leaf_paths(blocks["nondiff"])
    )
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    params.replace_by_pure_dict(blocks["params"])
    if blocks["nondiff"]:
        nondiff.replace_by_pure_dict(blocks["nondiff"])
    return nnx.merge(graphdef, params, rngs, nondiff), meta
