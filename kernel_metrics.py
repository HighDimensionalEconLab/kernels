import jax
import jax.numpy as jnp
from nlls_gram import Metric


def repeated_blockdiag_metric(
    block_metric: Metric,
    block_size: int,
    repeats: int,
    *,
    additional: tuple[Metric, int] | None = None,
) -> Metric:
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if additional is None:
        additional_metric = None
        additional_size = 0
    else:
        additional_metric, additional_size = additional
        if (
            not isinstance(additional_size, int)
            or isinstance(additional_size, bool)
            or additional_size < 1
        ):
            raise ValueError("additional size must be a positive integer")

    repeated_size = repeats * block_size
    total_size = repeated_size + additional_size

    def is_identity(metric):
        return (
            metric.solve is None
            and metric.norm is None
            and metric.inv_sqrt is None
            and metric.inv_sqrt_transpose is None
        )

    block_is_identity = is_identity(block_metric)
    additional_is_identity = additional_metric is not None and is_identity(
        additional_metric
    )

    def check_leading_size(x):
        if x.ndim not in (1, 2):
            raise ValueError("metric callbacks require a vector or matrix")
        if x.shape[0] != total_size:
            raise ValueError(
                f"metric leading size must be {total_size}, got {x.shape[0]}"
            )

    def identity(x):
        return x

    def make_apply(block_callback, additional_callback):
        if block_callback is None and not block_is_identity:
            return None
        if (
            additional_metric is not None
            and additional_callback is None
            and not additional_is_identity
        ):
            return None
        apply_block = identity if block_callback is None else block_callback
        apply_additional = (
            identity if additional_callback is None else additional_callback
        )

        def apply(x):
            check_leading_size(x)
            trailing_shape = x.shape[1:]
            combined = jnp.moveaxis(
                x[:repeated_size].reshape((repeats, block_size) + trailing_shape),
                0,
                1,
            ).reshape(block_size, -1)
            transformed = apply_block(combined)
            repeated = jnp.moveaxis(
                transformed.reshape((block_size, repeats) + trailing_shape),
                0,
                1,
            ).reshape((repeated_size,) + trailing_shape)
            if additional_metric is None:
                return repeated
            return jnp.concatenate(
                [repeated, apply_additional(x[repeated_size:])], axis=0
            )

        return apply

    def make_norm(block_norm, additional_norm):
        if block_norm is None and not block_is_identity:
            return None
        if (
            additional_metric is not None
            and additional_norm is None
            and not additional_is_identity
        ):
            return None
        norm_block = jnp.linalg.norm if block_norm is None else block_norm
        norm_additional = (
            jnp.linalg.norm if additional_norm is None else additional_norm
        )

        def norm(x):
            check_leading_size(x)
            if x.ndim != 1:
                raise ValueError("metric norm requires a vector")
            blocks = x[:repeated_size].reshape(repeats, block_size)
            squared_norm = jnp.sum(jax.vmap(norm_block)(blocks) ** 2)
            if additional_metric is not None:
                squared_norm = squared_norm + norm_additional(x[repeated_size:]) ** 2
            return jnp.sqrt(squared_norm)

        return norm

    additional_solve = None if additional_metric is None else additional_metric.solve
    additional_norm = None if additional_metric is None else additional_metric.norm
    additional_inv_sqrt = (
        None if additional_metric is None else additional_metric.inv_sqrt
    )
    additional_inv_sqrt_transpose = (
        None if additional_metric is None else additional_metric.inv_sqrt_transpose
    )
    return Metric(
        solve=make_apply(block_metric.solve, additional_solve),
        norm=make_norm(block_metric.norm, additional_norm),
        inv_sqrt=make_apply(block_metric.inv_sqrt, additional_inv_sqrt),
        inv_sqrt_transpose=make_apply(
            block_metric.inv_sqrt_transpose,
            additional_inv_sqrt_transpose,
        ),
    )
