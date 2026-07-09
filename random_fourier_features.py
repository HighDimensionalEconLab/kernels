from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax import config

from kernels import normalize_kernel

config.update("jax_enable_x64", True)

KernelName = Literal["rbf", "matern12", "matern32"]


@dataclass(frozen=True)
class RFFSettings:
    input_dim: int
    num_features: int
    output_dim: int = 1
    kernel: KernelName | float = "rbf"
    radius: float = 1.0
    sigma: float = 1.0
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float64


def normalize_kernel_name(kernel: KernelName | float) -> KernelName:
    name = normalize_kernel(kernel)
    if name not in ("rbf", "matern12", "matern32"):
        raise ValueError(f"Unsupported RFF kernel: {kernel!r}")
    return name


def sample_frequencies(
    key,
    *,
    input_dim: int,
    num_features: int,
    kernel: KernelName | float = "rbf",
    radius: float = 1.0,
    dtype=jnp.float64,
):
    kernel_name = normalize_kernel_name(kernel)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    shape = (num_features, input_dim)
    if kernel_name == "rbf":
        return jax.random.normal(key, shape, dtype=dtype) / radius

    df = 1.0 if kernel_name == "matern12" else 3.0
    normal_key, chi_square_key = jax.random.split(key)
    z = jax.random.normal(normal_key, shape, dtype=dtype)
    chi_square = jax.random.chisquare(
        chi_square_key,
        df=jnp.asarray(df, dtype=dtype),
        shape=(num_features, 1),
        dtype=dtype,
    )
    return z / jnp.sqrt(chi_square / df) / radius


def exact_stationary_kernel(
    x,
    y,
    *,
    kernel: KernelName | float = "rbf",
    radius: float = 1.0,
    sigma: float = 1.0,
):
    kernel_name = normalize_kernel_name(kernel)
    distance = jnp.linalg.norm(x - y, axis=-1)
    if kernel_name == "rbf":
        return sigma**2 * jnp.exp(-(distance**2) / (2.0 * radius**2))
    if kernel_name == "matern12":
        return sigma**2 * jnp.exp(-distance / radius)
    exponent = jnp.sqrt(3.0) * distance / radius
    return sigma**2 * (1.0 + exponent) * jnp.exp(-exponent)


def exact_kernel_matrix(
    x,
    y,
    *,
    kernel: KernelName | float = "rbf",
    radius: float = 1.0,
    sigma: float = 1.0,
):
    pairwise_differences = x[:, None, :] - y[None, :, :]
    return exact_stationary_kernel(
        pairwise_differences,
        jnp.zeros_like(pairwise_differences),
        kernel=kernel,
        radius=radius,
        sigma=sigma,
    )


class RFFFeatures(nnx.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_features: int,
        kernel: KernelName | float = "rbf",
        radius: float = 1.0,
        sigma: float = 1.0,
        dtype=jnp.float64,
        rngs: nnx.Rngs,
    ):
        kernel_name = normalize_kernel_name(kernel)
        frequency_key = rngs.params()
        phase_key = rngs.params()
        frequencies = sample_frequencies(
            frequency_key,
            input_dim=input_dim,
            num_features=num_features,
            kernel=kernel_name,
            radius=radius,
            dtype=dtype,
        )
        phases = jax.random.uniform(
            phase_key,
            shape=(num_features,),
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=dtype,
        )
        self.input_dim = input_dim
        self.num_features = num_features
        self.kernel = kernel_name
        self.radius = radius
        self.sigma = sigma
        self.frequencies = nnx.Variable(frequencies)
        self.phases = nnx.Variable(phases)

    def __call__(self, x):
        x = jnp.asarray(x)
        projection = x @ self.frequencies[...].T + self.phases[...]
        scale = jnp.sqrt(2.0 * self.sigma**2 / self.num_features)
        return scale * jnp.cos(projection)

    def gram(self, x, y):
        x_features = self(x)
        y_features = self(y)
        return x_features @ y_features.T


class RandomFourierFeatures(nnx.Module):
    def __init__(
        self,
        settings: RFFSettings,
        *,
        rngs: nnx.Rngs,
    ):
        self.features = RFFFeatures(
            input_dim=settings.input_dim,
            num_features=settings.num_features,
            kernel=settings.kernel,
            radius=settings.radius,
            sigma=settings.sigma,
            dtype=settings.dtype,
            rngs=rngs,
        )
        self.head = nnx.Linear(
            settings.num_features,
            settings.output_dim,
            use_bias=settings.use_bias,
            dtype=settings.dtype,
            param_dtype=settings.dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.head(self.features(x))
