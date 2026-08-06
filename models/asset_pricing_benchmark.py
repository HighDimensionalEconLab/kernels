import jax.numpy as jnp


# No-bubble fundamental price for the linear asset-pricing ODE
#   mu'(t) = r mu(t) - x(t),  x(t) = (x_0 + c/g) e^{g t} - c/g,
# with r > g. The general solution is mu(t) = mu_f(t) + zeta e^{r t}; the
# transversality (no-bubble) condition selects zeta = 0.
def mu_f_array(t, c, g, r, x_0):
    return (x_0 + c / g) * jnp.exp(g * t) / (r - g) - c / (g * r)
