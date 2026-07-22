# Adaptive recursive growth NN

`extra/neoclassical_growth_nn_recursive_adaptive.py` uses a one-function adaptive
state-space approximation:

\[
i_\theta(k)>0,\qquad
\dot k=i_\theta(k)-\delta k,\qquad
c(k)=k^\alpha-i_\theta(k).
\]

The NN input is $\log(k/k_0)$. The Euler residual is

\[
\left(\alpha k^{\alpha-1}-i_\theta'(k)\right)\dot k
-c(k)\left(\alpha k^{\alpha-1}-\delta-\rho\right)=0,
\]

where `jax.jvp` differentiates the same investment network to obtain
$i_\theta'(k)$. There is no second function approximation.

## Accepted-step collocation

Every residual evaluation integrates the current policy with Tsit5 or Rodas5P
and requests `SaveAt(steps=True)`. The active collocation rows are exactly the
initial state and accepted internal integrator states. Rejected attempts are
omitted by tinydiffeq, while unused capacity is masked out of both the Euler and
consumption-domain residual blocks.

The controller accept/reject decisions and timestep updates are discrete and
locally frozen under AD. Conditional on that accepted branch, every solver stage
and accepted state is differentiated with respect to all NN parameters. Thus LM
sees how a parameter perturbation moves the state-space collocation points; it
does not merely evaluate a changed policy on fixed states.

The default controller limits successive timestep growth to a factor of three
but imposes no maximum timestep. This keeps the cold stationary initialization
informative while allowing large steps across the near-steady-state tail.
The automatic training-attempt ceilings are 64 for Tsit5 and 128 for Rodas5P;
an explicit `adaptive_steps` value overrides either default.

## Evaluation and transversality

Training and in-domain testing cover $t\in[0,100]$. Testing is a fresh adaptive
integration saved at 401 requested times, rather than a reuse of the training
states. The log-TVC diagnostic is evaluated ex post over $t\in[100,200]$ and
never enters the residual, timestep acceptance, LM acceptance, or final solution
acceptance.

The NN uses the identity parameter metric. Unlike the moving-center kernel
solver, it has no Gram matrix, coefficient transfer, or callback-driven metric
refresh.
