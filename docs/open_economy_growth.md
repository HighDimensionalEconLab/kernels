# Continuous-time open-economy growth with initial net foreign assets

Status: implemented one-country stepping stone and two-country closed-world extension;
design complete for the static-trade extension. This note fixes the economic object,
derives the continuous-time equations, maps them to the repository's NLLS/RKHS
formulation, and records how the minimum-norm solution is compared ex post with the
conventional economic boundary-value solution.

## Executive decision

The first implementation should be a one-good deterministic small open economy with:

- Cobb-Douglas production and a fixed world interest rate;
- net foreign assets as a predetermined state;
- the concave capital-installation technology in Ravikumar, Santacreu, and Sposi (2019);
- CRRA preferences and perfect foresight;
- no debt-elastic interest premium, portfolio cost, NFA target, or other artificial
  stationarity device;
- an ex-post transversality/no-Ponzi check, not a solver callback; and
- an NLLS-gram kernel solution checked against both a boundary-value solver and
  semi-analytic identities.

This is the cleanest stepping stone because it isolates the desired history dependence.
With the interest rate equal to the household discount rate, consumption is flat. The
initial net foreign asset position changes both permanent consumption and the long-run
net foreign asset position. Capital still follows a standard adjustment-cost growth
transition.

The right description is **history dependence** or a **continuum of steady states indexed
by net foreign assets**, not multiple equilibrium paths conditional on the same complete
initial state. The current account is a flow; the predetermined initial condition is the
stock of net foreign assets, or equivalently external debt.

The target is therefore determinacy conditional on the complete initial state, together
with different long-run external positions across different initial states. Multiplicity
in the static trade equations or several interest-rate paths for the same complete state
would be a separate problem, not the hysteresis this example is intended to preserve.

## Literature decision

No single paper found supplies all four ingredients at once: continuous time, the desired
initial-asset dependence, the Ravikumar-Santacreu-Sposi trade structure, and a directly
portable quantitative calibration. The implementation should therefore use a transparent
synthesis.

1. [Ravikumar, Santacreu, and Sposi (2019), “Capital Accumulation and Dynamic Gains from Trade”](https://s3.amazonaws.com/real.stlouisfed.org/wp/2017/2017-005.pdf)
   is the target for the adjustment technology, common parameters, eventual Eaton-Kortum
   trade block, endogenous trade imbalances, and the 44-country quantitative exercise.
   It is deterministic and perfect-foresight, but explicitly discrete time.

2. [Alvarez (2017), “Capital Accumulation and International Trade”](https://www.sciencedirect.com/science/article/pii/S0304393217300946)
   is the continuous-time bridge to the eventual trade model. It embeds standard
   differential-equation growth dynamics in a multicountry Eaton-Kortum environment.
   Its deliberate limitation is continuously balanced trade: countries cannot borrow or
   lend, so it cannot generate the initial-NFA mechanism by itself.

3. [Turnovsky and Sen (1990/1991), “Fiscal Policy, Capital Accumulation, and Debt in an Open Economy”](https://www.nber.org/papers/w3489)
   is the continuous-time theory anchor. It combines a fixed world interest rate, capital
   adjustment costs, foreign assets, and long-run equilibria that depend on initial asset
   stocks. Its temporary-shock comparative statics are the relevant hysteresis mechanism.

4. [Schmitt-Grohé and Uribe (2003), “Closing Small Open Economy Models”](https://www.nber.org/papers/w9270)
   provides the standard warning: with an internationally traded bond and an exogenous
   interest rate, the steady state generally inherits initial conditions and the model has
   a nonstationary or unit-root component. Here that feature is intentional.

5. [Obstfeld and Rogoff (1994), “The Intertemporal Approach to the Current Account”](https://www.nber.org/system/files/working_papers/w4893/w4893.pdf)
   provides the consumption-smoothing and intertemporal-budget interpretation of the
   deterministic small-open-economy benchmark.

6. [Alvarez and Lucas (2007), “General Equilibrium Analysis of the Eaton–Kortum Model of International Trade”](https://www.nber.org/papers/w11764)
   establishes existence and, under additional restrictions, uniqueness up to numeraire
   for the static Eaton–Kortum system used as the eventual trade block. Those restrictions
   must be checked for the chosen extension rather than treating static uniqueness as
   automatic.

7. [Kehoe, Levine, and Romer (1990), “Determinacy of Equilibria in Dynamic Models with Finitely Many Consumers”](https://doi.org/10.1016/0022-0531(90)90082-U)
   gives the relevant Negishi-weight interpretation: a dynamic competitive equilibrium
   can be reduced to a finite-dimensional search over relative welfare/wealth weights,
   with local determinacy for almost all endowments under regularity conditions. This is
   a representation of the equilibrium, not a device that pins external assets.

The linked Google Scholar citation is the first paper above. I found no continuous-time
version of that paper on the authors' publication pages or among its revisions. The
continuous-time model below is therefore an adaptation, not a claim about the timing in
the original paper.

### Adjustment-cost normalization in the final paper

The final 2019 paper writes

\[
K_{t+1}
=
(1-\delta_d)K_t
+\delta_d^{\,1-\lambda}X_t^\lambda K_t^{1-\lambda}.
\]

This normalization ensures \(X^\star=\delta_d K^\star\). An earlier Dallas Fed revision
introduced a separate parameter \(\chi\), stated \(\chi=\delta_d^{1-\lambda}\), but also
printed \(\chi=0.28\) in its parameter table. Those statements are inconsistent:
\(0.06^{1-0.76}\simeq 0.509\), not \(0.28\). The final paper resolves the issue by
substituting \(\delta_d^{1-\lambda}\) directly into the law of motion and removing
\(\chi\) from the table. The implementation must follow the final version and must not
use \(0.28\) as its baseline. The literal archived-table value would imply
\(x^\star/k^\star\simeq0.132\), contradicting the stated \(0.06\) replacement rate; it
is relevant only as an archival sensitivity check.

## The stepping-stone model

### Environment and notation

Time is continuous, \(t\geq 0\). Labor is normalized to one. The household solves

\[
\max_{\{c_t,x_t\}_{t\geq 0}}
\int_0^\infty e^{-\rho t}
\frac{c_t^{1-\gamma}-1}{1-\gamma}\,dt,
\qquad
\gamma=\frac{1}{\sigma},
\]

subject to

\[
\dot a_t
=
r^\star a_t+f(k_t,z_t)-c_t-x_t,
\tag{1}
\]

\[
\dot k_t
=
\chi x_t^\lambda k_t^{1-\lambda}-\delta k_t,
\qquad
\chi\equiv\delta^{\,1-\lambda},
\tag{2}
\]

with \(k_0>0\) and \(a_0\) given. The production function is

\[
f(k,z)=z k^\alpha.
\tag{3}
\]

Here:

- \(a_t\) is net foreign assets, so \(a_t<0\) means external debt;
- \(d_t=-a_t\) is external debt under the opposite sign convention;
- \(CA_t=\dot a_t\) is the current account;
- \(x_t\) is expenditure on gross investment;
- \(r^\star\) is the fixed world interest rate; and
- \(z_t=1\) in the baseline, with a temporary deterministic productivity path used later.

Equation (2) is the continuous-time flow analogue of the paper's installation
technology. It is not a literal fixed-parameter limit of the annual difference equation:
the efficiency coefficient has time units and must be rescaled when the time unit changes.
The chosen normalization preserves the economically important property
\(x^\star=\delta k^\star\).

Equivalently, investment expenditure can be recovered from a proposed capital path as

\[
x
=
k\chi^{-1/\lambda}
\left(\frac{\dot k}{k}+\delta\right)^{1/\lambda}.
\]

For \(0<\lambda<1\), this inverse installation requirement is convex in capital growth.
The interior first-order condition used below requires \(x>0\), equivalently
\(\dot k/k> -\delta\). Baseline and shock experiments must verify this inequality. If a
larger negative shock reaches \(x=0\), the implementation must use the corresponding KKT
condition rather than extrapolate equation (6) through the corner.

### Hamiltonian and first-order conditions

Let \(m_t\) be the current-value co-state on net foreign assets and \(v_t\) the
current-value co-state on capital. The current-value Hamiltonian is

\[
\mathcal H
=
u(c)
+m\left[r^\star a+f(k,z)-c-x\right]
+v\left[\chi x^\lambda k^{1-\lambda}-\delta k\right].
\]

Define installed-capital value in units of the consumption good by

\[
q_t\equiv\frac{v_t}{m_t},
\qquad
s_t\equiv\frac{x_t}{k_t}.
\]

The consumption first-order condition and foreign-asset co-state equation are

\[
u'(c_t)=c_t^{-\gamma}=m_t,
\tag{4}
\]

\[
\dot m_t=(\rho-r^\star)m_t.
\tag{5}
\]

The investment first-order condition is

\[
1=q_t\chi\lambda s_t^{\lambda-1},
\]

so investment is an algebraic function of \(q_t\):

\[
s(q)
=
\left(\chi\lambda q\right)^{1/(1-\lambda)},
\qquad
x(q,k)=k\,s(q).
\tag{6}
\]

Combining the capital and foreign-asset co-state equations gives

\[
\dot q_t
=
\left[
r^\star+\delta-(1-\lambda)\chi s(q_t)^\lambda
\right]q_t
-f_k(k_t,z_t).
\tag{7}
\]

Equations (1), (2), (5), and (7), together with the algebraic equations (4) and
(6), are the simple model's semi-explicit DAE representation. It is especially useful
for the larger trade model because retaining \(m\), rather than differentiating a
consumption-price equation, avoids derivatives of an implicit static equilibrium.

### Flat consumption

Equation (4) implies

\[
\frac{\dot c_t}{c_t}
=
\frac{r^\star-\rho}{\gamma}.
\tag{8}
\]

The baseline imposes

\[
r^\star=\rho.
\tag{9}
\]

Hence \(m_t\) and \(c_t\) are constant. This result does not depend on the details of
capital adjustment. It does depend on the fixed interest rate, the one-good numeraire,
and the equality in (9). In the multicountry trade model, time-varying consumption prices
will generally make real consumption move even when the nominal marginal value \(m_i\)
is constant.

## Steady state and initial-asset dependence

Set \(z=1\). A stationary capital stock requires

\[
\chi(s^\star)^\lambda=\delta.
\]

With \(\chi=\delta^{1-\lambda}\),

\[
s^\star=\delta,
\qquad
q^\star=\frac{1}{\lambda}.
\tag{10}
\]

The paper's phrase “no adjustment cost in steady state” refers to the level identity
\(x^\star=\delta k^\star\). It does not mean the marginal installed-capital wedge
vanishes: \(q^\star=1/\lambda>1\) when \(\lambda<1\).

The stationary \(q\) equation then gives

\[
f_k(k^\star,1)
=
\delta+\frac{r^\star}{\lambda}.
\tag{11}
\]

For Cobb-Douglas production,

\[
k^\star
=
\left(
\frac{\alpha}{\delta+r^\star/\lambda}
\right)^{1/(1-\alpha)}.
\tag{12}
\]

The stationary net foreign asset equation is

\[
c^\star
=
f(k^\star,1)-\delta k^\star+r^\star a^\star.
\tag{13}
\]

Equation (13) does not determine \(a^\star\). There is a continuum of stationary
allocations indexed by the net foreign asset stock. More inherited debt implies lower
permanent consumption and, in the simple model, a correspondingly different long-run
debt stock.

### Exact comparative statics with respect to initial net foreign assets

Let

\[
g_t\equiv f(k_t,z_t)-x_t
\]

denote domestic resources after investment but before consumption. The no-Ponzi
condition and equation (1) imply the intertemporal budget constraint

\[
c
=
r^\star
\left[
a_0+\int_0^\infty e^{-r^\star t}g_t\,dt
\right]
\tag{14}
\]

when \(r^\star=\rho\), so consumption is constant. If \(g_t\to g_\infty\), then

\[
a_\infty=\frac{c-g_\infty}{r^\star}.
\tag{15}
\]

In this separable one-good model, the \((k,q)\) subsystem is independent of \(a_0\).
Therefore two solutions that differ only in initial net foreign assets obey

\[
\frac{\partial c}{\partial a_0}=r^\star,
\qquad
\frac{\partial a_t}{\partial a_0}=1
\quad\text{for every }t,
\qquad
\frac{\partial a_\infty}{\partial a_0}=1,
\qquad
\frac{\partial CA_t}{\partial a_0}=0.
\tag{16}
\]

Under the debt convention \(d=-a\),

\[
\frac{\partial c}{\partial d_0}=-r^\star,
\qquad
\frac{\partial d_\infty}{\partial d_0}=1.
\tag{17}
\]

This answers the key interpretation directly: yes, the initial debt stock selects a
different long-run debt stock. It also selects a different constant consumption level.
It does not create several paths from the same complete state. Inherited wealth alone
does not change the capital or current-account transition in this stripped-down model:
the trade balance shifts by \(-r^\star\Delta a_0\), exactly offsetting the change in net
foreign income \(r^\star\Delta a_0\).

If \(k_0\neq k^\star\), transition wealth also shifts the level of \(a_\infty\) relative
to \(a_0\). For example, a capital-scarce country can borrow during capital accumulation,
run an initial current-account deficit, and converge to a larger debt stock while keeping
consumption smooth. That sign is a quantitative result, not a theorem for every
parameterization; it must be checked in the baseline simulation. Merely changing \(d_0\)
shifts the entire debt path one-for-one and does not change its growth rate. Debt
accumulation comes from the capital gap or another temporary disturbance.

### A stronger hysteresis experiment

Varying \(a_0\) demonstrates state dependence. A temporary shock demonstrates hysteresis
more sharply. Let an **unanticipated at \(t=0\)** productivity path \(z_t\) rise
temporarily and return to one, with perfect foresight over that path after time zero. The shock changes the
transition path of \(g_t\), hence the present value in (14). After \(z_t\) has returned to
baseline, \(k_t\) converges back to the same \(k^\star\), but the accumulated net foreign
asset position, permanent consumption, and long-run trade balance need not return to
their pre-shock values.

Indeed,

\[
a_\infty
=
a_0+\int_0^\infty e^{-r^\star t}\left(g_t-g^\star\right)\,dt.
\]

This is the simplest quantitative hysteresis identity: a temporary event that changes
net output after investment permanently changes the external asset position.

## Parameterization

The common annual parameters should begin with Ravikumar, Santacreu, and Sposi's final
calibration.

| Object | Paper value | Continuous-time use |
|---|---:|---:|
| Capital share, \(\alpha\) | \(0.33\) | \(0.33\) |
| Annual discount factor, \(\beta\) | \(0.96\) | \(\rho=-\log\beta=0.04082199\) |
| Intertemporal elasticity, \(\sigma\) | \(0.5\) | \(\gamma=1/\sigma=2\) |
| Annual depreciation fraction, \(\delta_d\) | \(0.06\) | \(\delta=-\log(1-\delta_d)=0.06187540\) |
| Installation elasticity, \(\lambda\) | \(0.76\) | \(0.76\) |
| Installation efficiency | \(\delta_d^{1-\lambda}\) | \(\chi=\delta^{1-\lambda}=0.51281926\) |
| Eaton-Kortum trade elasticity, \(\theta\) | \(4\) | unused until the trade block |
| Intermediate-variety elasticity, \(\eta\) | \(2\) | unused until the trade block |
| Baseline productivity, \(z\) | country-specific in the paper | \(1\) normalization |
| World interest rate, \(r^\star\) | endogenous in the multicountry paper | \(r^\star=\rho\) in the stepping stone |

The exact annual-to-continuous conversions make survival and discounting agree over one
year. A sensitivity run should also use the common first-order convention
\(\rho=0.04\) and \(\delta=0.06\).

Under the exact conversion and \(z=1\),

\[
q^\star=1.31578947,
\qquad
f_k(k^\star)=0.11558855,
\qquad
k^\star=4.78628624.
\]

Use \(y^\star=f(k^\star,1)\), \(g^\star=y^\star-\delta k^\star\), and
\(m_{\mathrm{ref}}=(g^\star)^{-\gamma}\) as the default output, consumption, and
marginal-wealth scales.

The simple baseline should use \(k_0=1\), which makes the country initially
capital-scarce, and report scenarios at net foreign asset-to-steady-output ratios
\(a_0/y^\star\in\{-0.5,0,0.5\}\). Do not prefilter the kernel solve using the
intertemporal budget in (14). If an extreme debt case lacks a positive-consumption
economic benchmark, report that fact separately from DAE feasibility.

The authors' public
[replication-data record](https://data.mendeley.com/datasets/yw75dzs677/1) currently
contains a replication README rather than the underlying MATLAB code or calibrated
arrays. Use it as a source map, but plan to reconstruct country productivities, trade
costs, sectoral intensities, labor endowments, and initial NFA positions from the listed
public data unless a fuller author archive is located.

## DAE and independent boundary-value benchmark

The canonical DAE variables are

\[
(k_t,a_t,q_t,m_t).
\]

In the stepping-stone calibration \(r^\star=\rho\), so (5) makes \(m\), and therefore
\(c\), constant. The first implementation should eliminate that redundant path and solve
for the three paths \((k,a,q)\) plus one positive scalar \(c\). Retain the \(m\)
formulation in the pure model equations because it generalizes cleanly once the world
interest rate and consumption prices move.

Capital and net foreign assets are predetermined:

\[
k(0)=k_0,
\qquad
a(0)=a_0.
\tag{18}
\]

Installed-capital value and consumption are jump variables. The kernel solve uses no
boundary or terminal equation for either variable. It collocates only the DAE on a finite
observation window and lets the RKHS metric select the two missing degrees of freedom.

For an independent economic benchmark, the usual infinite-horizon saddle-path and
no-Ponzi conditions can be approximated accurately on a finite horizon. A direct
steady-state pin such as \(q(T)=q^\star\) is usable on a very long horizon, but the local
stable manifold gives a better benchmark truncation.

Define

\[
h_T\equiv\log\frac{k(T)}{k^\star},
\qquad
v_T\equiv\log\frac{q(T)}{q^\star}.
\]

The linearized \((h,v)\) system at the steady state is

\[
\begin{bmatrix}
\dot h\\
\dot v
\end{bmatrix}
=
\begin{bmatrix}
0 & \mathcal A\\
\mathcal B & r^\star
\end{bmatrix}
\begin{bmatrix}
h\\
v
\end{bmatrix},
\]

where

\[
\mathcal A=\frac{\delta\lambda}{1-\lambda},
\qquad
\mathcal B=(1-\alpha)(r^\star+\lambda\delta).
\]

Its stable root is

\[
\xi_-
=
\frac{r^\star-\sqrt{(r^\star)^2+4\mathcal A\mathcal B}}{2}<0,
\]

so the capital-tail residual is

\[
R_{\mathrm{stable},T}
=
v_T-\frac{\xi_-}{\mathcal A}h_T.
\tag{19}
\]

The benchmark's asset terminal condition should likewise account for the remaining
stable tail. Let \(g^\star=f(k^\star,1)-\delta k^\star\). Along the linear stable arm,

\[
g_t-g^\star
\simeq
\mathcal G h_T e^{\xi_-(t-T)},
\]

with

\[
\mathcal G
=
\alpha f(k^\star,1)-\delta k^\star
-\frac{\delta k^\star}{1-\lambda}\frac{\xi_-}{\mathcal A}.
\]

The present-value budget from \(T\) onward then gives

\[
R_{\mathrm{PV},T}
=
c(T)-r^\star a(T)-g^\star
-\frac{r^\star\mathcal G}{r^\star-\xi_-}h_T.
\tag{20}
\]

The independent boundary-value benchmark uses

\[
R_{\mathrm{stable},T}=0,
\qquad
\frac{R_{\mathrm{PV},T}}{y^\star}=0
\tag{21}
\]

as its two finite-horizon terminal restrictions. If the capital subsystem is exactly at
steady state, (20) reduces to \(CA(T)=0\). Thus \(q(T)=q^\star\) together with
\(CA(T)=0\) is a useful benchmark comparison, but not the preferred truncation. Neither
version enters the kernel residual vector, metric, warm start, or acceptance decision.

These are finite-horizon approximations, not economic terminal conditions. In particular,
do not impose \(a(T)=0\) or \(a(T)=a_0\): either condition would remove the intended
history dependence. Validation must verify that \(k(T)\) and \(q(T)\) are close to their
steady-state values, the tail approximation is accurate, and extending the horizon does
not change the reported path.

The full local spectrum has one stable capital root, one unstable capital root, the
unstable asset root \(r^\star\), and the zero root associated with the continuum of
long-run NFA positions. Given the two predetermined initial states, the conventional
economic BVP therefore needs the two restrictions in (21). The research question is
whether the minimum-RKHS-norm DAE solution converges to that BVP solution without seeing
either restriction.

### Ex-post transversality and solvency checks

Do not add TVC callbacks to the solver. After solving, report

\[
e^{-r^\star T}a(T),
\qquad
e^{-\rho T}m(T)a(T),
\qquad
e^{-\rho T}m(T)q(T)k(T),
\tag{22}
\]

together with terminal current account, terminal state drifts, and the maximum budget
identity error. Report the absolute magnitudes of the three discounted terms, not only
their signed values. None of these diagnostics enters `solution_accepted`, the rejection
reasons, initialization, or stopping. Also report the tail-corrected present-value gap

\[
\mathcal E_{\mathrm{PV}}
=
c-r^\star
\left[
a_0
+\int_0^T e^{-r^\star t}g_t\,dt
+e^{-r^\star T}
\left(
\frac{g^\star}{r^\star}
+\frac{\mathcal G h_T}{r^\star-\xi_-}
\right)
\right].
\tag{23}
\]

Repeat at a longer horizon. Bounded \(a(T)\), \(k(T)\), \(q(T)\), and \(m(T)\) make the
discounted quantities vanish as \(T\to\infty\).

A no-Ponzi inequality alone rules out explosive debt but can leave unused positive
wealth. The optimal solution must satisfy the budget equality and asset TVC. Likewise,
the paper's wording that debts are eventually paid must not be implemented as
\(a_\infty=0\): a country may retain negative NFA indefinitely while a permanent trade
surplus services the associated interest bill.

## NLLS-gram and integrated-kernel mapping

The implementation should follow the integrated-path design in
[neoclassical_growth_kernel_nlls.py](../neoclassical_growth_kernel_nlls.py) and the
multi-state residual organization in
[human_capital_kernel_nlls.py](../human_capital_kernel_nlls.py).

### Path parameterization

Let \(K\) and \(\widetilde K\) be the derivative and integrated kernel matrices returned
by [kernels.py](../kernels.py). Use positive log coordinates for \(k,q,c\), but retain a
signed level coordinate for net foreign assets:

\[
k(t)
=
k_0\exp\!\left(\widetilde K_t\alpha_k\right),
\]

\[
a(t)
=
a_0+y^\star\widetilde K_t\alpha_a,
\]

\[
q(t)
=
q^\star
\exp\!\left(q_0+\widetilde K_t\alpha_q\right),
\]

\[
c
=
c_{\mathrm{ref}}\exp(c_0),
\qquad
m=c^{-\gamma}.
\tag{24}
\]

In (24), \(q_0\) and \(c_0\) denote free log offsets, not levels. The integrated kernel
automatically enforces the initial conditions for \(k\) and \(a\). The log transforms
guarantee positivity without clipping, branch-dependent penalties, or softplus saturation.

For \(n\) collocation points, the unknowns are three length-\(n\) coefficient vectors plus
the two free log offsets: \(3n+2\) scalars.

### Dimensionless residuals

At each collocation point, compute \(s(q)\), \(x=ks\), \(m=c^{-\gamma}\),
\(f\), \(f_k\), and \(CA\) algebraically. Use

\[
R_k
=
\frac{\dot k}{k}
-\left[\chi s^\lambda-\delta\right],
\]

\[
R_a
=
\frac{\dot a}{y^\star}
-\frac{r^\star a+f-c-x}{y^\star},
\]

\[
R_q
=
\frac{\dot q}{q}
-\left[
r^\star+\delta-(1-\lambda)\chi s^\lambda-\frac{f_k}{q}
\right].
\tag{25}
\]

Do not append (21), a TVC, a no-Ponzi condition, a steady-state target, or any other
terminal residual. There are \(3n\) DAE equations in \(3n+2\) unknowns. The two missing
conditions are deliberately supplied by the minimum-metric selection, exactly as in the
repository's asset-pricing, growth, and optimal-advertising kernel examples.

The residual vector is scaled by \(1/\sqrt{3n}\). The parameter-space metric is

\[
M
=
\operatorname{blockdiag}(K,K,K,0_2)+\epsilon I,
\]

so the selected feasible path minimizes, up to the common \(O(\epsilon)\) shift,

\[
\|\dot{\log k}\|_{\mathcal H_K}^2
+\left\|\frac{\dot a}{y^\star}\right\|_{\mathcal H_K}^2
+\|\dot{\log q}\|_{\mathcal H_K}^2.
\]

The \(\epsilon I_2\) block makes the metric positive definite while leaving the jump
values nearly free. Invoke the optimizer through its public `solver.solve` interface and
report both the three jitter-free RKHS seminorms and the complete shifted metric norm.

### Warm starts and continuation

Use the zero coefficient vector and zero offsets as the baseline initialization. In
levels this is the constant path \((k_0,a_0,q^\star,g^\star)\); it contains no stable-arm,
present-value, terminal-asset, or benchmark information. Continuation from an economic
boundary solution would contaminate the selection experiment and is not allowed. A
numerical retry may change generic damping or initialization scales, but it must not use
TVC/no-Ponzi information.

The baseline calibration has \(\xi_-=-0.0889010\), a linear capital-gap half-life of
about 7.8 years. The implemented minimum-norm baseline uses a Matérn-\(5/2\) kernel,
\(T=160\), and 96 collocation points: 81 evenly spaced points over the active first 40
years and 15 tail points through year 160. A uniform 321-point solve provides the
density check and \(T=200\) provides the horizon check. The long, sparse tail is not a
terminal condition: it exposes explosive alternatives to the RKHS norm for long enough
that their cost can dominate.

### Sequential state-space neural mapping

The NN implementation follows the best-performing recursive formulation in
[neoclassical_growth_nn_recursive_i.py](../neoclassical_growth_nn_recursive_i.py): it
learns level policies over the endogenous state rather than paths over time or time
derivatives. A 32-by-3 GELU trunk maps \(\log(k/k_0)\) to two outputs. The first is
positive investment expenditure through `unit_softplus`; the second is the signed
current account:

\[
x_\theta(k)>0,
\qquad
b_\theta(k)=CA_\theta(k).
\]

The network rolls forward only the predetermined states,

\[
\dot k
=k\left[\chi\left(\frac{x_\theta(k)}{k}\right)^\lambda-\delta\right],
\qquad
\dot a=b_\theta(k),
\]

and recovers both jump variables algebraically:

\[
q_\theta(k)
=\frac{(x_\theta(k)/k)^{1-\lambda}}{\chi\lambda},
\qquad
c_\theta(k,a)
=r^\star a+f(k)-x_\theta(k)-b_\theta(k).
\tag{NN1}
\]

Automatic differentiation with respect to \(k\) supplies the two stationary policy
residuals. The q equation is

\[
R_q
=\frac{d\log q_\theta(k)}{dk}\dot k
-\left[
r^\star+\delta-(1-\lambda)\chi(x_\theta/k)^\lambda
-\frac{f_k}{q_\theta}
\right],
\]

while \(r^\star=\rho\) makes the consumption Euler equation

\[
R_c
=\dot c
=\left[f_k-x_\theta'(k)-b_\theta'(k)\right]\dot k
+r^\star b_\theta(k).
\tag{NN2}
\]

Thus capital accumulation, NFA accumulation, the investment FOC, and the budget identity
hold by construction. Training contains \(2n\) economic residuals plus \(n\) inactive
consumption-domain rows, against 2,242 neural parameters. There are no terminal rows.
The underdetermined LM solve uses the identity parameter metric, so it selects a
minimum-metric update relative to the random feature trunk and near-zero output head.
That initialization uses only current information: replacement investment
\(\delta k_0\), a near-zero current account, and the output scale at \(k_0\). It
contains no steady state, benchmark, stable-arm, present-value, TVC, or no-Ponzi
quantity.

Approximating consumption directly is intentionally avoided. A free constant-consumption
head would make the annuity root depend primarily on its arbitrary output anchor. In
contrast, on the stable capital arm the homogeneous current-account solution behaves
locally like

\[
b_h(k)\propto |k-k^\star|^{r^\star/\xi_-},
\]

which diverges because \(\xi_-<0\). A finite smooth neural policy cannot represent that
explosive branch at \(k^\star\); the architecture and identity-metric LM step therefore
provide the selection experiment rather than an imposed solvency condition.

Fixed-grid RK4 is the measured default. A 48-step Euler training rollout is faster for
seed 0 but misses the independent continuous-time residual gate for seeds 1 and 2. All
reported paths and acceptance residuals are therefore recomputed with a fixed-RK4
validation rollout. Adaptive Tsit5 gives similar accuracy but is slower to train and
substantially slower to compile for this small smooth problem.

## Independent benchmark

The implemented SciPy boundary-value benchmark is analogous to
[neoclassical_growth_benchmark.py](../neoclassical_growth_benchmark.py).

The benchmark has two layers:

1. Solve the two-dimensional \((k,q)\) saddle path with \(k(0)=k_0\) and the
   stable-manifold condition (19).
2. Compute constant consumption from (14), then recover \(a(t)\) from (1), preferably
   backward using the linear-tail value
   \[
   a(T)
   =
   \frac{c-g^\star}{r^\star}
   -\frac{\mathcal G}{r^\star-\xi_-}h_T,
   \]
   to avoid the unstable forward representation of the asset equation.

A direct solve_bvp solution with the three ODE paths, scalar \(c\) as an unknown
parameter, and the four boundary conditions in (18) and (21) is a second check. A
four-state formulation with \(\dot c=0\) is an additional implementation cross-check.
The semi-analytic decomposition is more valuable because it tests the model identities
independently of the kernel and of the full-system boundary solver.

## Acceptance tests

The example is ready only when all of the following hold.

### Algebra and economics

- The Hamiltonian first-order conditions match equations (4)-(7).
- \(x^\star=\delta k^\star\), \(q^\star=1/\lambda\), and equations (11)-(13) hold.
- Consumption is flat to numerical tolerance when \(r^\star=\rho\).
- Across the NFA sweep, capital and \(q\) are invariant while
  \(\Delta c=r^\star\Delta a_0\), \(\Delta a_t=\Delta a_0\), and
  \(\Delta CA_t=0\).
- Under the debt sign convention, inherited debt lowers consumption and shifts long-run
  debt one-for-one.
- A temporary productivity shock that returns to baseline can permanently change
  \(a_\infty\) and \(c_\infty\).
- In the closed-world extension, different initial NFA vectors may produce different
  long-run NFA vectors, but continuation from the same complete initial state must not
  reveal distinct regular equilibrium branches.
- The normalized static trade Jacobian and the independent benchmark's dynamic boundary
  Jacobian have the expected ranks; the only removed static direction is the common
  price numeraire.
- In the one-good \(N\)-country extension, the reduced Jacobian has \(N\) stable,
  \(2N-1\) unstable, and \(N-1\) center directions, and the return from (28) agrees with
  the Euler and q-equation diagnostics.
- No debt target, artificial stationarity device, TVC, no-Ponzi equation, or terminal
  residual appears in the kernel system.

### Numerical quality

- All of \(k,q,m,c,x\) remain strictly positive.
- Normalized training residuals are below \(10^{-5}\).
- Dense-grid validation residuals are below \(5\times10^{-3}\), with tighter reported
  residuals expected for the baseline.
- The maximum normalized path discrepancy against solve_bvp is below \(10^{-3}\) on the
  reporting window, but this is an ex-post research outcome rather than a feasibility
  gate.
- The ex-post stable-arm gap, present-value gap, discounted positions, and current
  account become small as the observation horizon grows.
- Extending \(T=160\) to \(T=200\) changes every reported path by less than \(10^{-3}\).
- The horizon sweep records the empirical selection rate; it does not assume the
  second-order truncation rate of the benchmark conditions in (19)-(20).
- Results are stable under modest changes in kernel length scale, collocation density, and
  NLLS metric shift.

### Performance

- Report compile time separately from warm solve time.
- Reuse compiled solves across the NFA and shock grids.
- Compare dense Cholesky, dense QR, and eligible structured/matrix-free paths at the same
  collocation points.
- Record NLLS steps, accepted/rejected steps, residual evaluations, and peak problem size.
- Treat the existing neoclassical-growth NLLS example at matched \(n\) as an engineering
  reference, while allowing for the extra state and algebra.

## Planned files and function boundaries

### Stage 1: small open economy

Implemented:

- open_economy_growth.py for pure equations, parameters, steady state, and diagnostics;
- open_economy_growth_benchmark.py for solve_bvp and semi-analytic checks;
- open_economy_growth_kernel_nlls.py for the integrated-kernel solve;
- open_economy_growth_nn_recursive.py for the sequential state-space NN solve;
- replication/open_economy_growth_kernel.py for cached kernel runs;
- replication/open_economy_growth_kernel_figures.py for kernel transition plots;
- replication/open_economy_growth_nn_recursive.py for cached NN runs; and
- replication/open_economy_growth_nn_recursive_figures.py for NN transition plots.

Keep these pure functions separately testable:

- production(k, z, params)
- marginal_product(k, z, params)
- investment_rate_from_tobin_q(tobin_q, params)
- consumption_from_marginal_wealth(m, params)
- capital_drift(k, tobin_q, params)
- current_account(k, a, tobin_q, c, z, params)
- tobin_q_drift(k, tobin_q, z, params)
- marginal_wealth_drift(m, params)
- steady_state(params)
- path_residuals(...)
- tail_diagnostics(...)
- tvc_diagnostics(...)

### Stage 2: deterministic shock and hysteresis

Add a smooth, temporary productivity shock \(z(t)\) that is unanticipated at time zero,
is thereafter perfectly foreseen, and returns exactly to one. Compare:

- the unshocked path;
- a positive temporary shock;
- a negative temporary shock; and
- the same shocks under several initial NFA positions.

Plot capital, investment, consumption, NFA, debt, current account, and the discounted
solvency diagnostics. This stage is the decisive test that a temporary event leaves a
permanent external-position effect.

### Stage 3: two-country closed world

Implemented in [multicountry_growth.py](../multicountry_growth.py),
[multicountry_growth_benchmark.py](../multicountry_growth_benchmark.py), and
[multicountry_growth_kernel_nlls.py](../multicountry_growth_kernel_nlls.py), with cached
experiments and figures under `replication/multicountry_growth_kernel*.py`.

Replace the fixed interest rate with an endogenous world rate \(r(t)\). To make country
size explicit, define \(k_i,a_i,c_i,x_i\) per worker and let fixed population be \(L_i\).
Use \(N-1\) independent NFA states and impose

\[
\sum_{i=1}^N L_i a_i(t)=0.
\]

Fixed \(L_i\) is part of the baseline. Population growth would add dilution terms to
per-worker capital and assets and must also enter the derivative of global asset clearing;
it cannot be introduced by changing calibration alone.

For the common-discount-rate baseline, the country marginal values of wealth satisfy

\[
\frac{m_i(t)}{m_j(t)}=\text{constant}.
\]

Introduce one positive common marginal-wealth path \(M(t)\) and positive constant
relative wealth weights \(\omega_i\), defined exactly by

\[
m_i(t)=\frac{M(t)}{\omega_i},
\qquad \omega_1=1.
\tag{26}
\]

The normalization removes only the common scale of \((M,\omega_1,\ldots,\omega_N)\);
it does not normalize consumption, assets, or any long-run external position. Household
optimality gives

\[
c_i(t)
=
\left(\frac{M(t)P_{c,i}(t)}{\omega_i}\right)^{-1/\gamma},
\qquad
r(t)=\rho-\frac{\dot M(t)}{M(t)}.
\tag{27}
\]

In the one-good model \(P_{c,i}=1\). Retain the \(N-1\) independent country asset laws
and one world resource-clearing residual. Do **not** try to determine \(r\) from the
omitted country's asset law. Substitution of the other asset laws and global asset
clearing makes every \(rL_i a_i\) term cancel, so that omitted law is precisely world
resource clearing. Equation (27), solved jointly with resource clearing and household
policies, identifies the interest-rate path even when every current asset position is
zero. Conditional on a complete differentiable path, a country with \(a_i\ne0\) also
recovers the same rate from its own budget law; this is a diagnostic, not the closure.

Each capital co-state equation supplies another exact diagnostic,

\[
r_i^{(q)}
=
\frac{\dot q_i}{q_i}-\delta+(1-\lambda)\chi s_i^\lambda
+\frac{R_{k,i}}{q_i},
\]

where \(R_{k,i}=f_{k,i}\) in the one-good model, and all \(r_i^{(q)}\) must equal the
rate in (27). This rearrangement is a consistency diagnostic, not an independent closure:
it is simply the q law solved for \(r\), with \(\dot q_i\) endogenous. The world return is
instead determined jointly by resource clearing and intertemporal saving and investment
optimality. There is no additional informative static bond-clearing equation once the
zero-net-supply
identity is imposed; with adjustment costs, the return also need not equal a raw marginal
product of physical capital at each instant.

For the one-good model this identification can be made explicit. Define

\[
\Phi(k,q,M;\omega)
=
\sum_{i=1}^N L_i\left[f_i(k_i)-c_i(M,\omega_i)-x_i(k_i,q_i)\right].
\]

World resource clearing is \(\Phi=0\). Differentiating it along the household and
capital first-order conditions gives an index-reduced algebraic formula for \(r\):

\[
r
\left[
\frac{1}{\gamma}\sum_i L_i c_i
+\frac{1}{1-\lambda}\sum_i L_i x_i
\right]
=
\sum_i L_i(R_{k,i}-s_i)\dot k_i
+\frac{\rho}{\gamma}\sum_i L_i c_i
-\frac{1}{1-\lambda}\sum_i L_i x_i\mathcal J_i,
\tag{28}
\]

where

\[
\mathcal J_i
=
\delta-(1-\lambda)\chi s_i^\lambda-\frac{R_{k,i}}{q_i}.
\]

The coefficient on \(r\) in (28) is strictly positive for interior \(c_i,x_i\), so the
one-good return is uniquely defined at every regular point, including when all
\(a_i=0\). In the full trade model endogenous static prices add implicit-derivative terms;
the corresponding index-reduction coefficient must be checked for nonzero rank rather
than assumed positive.

In the economic equilibrium, the \(N-1\) given initial asset positions determine the
\(N-1\) relative weights through their lifetime budget/PV conditions. Thus (26) is the
continuous-time Negishi representation of the competitive equilibrium, not an artificial
stationarity device. In the kernel DAE these weights remain free coordinates selected by
the metric, and their lifetime budgets are checked only ex post. Different initial NFA
vectors may select different weights and long-run NFA positions, while a given complete
initial state should select one locally determinate path.

An independent conventional BVP benchmark should construct its finite-horizon boundary
conditions from the unstable projection of the **full coupled Jacobian**, while retaining
its center directions. That benchmark uses exactly \(2N-1\) terminal restrictions:

- \(N\) coupled stable-subspace conditions for installed capital;
- \(N-1\) independent present-value tail conditions for external assets.

Do not additionally impose \(\dot M(T)=0\) or \(r(T)=\rho\) in that benchmark. Those
equalities hold only at the limiting stationary allocation; at finite \(T\), imposing one
on top of the full unstable projection cuts a legitimate stable direction and
overdetermines the economic BVP. The kernel system, by contrast, appends none of these
conditions and uses the BVP only for ex-post comparison.

The differential variables are the \(N\) capital paths, \(N\) installed-capital-value
paths, and \(N-1\) asset paths: \(3N-1\) paths. Marginal wealth \(M\) is algebraic,
because \(\Phi=0\) fixes its level given \((k,q,\omega)\). Either eliminate \(M\)
pointwise or represent it with an algebraic kernel block having no free initial offset;
do not give it an integrated-path initial condition. The free constants are the \(N\)
initial Tobin-q values and \(N-1\) relative weights, for a total of \(2N-1\).

If the algebraic \(M\) block has \(n\) coefficients, the kernel collocation count is
\(3Nn+2N-1\) unknowns: \((3N-1)n\) differential-path coefficients, \(n\) algebraic
coefficients, and \(2N-1\) constants. There are \(3Nn\) pointwise equations from the
capital laws, q laws, \(N-1\) budgets, and \(\Phi=0\). The system is underdetermined by
\(2N-1\), and its block RKHS metric selects those degrees of freedom. Verify numerically
that the reduced Jacobian has \(N\) stable roots, \(2N-1\) unstable roots, and \(N-1\)
center directions associated with relative wealth; also verify the normalized DAE
Jacobian and the independent benchmark projection have their intended ranks.

For the two-country one-good baseline, eliminate \(M\) exactly. With \(\omega_1=1\),
define aggregate resources available for consumption and the weight sum by

\[
\mathcal R(k,q)
=
\sum_{i=1}^2 L_i\left[f_i(k_i)-x_i(k_i,q_i)\right],
\qquad
\mathcal W(\omega_2)
=
L_1+L_2\omega_2^{1/\gamma}.
\]

Resource clearing gives

\[
M
=
\left(\frac{\mathcal W}{\mathcal R}\right)^\gamma,
\qquad
c_i
=
\omega_i^{1/\gamma}\frac{\mathcal R}{\mathcal W},
\qquad
a_2=-\frac{L_1}{L_2}a_1,
\]

provided \(\mathcal R>0\). Compute \(r\) with the explicit index-reduced formula (28),
which is equivalent to \(r=\rho+\gamma\dot{\mathcal R}/\mathcal R\) after substituting
the q equations. The differential paths are then only
\((k_1,k_2,q_1,q_2,a_1)\), and \(\omega_2\) is a positive constant. With \(n\)
collocation points, the minimal integrated-kernel system has \(5n+3\) unknowns and
\(5n\) equations: five coefficient vectors, the two initial q offsets,
\(\log\omega_2\), and five ODE residuals per point. The RKHS metric selects the three
free constants without capital-tail or asset-PV residuals. This is the preferred Stage 3
implementation because it removes the algebraic path before optimization without
changing the economics.

Begin with two symmetric countries, equal \(L_i\), and opposite initial NFA positions.
Verify global asset clearing, every retained budget law, the reconstructed country's
budget law, and Walras' law at every time. Continue initial NFA away from zero and check
that the same starting state does not produce distinct branches, while different starting
NFA positions retain their economically intended long-run differences.

## Mapping to the full Ravikumar-Santacreu-Sposi trade model

The final model keeps the same dynamic shell but replaces one-good production with an
implicit static trade equilibrium at each time.

For country \(i\), retain per-worker dynamic variables \((k_i,a_i,q_i)\), with only
\(N-1\) independent \(a_i\) paths, and solve for common marginal wealth \(M\)
algebraically from world resource clearing. Country marginal wealth is recovered from
(26), rather than represented by \(N\) redundant dynamic paths. The
static trade block returns the consumption price \(P_{c,i}\), investment price
\(P_{x,i}\), rental income per unit of capital \(R_{k,i}\), wages, trade shares, output,
and market-clearing residuals. Use the algebraic household equations

\[
c_i
=
\left(\frac{M P_{c,i}}{\omega_i}\right)^{-1/\gamma},
\tag{29}
\]

\[
P_{x,i}
=
q_i\chi\lambda s_i^{\lambda-1},
\qquad
s_i
=
\left(
\frac{\chi\lambda q_i}{P_{x,i}}
\right)^{1/(1-\lambda)}.
\tag{30}
\]

The endogenous world return and installed-capital co-state equations are

\[
r=\rho-\frac{\dot M}{M},
\tag{31}
\]

\[
\dot q_i
=
\left[r+\delta-(1-\lambda)\chi s_i^\lambda\right]q_i
-R_{k,i}.
\tag{32}
\]

The budget/current-account equation uses all prices in a common numeraire:

\[
\dot a_i
=
r a_i+\text{factor income}_i
-P_{c,i}c_i-P_{x,i}x_i.
\tag{33}
\]

The income term must use the same price system as the q equation. Under competitive
constant returns it is \(w_i+R_{k,i}k_i\), plus any rebated tariff revenue or profits
defined by the static block; it must reconcile exactly with value added so capital income
is neither omitted nor counted twice.

This choice avoids differentiating \(P_{c,i}\) or \(P_{x,i}\). The static prices and
allocations may be solved internally by a differentiable implicit function; they can also
be included as algebraic DAE unknowns and residuals if nested solves prove expensive or
fragile.

In the full trade block, the implicit derivative of resource clearing with respect to
\(M\) and the dynamic variables replaces the explicit one-good formula (28). Require the
resulting DAE/index-reduction Jacobian to have full rank along the entire path. A zero or
near-zero coefficient is a failure of local interest-rate identification, not a reason to
add an external debt-stationarity device.

All per-worker flows must be aggregated with \(L_i\). In particular,
\(\sum_i L_i a_i=0\), and the omitted asset path is reconstructed with those population
weights. Keep static_trade_equilibrium, static_trade_residuals, world_asset_clearing, and
multicountry_tail_diagnostics as explicit function boundaries. The last is evaluation
only and must never be concatenated into the kernel residual. This lets the same trade
block be tested first in isolation, then under a nested solve, and finally as part of the
joint NLLS system. In code, call the installed-capital value tobin_q and reserve r for
the world bond return; this avoids colliding with the paper's use of \(q_t\) for the bond
return.

Represent only \(N-1\) asset paths and set the last by global clearing. Retain one
independent global resource/static-market-clearing equation and drop only the genuinely
redundant Walras equation. The reconstructed country's asset equation must then hold as a
diagnostic; it cannot determine the world interest rate because its interest terms cancel
against those in the retained budget equations.

The Alvarez–Lucas uniqueness result is a useful static benchmark, but the exact RSS block
has capital, intermediates, and endogenous imbalances. Verify its assumptions or establish
local uniqueness directly from the normalized static Jacobian. The solver must reject a
rank-deficient static equilibrium rather than silently choosing a branch. This check is
separate from, and must not eliminate, the intended dynamic history dependence.

Impose one conventional numeraire normalization on the static price system. This removes
only a common nominal price scale. A constant rescaling correspondingly rescales \(M\),
but leaves \(-\dot M/M\), the real interest rate, allocations, and external positions
unchanged. The redundant normalization is one time-invariant global scale, not a fresh
independent scale at every date: a time-varying price rescaling changes the measured
numeraire return by its inflation rate. Fix the intertemporal numeraire consistently
before interpreting \(r(t)\).

Do not add debt-elastic rates, portfolio costs, borrowing wedges, target-NFA laws, or
terminal conditions of the form \(a_i(T)=0\) or \(a_i(T)=a_i(0)\). Such devices would
erase the phenomenon this example is meant to study. Heterogeneous discount rates are a
later extension: then \(m_i/m_j\) is no longer constant and the common-weight reduction
must be generalized rather than patched with a stationarity device.

The progression should be:

1. one country, one good, fixed \(r^\star\);
2. one country with a temporary productivity shock;
3. two countries, one good, endogenous \(r(t)\);
4. two countries with the static Eaton-Kortum trade block;
5. a small calibrated subset of countries;
6. all 44 countries and the paper's permanent trade-cost experiment; and
7. the larger multisector input-output extension only after the baseline replication
   clears all accounting and path-quality gates.

## Questions deliberately left for implementation experiments

- Whether the multicountry implementation should use an algebraic kernel block for
  common \(M\), with no free initial offset, or eliminate \(M\) pointwise through world
  resource clearing. It should not use an integrated \(M\) path or reintroduce \(N\)
  redundant marginal-wealth paths under common discounting.
- How quickly the minimum-RKHS solution approaches the independent BVP as the observation
  horizon grows, and how that rate changes with kernel order and length scale.
- Whether the static multicountry equilibrium should be a nested implicit solve or part
  of one larger NLLS residual.
- How aggressively to over-collocate once the DAE-only baseline is verified.
- Whether exact annual-rate conversion or the paper's direct annual rates provide the
  cleaner quantitative comparison. Both should be reported.

These are numerical design choices. They do not alter the economic object fixed above:
initial external assets are states, current accounts are flows, and long-run external
positions remain endogenous outcomes of the initial condition and transition.

## Verification record

An independent adversarial Codex review checked the Hamiltonian, steady state,
wealth-shift identities, boundary count, literature mapping, and RKHS equation count. It
verified that the stable-manifold and present-value tail conditions improve the
independent BVP benchmark over steady-state endpoint pins, including the signs and
coefficients in (19)-(20). Those conditions are not part of the kernel solve. The same
review motivated eliminating the constant \(m\) path from the stepping-stone kernel
system, reducing it from \(4n+2\) to \(3n+2\) unknowns against \(3n\) DAE equations.

The final extra-high-reasoning audit independently rederived equations (4)-(21), the
stable eigenvector, the PV-tail coefficient, and the conventional BVP boundary count. It
found one material error in the proposed multicountry extension: after imposing global
asset clearing, the omitted country budget law cannot determine \(r\), because its interest
terms cancel and it reduces to world resource clearing. Equations (26)-(33) incorporate
the resulting common-marginal-wealth correction. That audit
also added the investment-corner, shock-timing, population-weighting, static-uniqueness,
and no-artificial-stationarity qualifications above.

A fresh nonlinear SciPy solve_bvp check used \(k_0=1\), \(T=80\), and the baseline
calibration. It converged with stable-tail residual \(5.4\times10^{-20}\), tail-corrected
PV residual \(6.2\times10^{-8}\), and a \(1.5\times10^{-6}\) discrepancy between forward
and tail-reconstructed \(a(T)\). The NFA comparative-statics check gave
\(\max_t|\Delta CA_t|=0\), and the baseline stayed interior with
\(\min_t x(t)=0.2963\). The steady-state calculation gives
\(\xi_-=-0.0889010\), \(\xi_+=0.1297230\), with the stable-eigenvector residual at
machine precision.

Claude Code 2.1.211 was invoked through the required Opus model for an external review,
using `cat /tmp/claude-opus-open-economy-review.txt | claude -p --model opus`. The final
authorized call ran for 15 minutes, emitted no stdout, stderr, or partial report, and
ended with status 130 and `Execution error` when stopped at the time limit. Two preceding
five-minute Opus attempts failed the same way. A subsequent unsandboxed one-line Opus
health check succeeded, but a full-file read-only review again timed out after 15 minutes
without output. No lower model was substituted.

The review was therefore split into three self-contained, read-only Opus audits that did
not use repository tools. The one-country audit returned `CORRECT WITH REVISIONS`: it
verified every Hamiltonian sign, steady-state identity, annuity formula, NFA comparative
static, and the hysteresis-versus-indeterminacy distinction; its requested assumptions
(capital TVC, interiority, positive finite wealth, \(r=\rho\), and saddle-path
selection) are stated or tested above. The tail/NLLS audit found no definite error in the
linearization, \(\mathcal G\), or the PV residual and confirmed that the two benchmark
terminal conditions project out explosive modes without selecting an NFA level. The
subsequent implementation deliberately omits them from the RKHS system so this projection
must instead emerge from the metric.

The multicountry Opus audit proved the omitted-budget cancellation and verified the
one-good index reduction in (28), but found a second material counting error: treating
algebraic \(M(0)\) as a free initial value and adding \(r(T)=\rho\) introduced two
offsetting extra dimensions. The Stage 3 DAE now has \(2N-1\) free
q/relative-wealth constants selected by its RKHS metric; an independent BVP benchmark
would use \(2N-1\) terminal projections, with no terminal restriction on \(r\). This
correction was independently checked by differentiating \(\Phi=0\) to recover (28) and
by counting the reduced differential, unstable, and center dimensions. A separate JAX
automatic-differentiation check at a generic two-country interior point gave a resource
residual of \(2.2\times10^{-16}\), resource-residual drift of
\(4.2\times10^{-17}\), and agreement to \(2.8\times10^{-17}\) between (28) and
\(r=\rho+\gamma\dot{\mathcal R}/\mathcal R\).

## Stage 1 implementation record

The one-country model, independent benchmark, NLLS-gram solver, replication driver, and
figure driver are implemented in the files listed above. The kernel residual contains
only the \(3n\) DAE equations, starts from the zero coordinate, and uses no terminal or
present-value information. The default \(T=160\), \(n=96\) solve uses the public
`solver.solve` interface and converges in five steps. On CPU, the measured warm solve
time is about \(0.05\) seconds after roughly a two-second compile. The maximum unscaled
training residual is \(2.9\times10^{-9}\), the dense-grid residual is
\(3.4\times10^{-4}\), and the maximum normalized path discrepancy from the independent
benchmark is \(6.7\times10^{-4}\).

All three initial-NFA scenarios \(a_0/y^\star\in\{-0.5,0,0.5\}\) pass the same DAE
gates and match the benchmark at the reporting tolerance. Relative to \(a_0=0\), the
numerical errors in \(\Delta c=r^\star\Delta a_0\), capital-path invariance,
q-path invariance, \(\Delta a(t)=\Delta a_0\), and \(\Delta CA(t)=0\) are below
\(1.4\times10^{-9}\), \(3.5\times10^{-10}\), \(2.3\times10^{-8}\),
\(1.5\times10^{-10}\), and \(5.3\times10^{-11}\), respectively.

The horizon experiment is part of the result. With the same 81 active-region points and
an increasingly long sparse tail, the maximum benchmark discrepancy falls from
\(2.71\times10^{-1}\) at \(T=80\), to \(1.10\times10^{-2}\) at \(T=120\), to
\(6.62\times10^{-4}\) at \(T=160\), and \(2.66\times10^{-4}\) at \(T=200\).
The \(T=160\) and \(T=200\) reported paths differ by at most
\(6.7\times10^{-4}\). A uniform 321-point density check has
\(5.9\times10^{-4}\) benchmark discrepancy. Varying the metric shift from
\(10^{-8}\) to \(10^{-6}\) leaves the default selection essentially unchanged;
kernel order and length scale remain substantive inductive-bias choices and are reported,
not tuned using terminal residuals.

A regression case at \(T=80\) deliberately has an asset-TVC diagnostic above \(0.1\) and
does not match the economic benchmark, yet remains DAE-accepted. This locks in the
distinction between ex-post economic verification and numerical rejection.

The sequential NN uses the same DAE-only principle in state space. Across seeds 0--2,
the default RK4 policy converges in 13--15 LM steps, has maximum train and validation
residuals below \(8.6\times10^{-6}\), and has maximum normalized BVP discrepancy below
\(7.1\times10^{-4}\) over the 50-year reporting window. The baseline warm solve is about
0.34 seconds after a roughly seven-second CPU compile. The NFA sweep leaves the learned
capital, investment, and current-account policies numerically identical and satisfies
\(\Delta a_t=\Delta a_0\) and \(\Delta c=r^\star\Delta a_0\) to machine precision.
Initial capital values 2 and 7 also pass, with BVP discrepancies below \(10^{-3}\).

A deliberately short year-50 diagnostic still has an ex-post asset TVC around 0.67, yet
the same solution is DAE-accepted and matches the reporting-window benchmark. This is a
numerical regression test that TVC and no-Ponzi diagnostics do not enter the NN residual,
stopping criterion, or rejection logic.

## Stage 3 implementation record

The two-country one-good implementation eliminates common marginal wealth exactly and
collocates only the five ODE blocks for
\((k_1,k_2,q_1,q_2,a_1)\). With \(n\) points, the unknown vector contains five
length-\(n\) kernel-coordinate blocks plus the two initial q offsets and
\(\log\omega_2\): \(5n+3\) unknowns against \(5n\) equations. The metric is

\[
\operatorname{blockdiag}(K,K,K,K,K,0_3)+\epsilon I.
\]

No terminal projection, lifetime budget, TVC, no-Ponzi condition, debt target, or
benchmark quantity appears in the kernel residual, initialization, stopping criterion,
or rejection logic. The independent BVP separately uses three terminal restrictions
from the unstable projection of the full coupled Jacobian. At the limiting allocation,
that Jacobian has two stable, three unstable, and one center root, exactly matching the
dimension count above.

At the default \(T=160,n=96\), the Matérn-5/2 kernel converges in four LM steps. The
measured warm solve is about 0.08 seconds after a roughly 2.7-second compile. The baseline
maximum training residual is \(1.5\times10^{-8}\), the dense-grid residual is
\(8.0\times10^{-6}\), and the maximum reporting-window discrepancy from the independent
BVP is \(2.8\times10^{-5}\). Resource clearing, global asset clearing, the reconstructed
country budget, and both interest-rate identities hold at their corresponding numerical
tolerances.

For first-country initial NFA \(a_{1,0}\in\{-0.5,0,0.5\}\), all three DAE solves pass.
The selected \(\log\omega_2\) values agree with the conventional BVP to about
\(1.2\times10^{-7}\), and opposite initial NFA positions select opposite consumption
weights and distinct long-run NFA positions. Asymmetric initial capital and unequal
country sizes/productivities also pass with BVP discrepancies below \(7\times10^{-5}\).

The horizon sweep again treats economic selection as an outcome. With initial debt, a
year-40 kernel solve is DAE-accepted even though its asset-TVC diagnostic is about 0.24
and its BVP discrepancy is about 0.14. At years 80, 120, and 160, the BVP discrepancy
falls to roughly \(1.3\times10^{-2}\), \(7.3\times10^{-4}\), and
\(5.0\times10^{-5}\), respectively. This regression prevents ex-post solvency or
benchmark information from becoming a hidden acceptance gate.
