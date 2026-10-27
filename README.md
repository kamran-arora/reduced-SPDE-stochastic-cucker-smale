# On reduced SPDE models for stochastic Cucker-Smale flocking dynamics

Reference implementation for the paper *On reduced SPDE models for stochastic Cucker-Smale flocking dynamics* ([arXiv](/)). 

```flocking/``` is the library; ```experiments/``` contains the scripts that reproduce the results in the paper.

The `Field` and `Operator` abstractions introduced in this code have since been developed into the [SPDEax](https://github.com/kamran-arora/SPDEax) library providing "composable spatial discretisations and operators for solving SPDEs with Diffrax".

## Installation

To install the package run
```
git clone https://github.com/kamran-arora/reduced-SPDE-stochastic-cucker-smale.git
cd flocking
pip install -e .
```

Requires Python >= 3.10

## Quick example

We give a small example of solving a particle system and associated reduced SPDE in $d=2$. 

Import required libraries:

```Python
import jax
import diffrax as dfx
import jax.random as jr
import jax.numpy as jnp
import flocking as fl

jax.config.update("jax_enable_x64", True)
```

We first create two helper functions to generate random initial data:

```Python
def make_initial_particles(N, key):
    xmin, xmax = -50.0, 50.0
    vmin, vmax = -20.0, 20.0
    k1, k2 = jr.split(key, 2)
    xs0 = jr.uniform(key=k1, shape=(N, 2), minval=xmin, maxval=xmax)
    vs0 = jr.uniform(key=k2, shape=(N, 2), minval=vmin, maxval=vmax)
    return xs0, vs0


def make_spde_initial_state(particles, grid, kappa, x0, x_final, Lx, Ly):
    rho0, j0 = fl.density_from_particles_2d(particles, grid, kappa, Lx, Ly)
    return fl.ScalarField(x0, x_final, rho0), fl.VectorField(x0, x_final, j0)
```

We then setup a grid, generate the initial data, create our 'Cucker-Smale' arguments and choose a set of times to save the solution at:

```Python
ic_key, run_key = jr.split(jr.key(123))
N = 100

x0 = (-50.0, -50.0)
x_final = (50.0, 50.0)
n = (64, 64)
grid = fl.make_grid(x0, x_final, n)

xs0, vs0 = make_initial_particles(N, ic_key)
rho0, j0 = make_spde_initial_state((xs0, vs0), grid, 50.0, x0, x_final, 100.0, 100.0)
vbar0 = jnp.mean(vs0, axis=0)

args = fl.CuckerSmaleArgs(
    L=jnp.array([100.0, 100.0]),
    lambda_=50,
    r=0.5,
    sigma=1.0,
    vbar0=vbar0,
)

t0 = 0.0
t1 = 1.0
dt0 = 0.01
ts = jnp.linspace(t0, t1, int((t1 - t0) / dt0) + 1)
```

Finally, we just call the appropriate 'solve' functions:

```Python
xs, vs = fl.solve_CS_SDE(
    t0=t0,
    t1=t1,
    dt0=dt0,
    saveat=dfx.SaveAt(ts=ts),
    state0=(xs0, vs0),
    noise_type="vanishing-common",
    key=run_key,
    args=args,
)

rhos, js = fl.solve_spde_common_noise(
    t0=t0,
    t1=t1,
    dt0=dt0,
    saveat=dfx.SaveAt(ts=ts),
    state0=(rho0, j0),
    args=args,
    dim=2,
    kernel_type="full",
    key=run_key,
)
```