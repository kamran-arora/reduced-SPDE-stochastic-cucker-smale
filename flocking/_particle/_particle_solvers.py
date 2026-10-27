from typing import Tuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._args import CuckerSmaleArgs
from .._custom_solvers import PeriodicSolver
from ._particle_helpers import (
    _additive_diagonal_noise,
    _build_sde_term,
    _constant_drift,
    _dist_euclidean,
    _dist_periodic,
    _get_full_drift,
    _vanishing_diagonal_noise_constant,
    _vanishing_diagonal_noise_varying,
)


@eqx.filter_jit
def solve_CS_ODE(
    t0: float,
    t1: float,
    dt0: float,
    saveat: dfx.SaveAt,
    state0: Tuple[Array, Array],
    args: CuckerSmaleArgs,
    domain="periodic",
    kernel="full",
):
    """
    Solve the Cucker-Smale ODE system

    **Args**

    - t0: initial time
    - t1: final time
    - dt0: timestep
    - saveat: times to save solution at
    - state0: initial state
    - args: Cucker-Smale arguments
    - domain: one of
        - 'periodic'
        - 'unbounded'
    - kernel: one of
        - 'constant'
        - 'full'
    """
    # compute initial alignment velocity
    xs0, vs0 = state0
    vbar0 = jnp.mean(vs0, axis=0)
    args = eqx.tree_at(lambda x: x.vbar0, args, vbar0)
    # setup solver
    if kernel == "constant":
        term = dfx.ODETerm(_constant_drift)
    elif kernel == "full":
        if domain == "periodic":
            term = dfx.ODETerm(_get_full_drift(_dist_periodic))
        elif domain == "unbounded":
            term = dfx.ODETerm(_get_full_drift(_dist_euclidean))
    if domain == "periodic":
        solver = PeriodicSolver(dfx.Euler(), args.L)
    elif domain == "unbounded":
        solver = dfx.Euler()
    # solve
    sol = dfx.diffeqsolve(
        terms=term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state0,
        args=args,
        saveat=saveat,
        max_steps=10**6,
    )
    return sol.ys


@eqx.filter_jit
def solve_CS_SDE(
    t0: float,
    t1: float,
    dt0: float,
    saveat: dfx.SaveAt,
    state0: Tuple[Array, Array],
    noise_type: str,
    key: Key,
    args: CuckerSmaleArgs,
    domain="periodic",
    kernel="full",
):
    """
    Solve the Cucker-Smale SDE system for a specified noise_type

    **Args**

    - t0: initial time
    - t1: final time
    - dt0: timestep
    - saveat: times to save solution at
    - state0: initial state
    - noise_type: one of
        - 'iid'
        - 'vanishing-common'
        - 'vanishing-iid'
    - key: jax.random.key
    - args: Cucker-Smale arguments
    - domain: one of
        - 'periodic'
        - 'unbounded'
    - kernel: one of
        - 'constant'
        - 'full'
    """
    # compute alignment velocity
    xs0, vs0 = state0
    x_shape = jax.ShapeDtypeStruct(shape=xs0.shape, dtype=xs0.dtype)
    vs_shape = vs0.shape
    d = vs_shape[-1] if len(vs_shape) > 1 else 1
    vbar0 = jnp.mean(vs0, axis=0)
    args = eqx.tree_at(lambda x: x.vbar0, args, vbar0)
    # get noise type, and generate associated Brownian motion
    if noise_type == "iid":
        diffusion = _additive_diagonal_noise
        bm = dfx.UnsafeBrownianPath(shape=vs_shape, key=key)
    elif noise_type == "vanishing-common":
        diffusion = _vanishing_diagonal_noise_constant
        bm = dfx.UnsafeBrownianPath(shape=(d,), key=key)
    elif noise_type == "vanishing-iid":
        diffusion = _vanishing_diagonal_noise_varying
        bm = dfx.UnsafeBrownianPath(shape=vs_shape, key=key)
    else:
        raise ValueError("Noise type not recognised")
    # build sde term
    sde_term = _build_sde_term(diffusion, bm, x_shape)
    # setup ode term
    if kernel == "constant":
        ode_term = dfx.ODETerm(_constant_drift)
    elif kernel == "full":
        if domain == "periodic":
            ode_term = dfx.ODETerm(_get_full_drift(_dist_periodic))
        elif domain == "unbounded":
            ode_term = dfx.ODETerm(_get_full_drift(_dist_euclidean))
    # setup solver
    if domain == "periodic":
        solver = PeriodicSolver(dfx.Euler(), args.L)
    elif domain == "unbounded":
        solver = dfx.Euler()
    term = dfx.MultiTerm(ode_term, sde_term)
    # solve
    sol = dfx.diffeqsolve(
        terms=term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state0,
        args=args,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),  # explicitly set since I'm using UnsafeBrownianPath
        max_steps=10**6,
    )
    return sol.ys
