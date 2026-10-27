from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx

from .._args import CuckerSmaleArgs
from .._custom_control import CustomControl

"""
Implementation of (stochastic) Cucker-Smale particle systems on the torus or an 
unbounded domain

Coupled SDEs:
-------------

    dx_i = v_i dt
    dv_i = N^{-1} sum_j a(x_i - x_j)*(v_j - v_i) dt + dN_i

Kernel:
-------------

a(x_i - x_j) = lambda / (1 + ||x_i - x_j|| ** 2) ** r where lambda, r geq 0

Noises:
---------------

1. dN_i = sigma * dB_i where B_i are i.i.d Brownian motions

2. dN_i = sigma * (v_i - vbar(0))dB where B is a standard Brownian motion, and 
vbar(0) is the initial flocking velocity. This is because vbar is conserved 
under the dynamics

3. dN_i = sigma * (v_i - vbar(t))dB_i where B_i are i.i.d Brownian motions, and 
vbar(t) is the flocking velocity at time t. This is because vbar is not 
conserved under the dynamics.

"""


def _dist_periodic(xi, xj, L):
    """
    Periodic distance between xi and xj torus of length L
    """
    delta = xi - xj - L * jnp.round((xi - xj) / L)
    return jnp.linalg.norm(delta)


def _dist_euclidean(xi, xj, L):
    """
    Euclidean distance between xi and xj
    """
    return jnp.linalg.norm(xi - xj)


def _cs_kernel(x, lambda_, r):
    """
    Standard Cucker-Smale kernel
    """
    return lambda_ / (1 + x**2) ** r


def _get_full_drift(dist_fn: Callable):
    """
    Helper to compute the drift of the SDE for a given choice of distance
    """

    def _drift(t, state, args):
        xs, vs = state
        L = args.L
        lambda_ = args.lambda_
        r = args.r

        def _pairwise_force(xi, vi, xj, vj):
            dist = dist_fn(xi, xj, L)
            return _cs_kernel(dist, lambda_, r) * (vj - vi)

        dxs = vs

        dvs = jax.vmap(
            lambda xi, vi: jnp.mean(
                jax.vmap(lambda xj, vj: _pairwise_force(xi, vi, xj, vj))(xs, vs), axis=0
            )
        )(xs, vs)

        return dxs, dvs

    return _drift


def _constant_drift(t, state, args):
    """
    Helper to compute the drift of the SDE when the kernel is constant
    """
    xs, vs = state
    C_a = args.C_a

    dxs = vs
    vbar = jnp.mean(vs, axis=0)
    dvs = -C_a * (vs - vbar)
    return dxs, dvs


def _additive_diagonal_noise(t, state, args: CuckerSmaleArgs):
    """
    Defines an additive diagonal noise
    """
    xs, vs = state
    vs_shape = vs.shape
    sigma = args.sigma
    return lx.DiagonalLinearOperator(sigma * jnp.ones(vs_shape))


def _vanishing_diagonal_noise_constant(t, state, args: CuckerSmaleArgs):
    """
    Defines a mutliplicative diagonal noise that vanishes in the flocking
    regime where vbar = vbar(0)
    """
    xs, vs = state
    sigma = args.sigma
    vbar0 = args.vbar0
    return lx.DiagonalLinearOperator(sigma * (vs - vbar0))


def _vanishing_diagonal_noise_varying(t, state, args: CuckerSmaleArgs):
    """
    Defines a mutliplicative diagonal noise that vanishes in the flocking
    regime where vbar = vbar(t)
    """
    xs, vs = state
    vbar = jnp.mean(vs, axis=0)
    sigma = args.sigma
    return lx.DiagonalLinearOperator(sigma * (vs - vbar))


def _build_sde_term(diffusion, bm, x_shape):
    """
    Given a noise type and Brownian motion, build the associated SDE term
    """
    return CustomControl(diffusion, bm, x_shape)
