from typing import Tuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Key

from .._args import CuckerSmaleArgs
from .._custom_control import (
    CustomControl,
    CustomIIDControl,
    CustomVIIDControl,
    CustomVIIDControlNoVbar,
)
from .._operator import Convolution, ConvolutionPadded, Divergence, Gradient, Laplacian
from .._spatial import AbstractField, ScalarField, VectorField
from ._spde_noise import (
    IIDNoise1D,
    IIDNoiseND,
    VanishingNoise,
    VIIDNoise1D,
    VIIDNoise1DNoVbar,
    VIIDNoiseND,
)


@eqx.filter_jit
def solve_spde_common_noise(
    t0: float,
    t1: float,
    dt0: float,
    saveat: dfx.SaveAt,
    state0: tuple[AbstractField, AbstractField],
    args: CuckerSmaleArgs,
    dim: int,
    kernel_type: str,
    key: Key,
    conv_type="circular",
):
    """
    Solve the reduced Cucker-Smale SPDE with vanishing environmental noise in arbitrary dimension over the torus.

    The spatial discretisation is centered finite differences, and the timestepping is Euler-Maruyama.

    **Arguments**:
    - t0: initial time
    - t1: final time
    - dt0: timestep
    - saveat: times to save solution at
    - state0: a PyTree of AbstractFields
    - args: CuckerSmaleArgs object
    - dim: spatial dimension of the SPDE
    - kernel_type: 'constant' or 'full'
    - key: jax.random.key

    **Returns**:
    - The ys attribute of a Diffrax Solution object
    """

    # extract grid information from initial data
    rho0, j0 = state0
    x0 = rho0.x0
    x_final = rho0.x_final
    n = rho0.vals.shape
    hs = rho0.hs

    # true convolution kernel
    def _kernel(x, lambda_, r):
        # return lambda_ / (1 + jnp.sum(4 * jnp.sin(x / 2) ** 2)) ** r
        return lambda_ / (1 + jnp.sum(x**2, axis=0)) ** r

    if conv_type == "circular":
        _conv = Convolution(x0, x_final, n, _kernel, (args.lambda_, args.r))
    elif conv_type == "padded":
        _conv = ConvolutionPadded(x0, x_final, n, _kernel, (args.lambda_, args.r))
    else:
        raise ValueError("Unrecognised convolution type")

    # define non-linearity
    if kernel_type == "constant":
        F_j = lambda rho, j: -args.C_a * (j - args.vbar0 * rho)
    elif kernel_type == "full":
        F_j = lambda rho, j: rho * _conv(j) - j * _conv(rho)
    else:
        raise ValueError("Unrecognised kernel type")

    # define linear differential operators
    if dim == 1:
        A_rho = -Gradient()
        lplc = Laplacian()
        A_j = -(args.vbar0**2) * Gradient()

        def _vf(t, state, args: CuckerSmaleArgs):
            rho, j = state
            nu = jnp.abs(args.vbar0) * hs[0] / 2
            return A_rho(j) + nu * lplc(rho), A_j(rho) + nu * lplc(j) + F_j(rho, j)
    elif dim > 1:
        A_rho = -Divergence()
        A_j = -args.vbar0 * Divergence()
        lplc = Laplacian()

        def _vf(t, state, args: CuckerSmaleArgs):
            rho, j = state
            nu = 0.5 * hs[0] * jnp.linalg.norm(args.vbar0)
            j_reg_components = []
            for m in range(dim):
                j_m = ScalarField(j.x0, j.x_final, j.vals[..., m])
                j_reg_m = nu * lplc(j_m)
                j_reg_components.append(j_reg_m.vals)
            j_reg_vals = jnp.stack(j_reg_components, axis=-1)
            j_reg = VectorField(j.x0, j.x_final, j_reg_vals)
            return A_rho(j) + nu * lplc(rho), A_j(j) + j_reg + F_j(rho, j)
    else:
        raise ValueError("Unrecognised dimension")

    # setup the noise
    def _diffusion(t, state, args: CuckerSmaleArgs):
        return VanishingNoise(state, args)

    _zero_scalar_field = ScalarField(x0, x_final, jnp.zeros_like(rho0.vals))
    _x_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_scalar_field)
    _bm = dfx.UnsafeBrownianPath(shape=(dim,), key=key)
    _ode_term = dfx.ODETerm(_vf)
    _sde_term = CustomControl(_diffusion, _bm, _x_shape)
    _term = dfx.MultiTerm(_ode_term, _sde_term)

    sol = dfx.diffeqsolve(
        terms=_term,
        solver=dfx.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state0,
        args=args,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
        max_steps=10**6,
    )
    return sol.ys


@eqx.filter_jit
def solve_spde_viid(
    t0: float,
    t1: float,
    dt0: float,
    saveat: dfx.SaveAt,
    state0: tuple[AbstractField, AbstractField, AbstractField, float],
    args: CuckerSmaleArgs,
    dim: int,
    kernel_type: str,
    N: int,
    key: Key,
    conv_type="circular",
):
    """
    Solve the reduced Cucker-Smale SPDE with vanishing i.i.d noise over the torus.

    The spatial discretisation is centered finite differences, and the timestepping is Euler-Maruyama.

    **Arguments**:
    - t0: initial time
    - t1: final time
    - dt0: timestep
    - saveat: times to save solution at
    - state0: a PyTree of AbstractFields
    - args: CuckerSmaleArgs object
    - kernel_type: 'constant' or 'full'
    - N: number of particles
    - key: jax.random.key

    **Returns**:
    - The ys attribute of a Diffrax Solution object
    """

    # extract grid information from initial data
    rho0, j0, k0, vbar0 = state0
    x0 = rho0.x0
    x_final = rho0.x_final
    n = rho0.vals.shape
    hs = rho0.hs
    cell_vol = jnp.prod(jnp.array(hs))
    spatial_axes = tuple(range(dim))

    # true convolution kernel
    def _kernel(x, lambda_, r):
        # return lambda_ / (1 + jnp.sum(4 * jnp.sin(x / 2) ** 2)) ** r
        return lambda_ / (1 + jnp.sum(x**2, axis=0)) ** r

    if conv_type == "circular":
        _conv = Convolution(x0, x_final, n, _kernel, (args.lambda_, args.r))
    elif conv_type == "padded":
        _conv = ConvolutionPadded(x0, x_final, n, _kernel, (args.lambda_, args.r))
    else:
        raise ValueError("Unrecognised convolution type")

    # define non-linearity
    if kernel_type == "constant":

        def F_j(rho, j, k, vbar):
            return -args.C_a * (j - vbar * rho)

        def F_k(rho, j, k, vbar):
            # return (args.sigma**2 * (1 - 1 / N) ** 2 - 2 * args.C_a) * k
            return (
                args.sigma**2 * (1 - 2 / N) - 2 * args.C_a
            ) * k + args.sigma**2 * rho * (
                jnp.sum(k.vals, axis=spatial_axes) * cell_vol
            ) * (1 / N)
    elif kernel_type == "full":

        def F_j(rho, j, k, vbar):
            return rho * _conv(j) - j * _conv(rho)

        def F_k(rho, j, k, vbar):
            _term1 = 2 * (j - vbar * rho) * _conv(j - vbar * rho)
            _term2 = (args.sigma**2 * (1 - 2 / N)) * k + args.sigma**2 * rho * (
                jnp.sum(k.vals, axis=spatial_axes) * cell_vol
            ) * (1 / N)
            _term3 = -2 * _conv(rho) * k
            return _term1 + _term2 + _term3
    else:
        raise ValueError("Unrecognised kernel type")

    if dim == 1:
        A_rho = -Gradient()
        A_j = -Gradient()
        A_k = -Gradient()
        lplc = Laplacian()

        def _vf(t, state, args: CuckerSmaleArgs):
            rho, j, k, vbar = state
            nu = 0.5 * hs[0] * jnp.linalg.norm(vbar)
            return (
                A_rho(j) + nu * lplc(rho),
                vbar**2 * A_j(rho) + nu * lplc(j) + F_j(rho, j, k, vbar),
                vbar * A_k(k) + nu * lplc(k) + F_k(rho, j, k, vbar),
                0.0,
            )
    elif dim > 1:
        A_rho = -Divergence()
        A_j = -Divergence()
        A_k = -Divergence()
        lplc = Laplacian()

        def _vf(t, state, args: CuckerSmaleArgs):
            rho, j, k, vbar = state
            nu = 0.5 * hs[0] * jnp.linalg.norm(vbar)
            j_reg_components = []
            k_reg_components = []
            for m in range(dim):
                j_m = ScalarField(j.x0, j.x_final, j.vals[..., m])
                k_m = ScalarField(k.x0, k.x_final, k.vals[..., m])
                j_reg_m = nu * lplc(j_m)
                k_reg_m = nu * lplc(k_m)
                j_reg_components.append(j_reg_m.vals)
                k_reg_components.append(k_reg_m.vals)
            j_reg_vals = jnp.stack(j_reg_components, axis=-1)
            k_reg_vals = jnp.stack(k_reg_components, axis=-1)
            j_reg = VectorField(j.x0, j.x_final, j_reg_vals)
            k_reg = VectorField(k.x0, k.x_final, k_reg_vals)
            return (
                A_rho(j) + nu * lplc(rho),
                vbar * A_j(j) + j_reg + F_j(rho, j, k, vbar),
                vbar * A_k(k) + k_reg + F_k(rho, j, k, vbar),
                0.0,
            )

    # setup the noise
    if dim == 1:

        def _diffusion(t, state, args: CuckerSmaleArgs):
            return VIIDNoise1D(state, N, args)

        _zero_scalar_field = ScalarField(x0, x_final, jnp.zeros_like(rho0.vals))
        _x_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_scalar_field)
        _bm = dfx.UnsafeBrownianPath(shape=(n[0],), key=key)
        _ode_term = dfx.ODETerm(_vf)
        _sde_term = CustomVIIDControl(_diffusion, _bm, _x_shape, _x_shape)
        _term = dfx.MultiTerm(_ode_term, _sde_term)

    elif dim > 1:

        def _diffusion(t, state, args: CuckerSmaleArgs):
            return VIIDNoiseND(state, N, args)

        _zero_scalar_field = ScalarField(x0, x_final, jnp.zeros_like(rho0.vals))
        _x_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_scalar_field)
        _zero_vector_field = VectorField(x0, x_final, jnp.zeros_like(k0.vals))
        _k_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_vector_field)
        _bm = dfx.UnsafeBrownianPath(shape=(*n, dim), key=key)
        _ode_term = dfx.ODETerm(_vf)
        _sde_term = CustomVIIDControl(_diffusion, _bm, _x_shape, _k_shape)
        _term = dfx.MultiTerm(_ode_term, _sde_term)

    sol = dfx.diffeqsolve(
        terms=_term,
        solver=dfx.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state0,
        args=args,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
        max_steps=10**6,
    )
    return sol.ys


@eqx.filter_jit
def solve_spde_iid(
    t0: float,
    t1: float,
    dt0: float,
    saveat: dfx.SaveAt,
    state0: tuple[AbstractField, AbstractField, float],
    args: CuckerSmaleArgs,
    dim: int,
    kernel_type: str,
    N: int,
    key: Key,
    conv_type="circular",
):
    """
    Solve the reduced Cucker-Smale SPDE with vanishing i.i.d noise over the torus.

    The spatial discretisation is centered finite differences, and the timestepping is Euler-Maruyama.

    **Arguments**:
    - t0: initial time
    - t1: final time
    - dt0: timestep
    - saveat: times to save solution at
    - state0: a PyTree of AbstractFields
    - args: CuckerSmaleArgs object
    - kernel_type: 'constant' or 'full'
    - N: number of particles
    - key: jax.random.key

    **Returns**:
    - The ys attribute of a Diffrax Solution object
    """

    # extract grid information from initial data
    rho0, j0, vbar0 = state0
    x0 = rho0.x0
    x_final = rho0.x_final
    n = rho0.vals.shape

    # true convolution kernel
    def _kernel(x, lambda_, r):
        # return lambda_ / (1 + jnp.sum(4 * jnp.sin(x / 2) ** 2)) ** r
        return lambda_ / (1 + jnp.sum(x**2, axis=0)) ** r

    if conv_type == "circular":
        _conv = Convolution(x0, x_final, n, _kernel, (args.lambda_, args.r))
    elif conv_type == "padded":
        _conv = ConvolutionPadded(x0, x_final, n, _kernel, (args.lambda_, args.r))
    else:
        raise ValueError("Unrecognised convolution type")

    # define non-linearity
    if kernel_type == "constant":

        def F_j(rho, j, vbar):
            return -args.C_a * (j - vbar * rho)

    elif kernel_type == "full":

        def F_j(rho, j, vbar):
            return rho * _conv(j) - j * _conv(rho)

    else:
        raise ValueError("Unrecognised kernel type")

    if dim == 1:
        A_rho = -Gradient()
        A_j = -Gradient()

        def _vf(t, state, args: CuckerSmaleArgs):
            rho, j, vbar = state
            return (
                A_rho(j),
                vbar**2 * A_j(rho) + F_j(rho, j, vbar),
                0.0,
            )
    elif dim > 1:
        A_rho = -Divergence()
        A_j = -Divergence()

        def _vf(t, state, args: CuckerSmaleArgs):
            rho, j, vbar = state
            return (
                A_rho(j),
                vbar * A_j(j) + F_j(rho, j, vbar),
                0.0,
            )

    # setup the noise
    if dim == 1:

        def _diffusion(t, state, args: CuckerSmaleArgs):
            return IIDNoise1D(state, N, args)

        _zero_scalar_field = ScalarField(x0, x_final, jnp.zeros_like(rho0.vals))
        _x_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_scalar_field)
        _bm = dfx.UnsafeBrownianPath(shape=(n[0],), key=key)
        _ode_term = dfx.ODETerm(_vf)
        _sde_term = CustomIIDControl(_diffusion, _bm, _x_shape)
        _term = dfx.MultiTerm(_ode_term, _sde_term)

    elif dim > 1:

        def _diffusion(t, state, args: CuckerSmaleArgs):
            return IIDNoiseND(state, N, args)

        _zero_scalar_field = ScalarField(x0, x_final, jnp.zeros_like(rho0.vals))
        _x_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_scalar_field)
        _bm = dfx.UnsafeBrownianPath(shape=(*n, dim), key=key)
        _ode_term = dfx.ODETerm(_vf)
        _sde_term = CustomIIDControl(_diffusion, _bm, _x_shape)
        _term = dfx.MultiTerm(_ode_term, _sde_term)

    sol = dfx.diffeqsolve(
        terms=_term,
        solver=dfx.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state0,
        args=args,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
        max_steps=10**6,
    )
    return sol.ys


################################################################################
################################################################################
################################################################################


@eqx.filter_jit
def solve_spde_viid_NEW(
    t0: float,
    t1: float,
    dt0: float,
    saveat: dfx.SaveAt,
    state0: tuple[AbstractField, AbstractField, AbstractField, float],
    args: CuckerSmaleArgs,
    dim: int,
    kernel_type: str,
    N: int,
    key: Key,
    conv_type="circular",
):
    """
    Solve the reduced Cucker-Smale SPDE with vanishing i.i.d noise over the torus.

    THIS FUNCTION USES THE SPDE DERIVED FROM WRITING

        v_i^2 = (v_i - vbar + vbar)^2


    !!! ONLY SUPPORTS d=1 AT THE MOMENT !!!

    The spatial discretisation is centered finite differences, and the timestepping is Euler-Maruyama.

    **Arguments**:
    - t0: initial time
    - t1: final time
    - dt0: timestep
    - saveat: times to save solution at
    - state0: a PyTree of AbstractFields
    - args: CuckerSmaleArgs object
    - kernel_type: 'constant' or 'full'
    - N: number of particles
    - key: jax.random.key

    **Returns**:
    - The ys attribute of a Diffrax Solution object
    """

    # extract grid information from initial data
    rho0, j0, k0, vbar0 = state0
    x0 = rho0.x0
    x_final = rho0.x_final
    n = rho0.vals.shape
    hs = rho0.hs
    cell_vol = jnp.prod(jnp.array(hs))
    spatial_axes = tuple(range(dim))

    # true convolution kernel
    def _kernel(x, lambda_, r):
        # return lambda_ / (1 + jnp.sum(4 * jnp.sin(x / 2) ** 2)) ** r
        return lambda_ / (1 + jnp.sum(x**2, axis=0)) ** r

    if conv_type == "circular":
        _conv = Convolution(x0, x_final, n, _kernel, (args.lambda_, args.r))
    elif conv_type == "padded":
        _conv = ConvolutionPadded(x0, x_final, n, _kernel, (args.lambda_, args.r))
    else:
        raise ValueError("Unrecognised convolution type")

    # define non-linearity
    if kernel_type == "constant":

        def F_j(rho, j, k, vbar):
            return -args.C_a * (j - vbar * rho)

        def F_k(rho, j, k, vbar):
            # return (args.sigma**2 * (1 - 1 / N) ** 2 - 2 * args.C_a) * k
            return (
                args.sigma**2 * (1 - 2 / N) - 2 * args.C_a
            ) * k + args.sigma**2 * rho * (
                jnp.sum(k.vals, axis=spatial_axes) * cell_vol
            ) * (1 / N)
    elif kernel_type == "full":

        def F_j(rho, j, k, vbar):
            return rho * _conv(j) - j * _conv(rho)

        def F_k(rho, j, k, vbar):
            _term1 = 2 * (j - vbar * rho) * _conv(j - vbar * rho)
            _term2 = (args.sigma**2 * (1 - 2 / N)) * k + args.sigma**2 * rho * (
                jnp.sum(k.vals, axis=spatial_axes) * cell_vol
            ) * (1 / N)
            _term3 = -2 * _conv(rho) * k
            return _term1 + _term2 + _term3
    else:
        raise ValueError("Unrecognised kernel type")

    if dim == 1:
        A_rho = -Gradient()
        A_j = -Gradient()
        A_k = -Gradient()
        lplc = Laplacian()

        def _vf(t, state, args: CuckerSmaleArgs):
            rho, j, k, vbar = state
            nu = 0.5 * hs[0] * jnp.linalg.norm(vbar)
            return (
                A_rho(j) + nu * lplc(rho),
                vbar**2 * A_j(rho)
                + A_j(k)
                + nu * lplc(j)
                + 2 * vbar * A_k(j - vbar * rho)
                + F_j(rho, j, k, vbar),
                vbar * A_k(k) + nu * lplc(k) + F_k(rho, j, k, vbar),
                0.0,
            )
    elif dim > 1:
        raise NotImplementedError
        # A_rho = -Divergence()
        # A_j = -Divergence()
        # A_k = -Divergence()

        # def _vf(t, state, args: CuckerSmaleArgs):
        #     rho, j, k, vbar = state
        #     return (
        #         A_rho(j),
        #         vbar * A_j(j) + F_j(rho, j, k, vbar),
        #         vbar * A_k(k) + F_k(rho, j, k, vbar),
        #         0.0,
        #     )

    # setup the noise
    if dim == 1:

        def _diffusion(t, state, args: CuckerSmaleArgs):
            return VIIDNoise1D(state, N, args)

        _zero_scalar_field = ScalarField(x0, x_final, jnp.zeros_like(rho0.vals))
        _x_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_scalar_field)
        _bm = dfx.UnsafeBrownianPath(shape=(n[0],), key=key)
        _ode_term = dfx.ODETerm(_vf)
        _sde_term = CustomVIIDControl(_diffusion, _bm, _x_shape, _x_shape)
        _term = dfx.MultiTerm(_ode_term, _sde_term)

    elif dim > 1:
        raise NotImplementedError

        # def _diffusion(t, state, args: CuckerSmaleArgs):
        #     return VIIDNoiseND(state, N, args)

        # _zero_scalar_field = ScalarField(x0, x_final, jnp.zeros_like(rho0.vals))
        # _x_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_scalar_field)
        # _zero_vector_field = VectorField(x0, x_final, jnp.zeros_like(k0.vals))
        # _k_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_vector_field)
        # _bm = dfx.UnsafeBrownianPath(shape=(*n, dim), key=key)
        # _ode_term = dfx.ODETerm(_vf)
        # _sde_term = CustomVIIDControl(_diffusion, _bm, _x_shape, _k_shape)
        # _term = dfx.MultiTerm(_ode_term, _sde_term)

    sol = dfx.diffeqsolve(
        terms=_term,
        solver=dfx.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state0,
        args=args,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
        max_steps=10**6,
    )
    return sol.ys


################################################################################################################################################################################################################################################

# DO NOT USE - VERIFIES THAT VIID WITH CONSTANT VBAR DOESN'T WORK


@eqx.filter_jit
def solve_spde_viid_1d_no_vbar(
    t0: float,
    t1: float,
    dt0: float,
    saveat: dfx.SaveAt,
    state0,
    args: CuckerSmaleArgs,
    kernel_type: str,
    N,
    key,
    conv_type="circular",
):
    """
    Solve the reduced Cucker-Smale SPDE with vanishing i.i.d noise in one dimension over the torus.

    The spatial discretisation is centered finite differences, and the timestepping is Euler-Maruyama.

    **Arguments**:
    - t0: initial time
    - t1: final time
    - dt0: timestep
    - saveat: times to save solution at
    - state0: a PyTree of AbstractFields
    - args: CuckerSmaleArgs object
    - kernel_type: 'constant' or 'full'
    - N: number of particles
    - key: jax.random.key

    **Returns**:
    - The ys attribute of a Diffrax Solution object
    """

    # extract grid information from initial data
    rho0, j0, k0 = state0
    x0 = rho0.x0
    x_final = rho0.x_final
    n = rho0.vals.shape
    h = rho0.hs[0]
    vbar0 = args.vbar0

    # true convolution kernel
    def _kernel(x, lambda_, r):
        # return lambda_ / (1 + jnp.sum(4 * jnp.sin(x / 2) ** 2)) ** r
        return lambda_ / (1 + jnp.sum(x**2, axis=0)) ** r

    if conv_type == "circular":
        _conv = Convolution(x0, x_final, n, _kernel, (args.lambda_, args.r))
    elif conv_type == "padded":
        _conv = ConvolutionPadded(x0, x_final, n, _kernel, (args.lambda_, args.r))
    else:
        raise ValueError("Unrecognised convolution type")

    # define non-linearity
    if kernel_type == "constant":

        def F_j(rho, j, k):
            return -args.C_a * (j - vbar0 * rho)

        def F_k(rho, j, k):
            # return (args.sigma**2 * (1 - 1 / N) ** 2 - 2 * args.C_a) * k
            return (
                args.sigma**2 * (1 - 2 / N) - 2 * args.C_a
            ) * k + args.sigma**2 * rho * (jnp.sum(k.vals) * h) / N
    elif kernel_type == "full":

        def F_j(rho, j, k):
            return rho * _conv(j) - j * _conv(rho)

        def F_k(rho, j, k):
            _term1 = 2 * (j - vbar0 * rho) * _conv(j - vbar0 * rho)
            _term2 = (args.sigma**2 * (1 - 2 / N)) * k + args.sigma**2 * rho * (
                jnp.sum(k.vals) * h
            ) / N
            _term3 = -2 * _conv(rho) * k
            return _term1 + _term2 + _term3
    else:
        raise ValueError("Unrecognised kernel type")

    A_rho = -Gradient()
    A_j = -Gradient()
    A_k = -Gradient(method="upwind")

    def _vf(t, state, args: CuckerSmaleArgs):
        rho, j, k = state
        return (
            A_rho(j),
            vbar0**2 * A_j(rho) + F_j(rho, j, k),
            vbar0 * A_k(k, velocity=vbar0) + F_k(rho, j, k),
        )

    # setup the noise
    def _diffusion(t, state, args: CuckerSmaleArgs):
        return VIIDNoise1DNoVbar(state, N, args)

    _zero_scalar_field = ScalarField(x0, x_final, jnp.zeros_like(rho0.vals))
    _x_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_scalar_field)
    _bm = dfx.UnsafeBrownianPath(shape=(n[0],), key=key)
    _ode_term = dfx.ODETerm(_vf)
    _sde_term = CustomVIIDControlNoVbar(_diffusion, _bm, _x_shape, _x_shape)
    _term = dfx.MultiTerm(_ode_term, _sde_term)

    sol = dfx.diffeqsolve(
        terms=_term,
        solver=dfx.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state0,
        args=args,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
        max_steps=10**6,
    )
    return sol.ys


################################################################################
################################################################################
################################################################################

# DIFFUSIVE REGULARISATION WITH A CUTOFF (ZIMPER ET AL.)

# def _var_phi(y, V, W, h, d):
#     """
#     Smooth cut-off function for the regularisation term.
#     """
#     lower_bound = V * (h**d)
#     upper_bound = (V + 1.0) * (h**d)

#     # Scale y to [0, 1] within the transition region
#     x = jnp.clip((y - lower_bound) / (upper_bound - lower_bound), 0.0, 1.0)

#     # Smoothstep interpolation: 3x^2 - 2x^3
#     smooth_val = W * (x**2) * (3.0 - 2.0 * x)

#     # Apply conditions
#     return jnp.where(y < lower_bound, 0.0, jnp.where(y >= upper_bound, W, smooth_val))

# if DIFFUSIVE_REG:
#     grad_rho = grad(rho)
#     H1_squared_rho = jnp.sum(args.vbar0) ** 2 * jnp.sum(
#         grad_rho.vals**2, axis=-1
#     )
#     H1_squared_j = jnp.zeros_like(H1_squared_rho)
#     for m in range(dim):
#         j_m = ScalarField(j.x0, j.x_final, j.vals[..., m])
#         grad_j_m = grad(j_m)
#         H1_squared_j += jnp.sum(grad_j_m.vals**2, axis=-1)
#     H1_norm = jnp.sqrt(H1_squared_rho + H1_squared_j)

#     # jax.debug.print("Max H1 Norm: {}", jnp.max(H1_norm))
#     # jax.debug.print("Mean H1 Norm: {}", jnp.mean(H1_norm))

#     # smooth cutoff
#     reg_vals = _var_phi(H1_norm, 1e-16, 10.0, rho.hs[0], dim)
#     reg = ScalarField(rho.x0, rho.x_final, reg_vals)

#     # diffusive regularisation
#     rho_reg = div(reg * grad_rho)
#     j_reg_components = []
#     for m in range(dim):
#         j_m = ScalarField(j.x0, j.x_final, j.vals[..., m])
#         grad_j_m = grad(j_m)
#         j_reg_m = div(reg * grad_j_m)
#         j_reg_components.append(j_reg_m.vals)
#     j_reg_vals = jnp.stack(j_reg_components, axis=-1)
#     j_reg = VectorField(j.x0, j.x_final, j_reg_vals)

#     return A_rho(j) + rho_reg, A_j(j) + j_reg + F_j(rho, j)


################################################################################################################################################################################################################################################


# @eqx.filter_jit
# def solve_spde_viid_2d(
#     t0: float,
#     t1: float,
#     dt0: float,
#     saveat: dfx.SaveAt,
#     state0,
#     args: CuckerSmaleArgs,
#     kernel_type: str,
#     N,
#     key,
#     conv_type="circular",
# ):
#     """
#     Solve the reduced Cucker-Smale SPDE with vanishing i.i.d noise in one dimension over the torus.

#     The spatial discretisation is centered finite differences, and the timestepping is Euler-Maruyama.

#     **Arguments**:
#     - t0: initial time
#     - t1: final time
#     - dt0: timestep
#     - saveat: times to save solution at
#     - state0: a PyTree of AbstractFields
#     - args: CuckerSmaleArgs object
#     - kernel_type: 'constant' or 'full'
#     - N: number of particles
#     - key: jax.random.key

#     **Returns**:
#     - The ys attribute of a Diffrax Solution object
#     """

#     # extract grid information from initial data
#     rho0, j0, k0, vbar0 = state0
#     x0 = rho0.x0
#     x_final = rho0.x_final
#     nx, ny = rho0.vals.shape
#     hx, hy = rho0.hs

#     # true convolution kernel
#     def _kernel(x, lambda_, r):
#         # return lambda_ / (1 + jnp.sum(4 * jnp.sin(x / 2) ** 2)) ** r
#         return lambda_ / (1 + jnp.sum(x**2, axis=0)) ** r

#     if conv_type == "circular":
#         _conv = Convolution(x0, x_final, (nx, ny), _kernel, (args.lambda_, args.r))
#     elif conv_type == "padded":
#         _conv = ConvolutionPadded(
#             x0, x_final, (nx, ny), _kernel, (args.lambda_, args.r)
#         )
#     else:
#         raise ValueError("Unrecognised convolution type")

#     # define non-linearity
#     if kernel_type == "constant":

#         def F_j(rho, j, k, vbar):
#             return -args.C_a * (j - vbar * rho)

#         def F_k(rho, j, k, vbar):
#             # return (args.sigma**2 * (1 - 1 / N) ** 2 - 2 * args.C_a) * k
#             return (
#                 args.sigma**2 * (1 - 2 / N) - 2 * args.C_a
#             ) * k + args.sigma**2 * rho * (jnp.sum(k.vals, axis=(0, 1)) * hx * hy) / N
#     elif kernel_type == "full":

#         def F_j(rho, j, k, vbar):
#             return rho * _conv(j) - j * _conv(rho)

#         def F_k(rho, j, k, vbar):
#             _term1 = 2 * (j - vbar * rho) * _conv(j - vbar * rho)
#             _term2 = (args.sigma**2 * (1 - 2 / N)) * k + args.sigma**2 * rho * (
#                 jnp.sum(k.vals, axis=(0, 1)) * hx * hy
#             ) / N
#             _term3 = -2 * _conv(rho) * k
#             return _term1 + _term2 + _term3
#     else:
#         raise ValueError("Unrecognised kernel type")

#     A_rho = -Divergence()
#     A_j = -Divergence()
#     A_k = -Divergence()

#     def _vf(t, state, args: CuckerSmaleArgs):
#         rho, j, k, vbar = state
#         return (
#             A_rho(j),
#             vbar * A_j(j) + F_j(rho, j, k, vbar),
#             vbar * A_k(k) + F_k(rho, j, k, vbar),
#             0.0,
#         )

#     # setup the noise
#     def _diffusion(t, state, args: CuckerSmaleArgs):
#         return VIIDNoise2D(state, N, args)

#     _zero_scalar_field = ScalarField(x0, x_final, jnp.zeros_like(rho0.vals))
#     _x_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_scalar_field)
#     _zero_vector_field = VectorField(x0, x_final, jnp.zeros_like(k0.vals))
#     _k_shape = jtu.tree_map(lambda arr: jnp.zeros_like(arr), _zero_vector_field)
#     _bm = dfx.UnsafeBrownianPath(shape=(nx, ny, 2), key=key)
#     _ode_term = dfx.ODETerm(_vf)
#     _sde_term = CustomVIIDControl(_diffusion, _bm, _x_shape, _k_shape)
#     _term = dfx.MultiTerm(_ode_term, _sde_term)

#     sol = dfx.diffeqsolve(
#         terms=_term,
#         solver=dfx.Euler(),
#         t0=t0,
#         t1=t1,
#         dt0=dt0,
#         y0=state0,
#         args=args,
#         saveat=saveat,
#         adjoint=dfx.ForwardMode(),
#         max_steps=10**6,
#     )
#     return sol.ys
