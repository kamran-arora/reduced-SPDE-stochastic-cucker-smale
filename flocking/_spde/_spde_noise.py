import jax.numpy as jnp
import lineax as lx
from jaxtyping import ArrayLike

from .._spatial import ScalarField, VectorField


class VanishingNoise(lx.AbstractLinearOperator):
    state: any
    sigma: float
    vbar0: ArrayLike

    def __init__(self, state, args):
        self.state = state
        self.sigma = args.sigma
        self.vbar0 = args.vbar0

    def mv(self, x):
        """
        Computes a matrix-vector product (matrix-free)

        e.g. in 2D:

            [j_x - vbar_x*rho 0][B_x]
            [0 j_y - vbar_y*rho][B_y]
        """
        rho, j = self.state
        return self.sigma * (j - self.vbar0 * rho) * x

    def as_matrix(self):
        raise NotImplementedError

    def in_structure(self):
        raise NotImplementedError

    def out_structure(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError


@lx.is_symmetric.register(VanishingNoise)
def _(operator):
    return False


class VIIDNoise1D(lx.AbstractLinearOperator):
    state: any
    sigma: float
    N: int

    def __init__(self, state, N, args):
        self.state = state
        self.sigma = args.sigma
        self.N = N

    def mv(self, x):
        # extract state
        rho, j, k, vbar = self.state
        h = k.hs[0]
        # safely square root k
        k_pos = jnp.where(k.vals < 0, 0.0, k.vals)
        k_sqrt = jnp.sqrt(k_pos)
        # compute N^{-1/2}*sqrt(k)*dW and store as a ScalarField
        _k_vals = self.sigma * k_sqrt * x / (jnp.sqrt(self.N * h))
        _k_field = ScalarField(k.x0, k.x_final, _k_vals)
        # compute vbar update

        _vbar_vals = h * jnp.sum(_k_vals)
        return _k_field, _vbar_vals

    def as_matrix(self):
        raise NotImplementedError

    def in_structure(self):
        raise NotImplementedError

    def out_structure(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError


@lx.is_symmetric.register(VIIDNoise1D)
def _(operator):
    return False


class VIIDNoiseND(lx.AbstractLinearOperator):
    state: any
    sigma: float
    N: int

    def __init__(self, state, N, args):
        self.state = state
        self.sigma = args.sigma
        self.N = N

    def mv(self, x):
        # extract state
        rho, j, k, vbar = self.state
        # safely square root k
        k_pos = jnp.where(k.vals < 0, 0.0, k.vals)
        k_sqrt = jnp.sqrt(k_pos)
        # compute N^{-1/2}*sqrt(k)*dW and store as a ScalarField
        _k_vals = (
            self.sigma * k_sqrt * x / (jnp.sqrt(self.N * jnp.prod(jnp.array(k.hs))))
        )
        _k_field = VectorField(k.x0, k.x_final, _k_vals)
        # compute vbar update
        d = len(k.hs)
        _spatial_axes = tuple(range(d))
        hs = jnp.array([k.hs])
        _vbar_vals = jnp.prod(hs) * jnp.sum(_k_vals, axis=_spatial_axes)
        return _k_field, _vbar_vals

    def as_matrix(self):
        raise NotImplementedError

    def in_structure(self):
        raise NotImplementedError

    def out_structure(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError


@lx.is_symmetric.register(VIIDNoiseND)
def _(operator):
    return False


class VIIDNoise1DNoVbar(lx.AbstractLinearOperator):
    state: any
    sigma: float
    N: int

    def __init__(self, state, N, args):
        self.state = state
        self.sigma = args.sigma
        self.N = N

    def mv(self, x):
        # extract state
        rho, j, k = self.state
        h = k.hs[0]
        # safely square root k
        k_pos = jnp.where(k.vals < 0, 0.0, k.vals)
        k_sqrt = jnp.sqrt(k_pos)
        # compute N^{-1/2}*sqrt(k)*dW and store as a ScalarField
        _k_vals = self.sigma * k_sqrt * x / (jnp.sqrt(self.N * h))
        _k_field = ScalarField(k.x0, k.x_final, _k_vals)
        return _k_field

    def as_matrix(self):
        raise NotImplementedError

    def in_structure(self):
        raise NotImplementedError

    def out_structure(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError


@lx.is_symmetric.register(VIIDNoise1DNoVbar)
def _(operator):
    return False


class IIDNoise1D(lx.AbstractLinearOperator):
    state: any
    sigma: float
    N: int

    def __init__(self, state, N, args):
        self.state = state
        self.sigma = args.sigma
        self.N = N

    def mv(self, x):
        # extract state
        rho, j, vbar = self.state
        h = rho.hs[0]
        # safely square root rho
        rho_pos = jnp.where(rho.vals < 0, 0.0, rho.vals)
        rho_sqrt = jnp.sqrt(rho_pos)
        # compute N^{-1/2}*sqrt(k)*dW and store as a ScalarField
        _rho_vals = self.sigma * rho_sqrt * x / (jnp.sqrt(self.N * h))
        _rho_field = ScalarField(rho.x0, rho.x_final, _rho_vals)
        # compute vbar update

        _vbar_vals = h * jnp.sum(_rho_vals)
        return _rho_field, _vbar_vals

    def as_matrix(self):
        raise NotImplementedError

    def in_structure(self):
        raise NotImplementedError

    def out_structure(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError


@lx.is_symmetric.register(IIDNoise1D)
def _(operator):
    return False


class IIDNoiseND(lx.AbstractLinearOperator):
    state: any
    sigma: float
    N: int

    def __init__(self, state, N, args):
        self.state = state
        self.sigma = args.sigma
        self.N = N

    def mv(self, x):
        # extract state
        rho, j, vbar = self.state
        # safely square root k
        rho_pos = jnp.where(rho.vals < 0, 0.0, rho.vals)
        rho_sqrt = jnp.sqrt(rho_pos)
        # compute N^{-1/2}*sqrt(k)*dW and store as a ScalarField
        _rho_vals = (
            self.sigma * rho_sqrt * x / (jnp.sqrt(self.N * jnp.prod(jnp.array(rho.hs))))
        )
        _rho_field = VectorField(rho.x0, rho.x_final, _rho_vals)
        # compute vbar update
        d = len(rho.hs)
        _spatial_axes = tuple(range(d))
        hs = jnp.array([rho.hs])
        _vbar_vals = jnp.prod(hs) * jnp.sum(_rho_vals, axis=_spatial_axes)
        return _rho_field, _vbar_vals

    def as_matrix(self):
        raise NotImplementedError

    def in_structure(self):
        raise NotImplementedError

    def out_structure(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError


@lx.is_symmetric.register(IIDNoiseND)
def _(operator):
    return False


####################################
# Dean-Kawasaki style noises below
# they need checking and refactoring
####################################

# class DKNoise(lx.AbstractLinearOperator):
#     sqrt_rho_plus: ScalarField
#     div: Divergence
#     noise_coeff: float

#     def __init__(self, rho, div, noise_coeff):
#         if rho.spatial_dim == 1:
#             raise ValueError("DKNoise is for spatial dimensions d>1. If you are working in d=1, use DKNoise1D")
#         self.div = div
#         self.noise_coeff = noise_coeff
#         _vals = jnp.sqrt(jnp.where(rho.vals < 0, 0, rho.vals))
#         self.sqrt_rho_plus = ScalarField(rho.x0, rho.x_final, _vals)

#     def mv(self, x):
#         _vals = jnp.einsum("..., ...l -> ...l", self.sqrt_rho_plus.vals, x)
#         return self.noise_coeff*self.div(VectorField(self.sqrt_rho_plus.x0, self.sqrt_rho_plus.x_final, _vals))

#     def as_matrix(self):
#         raise NotImplementedError

#     def in_structure(self):
#         raise NotImplementedError

#     def out_structure(self):
#         raise NotImplementedError

#     def transpose(self):
#         raise NotImplementedError

# class DKNoise1D(lx.AbstractLinearOperator):
#     sqrt_rho_plus: ScalarField
#     grad: Gradient
#     noise_coeff: float

#     def __init__(self, rho, grad, noise_coeff):
#         if rho.spatial_dim != 1:
#             raise ValueError("DKNoise1D is for spatial dimension d=1. If you are working in d>1, use DKNoise")
#         self.grad = grad
#         self.noise_coeff = noise_coeff
#         _vals = jnp.sqrt(jnp.where(rho.vals < 0, 0, rho.vals))
#         self.sqrt_rho_plus = ScalarField(rho.x0, rho.x_final, _vals)

#     def mv(self, x):
#         _vals = self.sqrt_rho_plus.vals*x
#         return self.noise_coeff*self.grad(ScalarField(self.sqrt_rho_plus.x0, self.sqrt_rho_plus.x_final, _vals))

#     def as_matrix(self):
#         raise NotImplementedError

#     def in_structure(self):
#         raise NotImplementedError

#     def out_structure(self):
#         raise NotImplementedError

#     def transpose(self):
#         raise NotImplementedError

# class DKNoiseMixed(lx.AbstractLinearOperator):
#     sqrt_rho_plus: ScalarField
#     noise_coeff: float

#     def __init__(self, rho, noise_coeff):
#         if rho.spatial_dim == 1:
#             raise ValueError("DKNoiseMixed is for spatial dimensions d>1. If you are working in d=1, use DKNoiseMixed1D")
#         self.noise_coeff = noise_coeff
#         _vals = jnp.sqrt(jnp.where(rho.vals < 0, 0, rho.vals))
#         self.sqrt_rho_plus = ScalarField(rho.x0, rho.x_final, _vals)

#     def mv(self, x):
#         _vals = jnp.einsum("..., ...l -> ...l", self.sqrt_rho_plus.vals, x)
#         return self.noise_coeff*VectorField(self.sqrt_rho_plus.x0, self.sqrt_rho_plus.x_final, _vals)

#     def as_matrix(self):
#         raise NotImplementedError

#     def in_structure(self):
#         raise NotImplementedError

#     def out_structure(self):
#         raise NotImplementedError

#     def transpose(self):
#         raise NotImplementedError


class DKNoiseMixed1D(lx.AbstractLinearOperator):
    sqrt_field_plus: ScalarField
    noise_coeff: float

    def __init__(self, field, noise_coeff):
        if field.spatial_dim != 1:
            raise ValueError(
                "DKNoiseMixed1D is for spatial dimension d=1. If you are working in d>1, use DKNoiseMixed"
            )
        self.noise_coeff = noise_coeff
        _vals = jnp.sqrt(jnp.where(field.vals < 0, 0, field.vals))
        self.sqrt_field_plus = ScalarField(field.x0, field.x_final, _vals)

    def mv(self, x):
        _vals = self.sqrt_field_plus.vals * x
        return self.noise_coeff * ScalarField(
            self.sqrt_field_plus.x0, self.sqrt_field_plus.x_final, _vals
        )

    def as_matrix(self):
        raise NotImplementedError

    def in_structure(self):
        raise NotImplementedError

    def out_structure(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError


# @lx.is_symmetric.register(DKNoise)
# def _(operator):
#     return False

# @lx.is_symmetric.register(DKNoise1D)
# def _(operator):
#     return False

# @lx.is_symmetric.register(DKNoiseMixed)
# def _(operator):
#     return False


@lx.is_symmetric.register(DKNoiseMixed1D)
def _(operator):
    return False
