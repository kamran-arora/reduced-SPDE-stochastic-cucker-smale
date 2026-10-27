from collections.abc import Callable
from typing import TypeVar

import diffrax as dfx
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from diffrax import AbstractPath
from jax import ShapeDtypeStruct

from ._custom_types import VF, Args, Control, RealScalarLike, Y

_VF = TypeVar("_VF", bound=VF)
_Control = TypeVar("_Control", bound=Control)

"""
Custom diffrax control to deal with coupled SDEs of the following form:

dx(t) = h(v(t)) dt
dv(t) = f(t, x(t), v(t))dt + g(t, x(t), v(t))dw(t)

"""


def _prod(vf, control):
    return jnp.tensordot(jnp.conj(vf), control, axes=jnp.ndim(control))


class CustomControl(dfx.AbstractTerm):
    vector_field: Callable[[RealScalarLike, Y, Args], _VF]
    control: AbstractPath[_Control]
    x_shape: ShapeDtypeStruct

    """
    Custom diffrax control to deal with coupled SDEs of the following form:

    dx(t) = h(v(t)) dt
    dv(t) = f(t, x(t), v(t))dt + g(t, x(t), v(t))dw(t)
    
    """

    def __init__(
        self,
        vector_field: Callable[[RealScalarLike, Y, Args], _VF],
        control: AbstractPath[_Control]
        | Callable[[RealScalarLike, RealScalarLike], _Control],
        x_shape: ShapeDtypeStruct,
    ):
        self.vector_field = vector_field
        self.control = control
        self.x_shape = x_shape

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> VF:
        """
        Diffusion vector field for the v-SDE i.e. g(t, x(t), v(t))
        """
        return self.vector_field(t, y, args)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        """
        This is the Brownian motion for the v-SDE i.e. dw(t)
        """
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf: _VF, control: _Control) -> Y:
        """
        This contracts the vector-field against the Brownian motion i.e. it computes g(t, x(t), v(t))dw(t)

        It returns a tuple (zeros, g(t, x(t), v(t))dw(t))

        Notes on shapes:
        ---------------
        1. The contraction g(t, x(t), v(t))dw(t) should return a PyTree with the same structure as v(t).
        2. x_out is a PyTree of zeros with structure given by x_shape

        """
        x_out = jtu.tree_map(jnp.zeros_like, self.x_shape)
        if isinstance(vf, lx.AbstractLinearOperator):
            v_out = vf.mv(control)
        else:
            v_out = jtu.tree_map(_prod, vf, control)
        return x_out, v_out


class CustomVIIDControl(dfx.AbstractTerm):
    vector_field: Callable[[RealScalarLike, Y, Args], _VF]
    control: AbstractPath[_Control]
    rho_shape: ShapeDtypeStruct
    k_shape: ShapeDtypeStruct

    """
    Custom diffrax control to deal with the vanishing i.i.d noise SPDE in the
    variables (rho, j, k, vbar)

    The noise is only on (j, vbar)
    """

    def __init__(
        self,
        vector_field: Callable[[RealScalarLike, Y, Args], _VF],
        control: AbstractPath[_Control]
        | Callable[[RealScalarLike, RealScalarLike], _Control],
        rho_shape: ShapeDtypeStruct,
        k_shape: ShapeDtypeStruct,
    ):
        self.vector_field = vector_field
        self.control = control
        self.rho_shape = rho_shape
        self.k_shape = k_shape

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> VF:
        """
        Diffusion vector field for the v-SDE i.e. g(t, x(t), v(t))
        """
        return self.vector_field(t, y, args)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        """
        This is the Brownian motion for the v-SDE i.e. dw(t)
        """
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf: _VF, control: _Control) -> Y:
        """ """
        rho_out = jtu.tree_map(jnp.zeros_like, self.rho_shape)
        k_out = jtu.tree_map(jnp.zeros_like, self.k_shape)
        if isinstance(vf, lx.AbstractLinearOperator):
            j_out, vbar_out = vf.mv(control)
        else:
            j_out, vbar_out = jtu.tree_map(_prod, vf, control)
        return rho_out, j_out, k_out, vbar_out


class CustomVIIDControlNoVbar(dfx.AbstractTerm):
    vector_field: Callable[[RealScalarLike, Y, Args], _VF]
    control: AbstractPath[_Control]
    rho_shape: ShapeDtypeStruct
    k_shape: ShapeDtypeStruct

    """
    Custom diffrax control to deal with the vanishing i.i.d noise SPDE in the
    variables (rho, j, k)

    The noise is only on (j,)
    """

    def __init__(
        self,
        vector_field: Callable[[RealScalarLike, Y, Args], _VF],
        control: AbstractPath[_Control]
        | Callable[[RealScalarLike, RealScalarLike], _Control],
        rho_shape: ShapeDtypeStruct,
        k_shape: ShapeDtypeStruct,
    ):
        self.vector_field = vector_field
        self.control = control
        self.rho_shape = rho_shape
        self.k_shape = k_shape

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> VF:
        """
        Diffusion vector field for the v-SDE i.e. g(t, x(t), v(t))
        """
        return self.vector_field(t, y, args)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        """
        This is the Brownian motion for the v-SDE i.e. dw(t)
        """
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf: _VF, control: _Control) -> Y:
        """ """
        rho_out = jtu.tree_map(jnp.zeros_like, self.rho_shape)
        k_out = jtu.tree_map(jnp.zeros_like, self.k_shape)
        if isinstance(vf, lx.AbstractLinearOperator):
            j_out = vf.mv(control)
        else:
            j_out = jtu.tree_map(_prod, vf, control)
        return rho_out, j_out, k_out


class CustomIIDControl(dfx.AbstractTerm):
    vector_field: Callable[[RealScalarLike, Y, Args], _VF]
    control: AbstractPath[_Control]
    rho_shape: ShapeDtypeStruct

    """
    Custom diffrax control to deal with the vanishing i.i.d noise SPDE in the
    variables (rho, j, k, vbar)

    The noise is only on (j, vbar)
    """

    def __init__(
        self,
        vector_field: Callable[[RealScalarLike, Y, Args], _VF],
        control: AbstractPath[_Control]
        | Callable[[RealScalarLike, RealScalarLike], _Control],
        rho_shape: ShapeDtypeStruct,
    ):
        self.vector_field = vector_field
        self.control = control
        self.rho_shape = rho_shape

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> VF:
        """
        Diffusion vector field for the v-SDE i.e. g(t, x(t), v(t))
        """
        return self.vector_field(t, y, args)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        """
        This is the Brownian motion for the v-SDE i.e. dw(t)
        """
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf: _VF, control: _Control) -> Y:
        """ """
        rho_out = jtu.tree_map(jnp.zeros_like, self.rho_shape)
        if isinstance(vf, lx.AbstractLinearOperator):
            j_out, vbar_out = vf.mv(control)
        else:
            j_out, vbar_out = jtu.tree_map(_prod, vf, control)
        return rho_out, j_out, vbar_out
