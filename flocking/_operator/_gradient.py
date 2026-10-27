import jax.numpy as jnp
import numpy as onp

from .._spatial import ScalarField, VectorField
from ._base import (
    AbstractLinearOperator,
    _central_difference_along_axis,
    _upwind_difference_along_axis,
)


class Gradient(AbstractLinearOperator):
    method: str = "central"
    use_lx: bool = False
    lx_operator: any = None

    # def __init__(self, method="central"):
    #     self.use_lx = False
    #     self.lx_operator = None
    #     self.method = method

    def _call_std(self, u, velocity=None):
        if not isinstance(u, ScalarField):
            raise TypeError(f"Gradient not supported for type {type(u)}")
        if self.method == "upwind" and velocity is None:
            raise ValueError("Velocity field must be provided for upwind gradient")
        if u.spatial_dim == 1:
            # gradients of (n_1,)
            if self.method == "central":
                _grad = _central_difference_along_axis(u.vals, axis=0, h=u.hs[0])
            elif self.method == "upwind":
                _grad = _upwind_difference_along_axis(
                    u.vals, axis=0, h=u.hs[0], velocity=velocity
                )
            else:
                raise ValueError(f"Unknown gradient method: {self.method}")
            return ScalarField(u.x0, u.x_final, vals=_grad)
        else:
            # gradients of (n_1, ..., n_d)
            # manual unrolling is fine as main use case is d=1,2,3
            _axes = onp.arange(u.spatial_dim)
            _hs = u.hs
            if self.method == "central":
                _partials = [
                    _central_difference_along_axis(u.vals, ax, h)
                    for (ax, h) in zip(_axes, _hs)
                ]
            elif self.method == "upwind":
                _partials = [
                    _upwind_difference_along_axis(
                        u.vals,
                        axis=ax,
                        h=h,
                        velocity=velocity[ax],
                    )
                    for ax, h in zip(_axes, _hs)
                ]
            return VectorField(u.x0, u.x_final, vals=jnp.stack(_partials, axis=-1))
