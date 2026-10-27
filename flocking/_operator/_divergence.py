import numpy as onp
from ._base import AbstractLinearOperator
from .._spatial import ScalarField, VectorField
from ._base import _central_difference_along_axis

class Divergence(AbstractLinearOperator):

    def __init__(self):
        self.use_lx = False
        self.lx_operator = None
    
    def _call_std(self, u):
        if not isinstance(u, VectorField):
            raise TypeError(f"Divergence not supported for type {type(u)}")
        _axes = onp.arange(u.spatial_dim)
        _hs = u.hs
        # divergence of vector field Rd -> Rd
        # manual unrolling is fine as main use case is d=2,3
        _partials = [_central_difference_along_axis(u.vals[..., ax], ax, h) for (ax, h) in zip(_axes, _hs)]
        return ScalarField(u.x0, u.x_final, vals=sum(_partials))