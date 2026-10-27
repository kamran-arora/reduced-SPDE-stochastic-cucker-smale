from .._spatial import ScalarField
from ._base import AbstractLinearOperator, _laplacian_along_axis


class Laplacian(AbstractLinearOperator):
    def __init__(self):
        self.use_lx = False
        self.lx_operator = None

    def _call_std(self, u):
        if not isinstance(u, ScalarField):
            raise TypeError(f"Laplacian not supported for type {type(u)}")
        if u.spatial_dim == 1:
            _laplacian = _laplacian_along_axis(u.vals, axis=0, h=u.hs[0])
            return ScalarField(u.x0, u.x_final, vals=_laplacian)
        elif u.spatial_dim > 1:
            _partials = [
                _laplacian_along_axis(u.vals, ax, h)
                for (ax, h) in zip(range(u.spatial_dim), u.hs)
            ]
            _laplacian = sum(_partials)
            return ScalarField(u.x0, u.x_final, vals=_laplacian)
        else:
            raise NotImplementedError
