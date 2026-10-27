from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, PyTree, Real, Shaped

RealScalarLike = Real[ArrayLike, ""]
BoolScalarLike = Bool[ArrayLike, ""]
Y = PyTree[Shaped[ArrayLike, "?*y"], "Y"]
VF = PyTree[Shaped[ArrayLike, "?*vf"], "VF"]
Control = PyTree[Shaped[ArrayLike, "?*control"], "C"]
Args = PyTree[Any]
DenseInfo = dict[str, PyTree[Array]]


def _is_scalar(obj):
    # need to allow for jnp.array([1.0])
    if jnp.isscalar(obj) or (isinstance(obj, jax.Array) and obj.ndim == obj.size == 1):
        return True


def _is_vector_like(obj):
    try:
        _vec = jnp.asarray(obj)
        return _vec.ndim >= 1 and _vec.size > 1
    except Exception:
        return False
