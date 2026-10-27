from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Float

from .._custom_types import RealScalarLike, _is_scalar, _is_vector_like


def make_grid(x0: Tuple[float, ...], x_final: Tuple[float, ...], n: Tuple[int, ...]):
    """
    Helper to create grids from endpoints and number of nodes per axis

    **Args**

    - x0: tuple of left endpoints
    - x_final: tuple of right endpoints
    - n: tuple of nodes per axis

    """
    assert len(x0) == len(x_final) == len(n)
    _axes = [
        jnp.linspace(x0[i], x_final[i], n[i], endpoint=False) for i in range(len(n))
    ]
    _mesh = jnp.meshgrid(*_axes, indexing="ij")
    return jnp.stack(_mesh, axis=-1)


class AbstractField(eqx.Module):
    """
    Abstract base class for Fields

    A Field is a lightweight Equinox module that encapsulates the values of a function over a d-dimensional spatial grid. Specifically, we consider:

    (a) ScalarField: a scalar-valued function over a d-dimensional grid whose values are stored in an Array of shape (n_1, ..., n_d)

    (b) VectorField: a vector-valued function over a d-dimensional grid whose values are stored in an Array of shape (n_1, ..., n_d, d)
    """

    x0: Tuple[float, ...] = eqx.field(static=True)
    x_final: Tuple[float, ...] = eqx.field(static=True)
    vals: Float[Array, "..."]

    @property
    @abstractmethod
    def hs(self) -> Tuple[float, ...]:
        """
        Returns a tuple of grid-spacings in each coordinate direction
        """
        raise NotImplementedError

    @abstractmethod
    def _vec_mul(self, other: ArrayLike) -> AbstractField:
        """
        Defines the product between a d-dimensional vector and an AbstractField defined over a d-dimensional grid
        """
        raise NotImplementedError

    @abstractmethod
    def _field_mul(self, other: AbstractField) -> AbstractField:
        """
        Defines mutliplication between AbstractFields (where this is a well-defined operation)
        """
        raise NotImplementedError

    def _check_compatible_binop(self, other):
        if isinstance(other, type(self)):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretisations")
        else:
            raise TypeError("Incompatible field types")

    def __add__(self, other: AbstractField) -> AbstractField:
        self._check_compatible_binop(other)
        return type(self)(self.x0, self.x_final, self.vals + other.vals)

    def __radd__(self, other: AbstractField) -> AbstractField:
        return self.__add__(other)

    def __sub__(self, other: AbstractField) -> AbstractField:
        self._check_compatible_binop(other)
        return type(self)(self.x0, self.x_final, self.vals - other.vals)

    def __rsub__(self, other: AbstractField) -> AbstractField:
        self._check_compatible_binop(other)
        return type(self)(self.x0, self.x_final, other.vals - self.vals)

    def __mul__(self, other: ArrayLike) -> AbstractField:
        # we distinguish between multiplying by a scalar, a vector, or an AbstractField
        if _is_scalar(other):
            return type(self)(self.x0, self.x_final, other * self.vals)
        elif _is_vector_like(other):
            return self._vec_mul(other)
        elif isinstance(other, AbstractField):
            return self._field_mul(other)
        else:
            raise TypeError(f"Unsupported type {type(other)} for multiplication")

    def __rmul__(self, other: ArrayLike) -> AbstractField:
        return self.__mul__(other)

    def __neg__(self) -> AbstractField:
        return -1 * self

    def __truediv__(self, other: RealScalarLike) -> AbstractField:
        if isinstance(other, (float, int)):
            return (1 / other) * self
        else:
            raise TypeError(f"Division of field by type {type(other)} not supported")

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.x0 == other.x0
            and self.x_final == other.x_final
            and jnp.allclose(self.vals, other.vals)
        )

    @classmethod
    @abstractmethod
    def discretise_fn(
        cls,
        x0: Tuple[float, ...],
        x_final: Tuple[float, ...],
        n: Tuple[int, ...],
        fn: Callable,
    ) -> AbstractField:
        """
        Discretise functions f: Rd -> R and f: Rd -> Rd over grids

        Accepts functions of the form f(x_1, ... x_d)

        Returns a ScalarField or VectorField depending on output shape
        """
        raise NotImplementedError


class ScalarField(AbstractField):
    """
    Represents a scalar-valued field in d-dimensions with values stored as an array (n_1, ..., n_d)
    """

    @property
    def hs(self) -> Tuple[float, ...]:
        return tuple(
            (xf_i - x0_i) / n_i
            for (x0_i, xf_i, n_i) in zip(self.x0, self.x_final, self.vals.shape)
        )

    @classmethod
    def discretise_fn(
        cls,
        x0: Tuple[float, ...],
        x_final: Tuple[float, ...],
        n: Tuple[int, ...],
        fn: Callable,
    ) -> ScalarField:
        if any(ni < 2 for ni in n):
            raise ValueError("Must discretise into at least two points along each axis")
        assert len(x0) == len(x_final) == len(n)
        # handle d=1 separately
        if len(n) == 1:
            # _vals will always be of shape (n,) i.e. ScalarField
            _vals = jax.vmap(fn)(jnp.linspace(*x0, *x_final, *n, endpoint=False))
            return ScalarField(x0, x_final, _vals)
        if len(n) == 2:
            xs = jnp.linspace(x0[0], x_final[0], n[0], endpoint=False)
            ys = jnp.linspace(x0[1], x_final[1], n[1], endpoint=False)
            vals = jax.vmap(lambda x: jax.vmap(lambda y: fn(x, y))(ys))(xs)
            return cls(x0, x_final, vals)
        if len(n) == 3:
            xs = jnp.linspace(x0[0], x_final[0], n[0], endpoint=False)
            ys = jnp.linspace(x0[1], x_final[1], n[1], endpoint=False)
            zs = jnp.linspace(x0[2], x_final[2], n[2], endpoint=False)
            vals = jax.vmap(
                lambda x: jax.vmap(lambda y: jax.vmap(lambda z: fn(x, y, z))(zs))(ys)
            )(xs)
            vals = jax.vmap(lambda x: jax.vmap(lambda y: fn(x, y))(ys))(xs)
            return cls(x0, x_final, vals)
        else:
            raise NotImplementedError

    def _vec_mul(self, other: ArrayLike) -> VectorField:
        # equivalent to einsum(..., l -> ...l)
        _vec = jnp.asarray(other)
        assert _vec.shape[0] == self.spatial_dim
        _vals = jnp.einsum("..., l -> ...l", self.vals, _vec)
        return VectorField(self.x0, self.x_final, _vals)

    def _field_mul(self, other: AbstractField) -> AbstractField:
        # here we are saying the following:
        # 1. ScalarField*ScalarField -> ScalarField under pointwise multiplication of values
        # 2. ScalarField*VectorField -> VectorField under pointwise multiplication of values per component.
        if isinstance(other, ScalarField):
            return ScalarField(self.x0, self.x_final, self.vals * other.vals)
        elif isinstance(other, VectorField):
            _vals = jnp.einsum("..., ... l -> ...l", self.vals, other.vals)
            return VectorField(self.x0, self.x_final, _vals)

    @property
    def spatial_dim(self) -> int:
        return self.vals.ndim

    @property
    def field_dim(self) -> int:
        return 1


class VectorField(AbstractField):
    """
    Represents a vector-valued field in d-dimensions with values stored as an array (n_1, ..., n_d, d)
    """

    @property
    def hs(self) -> Tuple[float, ...]:
        return tuple(
            (xf_i - x0_i) / n_i
            for (x0_i, xf_i, n_i) in zip(self.x0, self.x_final, self.vals.shape[:-1])
        )

    @classmethod
    def discretise_fn(
        cls,
        x0: Tuple[float, ...],
        x_final: Tuple[float, ...],
        n: Tuple[int, ...],
        fn: Callable,
    ) -> VectorField:
        if any(ni < 2 for ni in n):
            raise ValueError("Must discretise into at least two points along each axis")
        assert len(x0) == len(x_final) == len(n)
        if len(n) == 2:
            xs = jnp.linspace(x0[0], x_final[0], n[0], endpoint=False)
            ys = jnp.linspace(x0[1], x_final[1], n[1], endpoint=False)
            vals = jax.vmap(lambda x: jax.vmap(lambda y: fn(x, y))(ys))(xs)
            return cls(x0, x_final, vals)
        if len(n) == 3:
            xs = jnp.linspace(x0[0], x_final[0], n[0], endpoint=False)
            ys = jnp.linspace(x0[1], x_final[1], n[1], endpoint=False)
            zs = jnp.linspace(x0[2], x_final[2], n[2], endpoint=False)
            vals = jax.vmap(
                lambda x: jax.vmap(lambda y: jax.vmap(lambda z: fn(x, y, z))(zs))(ys)
            )(xs)
            vals = jax.vmap(lambda x: jax.vmap(lambda y: fn(x, y))(ys))(xs)
            return cls(x0, x_final, vals)
        else:
            raise NotImplementedError

    def _vec_mul(self, other: ArrayLike) -> VectorField:
        # for VectorFields we use _vec_mul (which lives inside __mul__) to mean the operation jnp.einsum("...l, l -> ...l")
        # the operation jnp.einsum("...l, l -> ...") i.e. contraction against a vector is given inside __matmul__
        _vec = jnp.asarray(other)
        # assert _vec.ndim == self.vals.shape[-1]
        _vals = jnp.einsum("...l, l -> ...l", self.vals, _vec)
        return VectorField(self.x0, self.x_final, _vals)

    def __matmul__(self, other: ArrayLike) -> ScalarField:
        # this is a contraction of a VectorField against a vector
        _vec = jnp.asarray(other)
        # assert _vec.ndim == self.vals.shape[-1]
        _vals = jnp.einsum("...l, l -> ...", self.vals, _vec)
        return ScalarField(self.x0, self.x_final, _vals)

    def _field_mul(self, other: AbstractField) -> AbstractField:
        if isinstance(other, ScalarField):
            _vals = jnp.einsum("...l, ... -> ...l", self.vals, other.vals)
            return VectorField(self.x0, self.x_final, _vals)
        elif isinstance(other, VectorField):
            # computes elementwise (Hadamard) product of VectorFields
            return VectorField(self.x0, self.x_final, self.vals * other.vals)

    @property
    def spatial_dim(self) -> int:
        return self.vals.ndim - 1

    @property
    def field_dim(self) -> int:
        return self.vals.shape[-1]
