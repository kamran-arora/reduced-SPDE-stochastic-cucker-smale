from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from .._spatial import AbstractField


class AbstractOperator(eqx.Module):
    """
    Abstract base class for operators
    """

    def __call__(self, u: AbstractField, **kwargs) -> AbstractField:
        return self._call(u, **kwargs)

    @abstractmethod
    def _call(self, u: AbstractField, **kwargs) -> AbstractField:
        raise NotImplementedError

    def __add__(self, other: AbstractOperator) -> AbstractOperator:
        return OperatorSum(self, other)

    def __neg__(self) -> OperatorScale:
        return OperatorScale(-1.0, self)

    def __sub__(self, other: AbstractOperator) -> AbstractOperator:
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, (float, int, jnp.ndarray, list, tuple)):
            return OperatorScale(other, self)
        elif isinstance(other, AbstractOperator):
            return OperatorPointwiseMul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return OperatorComposition(self, other)


class OperatorSum(AbstractOperator):
    A: AbstractOperator
    B: AbstractOperator

    def _call(self, u: AbstractField, **kwargs) -> AbstractField:
        return self.A(u, **kwargs) + self.B(u, **kwargs)


class OperatorScale(AbstractOperator):
    scalar: float | int | jnp.ndarray | list | tuple
    A: AbstractOperator

    def _call(self, u: AbstractField, **kwargs) -> AbstractField:
        return self.scalar * self.A(u, **kwargs)


class OperatorPointwiseMul(AbstractOperator):
    A: AbstractOperator
    B: AbstractOperator

    def _call(self, u: AbstractField, **kwargs) -> AbstractField:
        return self.A(u, **kwargs) * self.B(u, **kwargs)


class OperatorComposition(AbstractOperator):
    A: AbstractOperator
    B: AbstractOperator

    def _call(self, u: AbstractField, **kwargs) -> AbstractField:
        return self.A(self.B(u, **kwargs), **kwargs)


class AbstractNonLinearOperator(AbstractOperator):
    @abstractmethod
    def _call(self, u: AbstractField) -> AbstractField:
        raise NotImplementedError


class AbstractLinearOperator(AbstractOperator):
    use_lx: Bool
    lx_operator: Optional[lx.AbstractLinearOperator] = None

    @abstractmethod
    def _call_std(self, u: AbstractField, **kwargs) -> AbstractField:
        raise NotImplementedError

    def _call(self, u: AbstractField, **kwargs) -> AbstractField:
        if self.use_lx:
            return self._call_lx(u)
        else:
            return self._call_std(u, **kwargs)

    def _call_lx(self, u: AbstractField) -> AbstractField:
        vals = self.lx_operator.mv(u.vals)
        return type(u)(u.x0, u.x_final, vals)


@partial(jax.jit, static_argnums=(1, 2))
def _central_difference_along_axis(
    u: Float[Array, "..."], axis: int, h: float
) -> Float[Array, "..."]:
    return (jnp.roll(u, shift=-1, axis=axis) - jnp.roll(u, shift=1, axis=axis)) / (
        2 * h
    )


@partial(jax.jit, static_argnums=(1, 2))
def _forward_difference_along_axis(
    u: Float[Array, "..."], axis: int, h: float
) -> Float[Array, "..."]:
    return (jnp.roll(u, shift=-1, axis=axis) - u) / h


@partial(jax.jit, static_argnums=(1, 2))
def _backward_difference_along_axis(
    u: Float[Array, "..."], axis: int, h: float
) -> Float[Array, "..."]:
    return (u - jnp.roll(u, shift=1, axis=axis)) / h


@partial(jax.jit, static_argnums=(1, 2))
def _upwind_difference_along_axis(
    u: Float[Array, "..."], axis: int, h: float, velocity: Float[Array, "..."]
) -> Float[Array, "..."]:
    forward = _forward_difference_along_axis(u, axis, h)
    backward = _backward_difference_along_axis(u, axis, h)
    return jnp.where(velocity > 0, backward, forward)


@partial(jax.jit, static_argnums=(1, 2))
def _laplacian_along_axis(
    u: Float[Array, "..."], axis: int, h: float
) -> Float[Array, "..."]:
    return (
        jnp.roll(u, shift=-1, axis=axis) - 2 * u + jnp.roll(u, shift=1, axis=axis)
    ) / h**2
