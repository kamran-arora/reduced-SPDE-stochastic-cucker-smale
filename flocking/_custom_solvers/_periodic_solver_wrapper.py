from collections.abc import Callable
from typing import ClassVar, TypeAlias

import diffrax as dfx
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import (
    RESULTS,
    AbstractTerm,
    LocalLinearInterpolation,
)
from jaxtyping import ArrayLike

from .._custom_types import (
    VF,
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    Y,
)

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


def _wrap_to_torus(y: Y, L: ArrayLike) -> Y:
    x, v = y
    x_wrapped = jtu.tree_map(lambda xi: xi - L * jnp.round(xi / L), x)
    return x_wrapped, v


class PeriodicSolver(dfx.AbstractWrappedSolver):
    solver: dfx.AbstractSolver
    L: ArrayLike
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    """
    For SDEs in (x, v), we wrap any Diffrax solver such that the position is 
    confined to a torus
    """

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        y1, _, dense_info, _, RESULTS.successful = self.solver.step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )
        y1 = _wrap_to_torus(y1, self.L)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms: AbstractTerm, t0: RealScalarLike, y0: Y, args: Args) -> VF:
        return terms.vf(t0, y0, args)
