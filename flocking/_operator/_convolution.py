from collections.abc import Callable
from typing import Any, Tuple

import jax.numpy as jnp
from jax.numpy.fft import fftfreq, fftn, ifftn, ifftshift

from .._spatial import AbstractField, ScalarField, VectorField
from ._base import AbstractOperator


class Convolution(AbstractOperator):
    """
    d-dimensional convolution of either a ScalarField or VectorField with a scalar-valued kernel

    Circular convolution
    """

    kernel_fft: ScalarField

    def __init__(
        self, x0: Tuple, x_final: Tuple, n: Tuple, kernel: Callable, kernel_args: Any
    ):
        """
        kernel: Callable(x1, ..., xd, arg_1, ..., arg_m) -> ()
        kernel_args: (arg_1, ..., arg_m)
        """

        def _p_kernel(*xs):
            coords = jnp.stack(xs, axis=0)
            return kernel(coords, *kernel_args)

        _kernel_vals = ScalarField.discretise_fn(x0, x_final, n, _p_kernel)
        # self.kernel_fft = ScalarField(x0, x_final, fftn(_kernel_vals.vals))
        _shifted_kernel = ifftshift(_kernel_vals.vals)
        self.kernel_fft = ScalarField(x0, x_final, fftn(_shifted_kernel))

    def _call(self, u: AbstractField) -> AbstractField:
        if isinstance(u, ScalarField):
            # for scalar fields we use convolution theorem
            _u_fft = fftn(u.vals)
            cell_vol = jnp.prod(jnp.array(u.hs))
            _conv = cell_vol * jnp.real(ifftn(_u_fft * self.kernel_fft.vals))
            return type(u)(u.x0, u.x_final, _conv)
        elif isinstance(u, VectorField):
            # for tensor fields we use convolution theorem componentwise then stack
            # manual unrolling fine as use case is often just d=1,2,3
            _u_ffts = [fftn(u.vals[..., i]) for i in range(u.vals.shape[-1])]
            cell_vol = jnp.prod(jnp.array(u.hs))
            _conv = [
                cell_vol * jnp.real(ifftn(ufft * self.kernel_fft.vals))
                for ufft in _u_ffts
            ]
            return type(u)(u.x0, u.x_final, jnp.stack(_conv, axis=-1))


class ConvolutionPadded(AbstractOperator):
    """
    d-dimensional convolution of either a ScalarField or VectorField with a scalar-valued kernel

    Padded convolution
    """

    kernel_fft: ScalarField

    def __init__(
        self, x0: Tuple, x_final: Tuple, n: Tuple, kernel: Callable, kernel_args: Any
    ):
        """
        kernel: Callable(x1, ..., xd, arg_1, ..., arg_m) -> ()
        kernel_args: (arg_1, ..., arg_m)
        """

        def _p_kernel(*xs):
            coords = jnp.stack(xs, axis=0)
            return kernel(coords, *kernel_args)

        _dx = (x_final[0] - x0[0]) / n[0]
        _grid_padded = fftfreq(2 * n[0], d=1 / (2 * n[0])) * _dx
        _kernel_vals = _p_kernel(_grid_padded)
        self.kernel_fft = ScalarField(x0, x_final, fftn(_kernel_vals))

    def _call(self, u: AbstractField) -> AbstractField:
        if isinstance(u, ScalarField):
            # for scalar fields we use convolution theorem
            _n = u.vals.shape[0]
            _u_padded = jnp.zeros(2 * _n)
            _u_padded = _u_padded.at[:_n].set(u.vals)
            _u_padded_fft = fftn(_u_padded)
            _conv = u.hs[0] * jnp.real(ifftn(_u_padded_fft * self.kernel_fft.vals))
            return type(u)(u.x0, u.x_final, _conv[:_n])
        elif isinstance(u, VectorField):
            # for tensor fields we use convolution theorem componentwise then stack
            # manual unrolling fine as use case is often just d=1,2,3
            raise NotImplementedError
