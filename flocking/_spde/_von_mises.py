import jax
import jax.numpy as jnp
from jax.scipy.special import i0


def _von_mises_1d(kappa, L):
    """
    One-dimensional Von-Mises kernel on torus [-L//2, L//2] with scale factor kappa =
    1/eps**2, and centered at zero
    """

    def _vm(x):
        return jnp.exp(kappa * jnp.cos(2 * jnp.pi * x / L))

    _Z = L * i0(kappa)
    return lambda x: _vm(x) / _Z


def _von_mises_2d(kappa, Lx, Ly):
    """
    Two-dimensional Von-Mises kernel on torus [-Lx//2, Lx//2] x [-Ly//2, Ly//2] with scale factor kappa = 1/eps**2, and centered at zero.
    """
    _vm_x = _von_mises_1d(kappa, Lx)
    _vm_y = _von_mises_1d(kappa, Ly)

    def _vm_kernel(xy):
        # xy has shape (2,)
        x, y = xy
        return _vm_x(x) * _vm_y(y)

    return _vm_kernel


def _von_mises_3d(kappa, Lx, Ly, Lz):
    """
    Two-dimensional Von-Mises kernel on torus [-Lx//2, Lx//2] x [-Ly//2, Ly//2] x [-Lz//2, Lz//2]
    with scale factor kappa = 1/eps**2, and centered at zero.
    """
    _vm_x = _von_mises_1d(kappa, Lx)
    _vm_y = _von_mises_1d(kappa, Ly)
    _vm_z = _von_mises_1d(kappa, Lz)

    def _vm_kernel(xyz):
        # xyz has shape (3,)
        x, y, z = xyz
        return _vm_x(x) * _vm_y(y) * _vm_z(z)

    return _vm_kernel


def _periodic_diffs(xi, xj, L):
    """
    Periodic difference between xi and xj on torus of length L
    """
    delta = xi - xj - (L) * jnp.round((xi - xj) / (L))
    return delta


def density_from_particles_1d(particles, grid, kappa, L, use_k=False):
    """
    Generate a one-dimensional density given initial particle positions by
    summing Von-Mises kernels
    """
    xs0, vs0 = particles
    vbar0 = jnp.mean(vs0, axis=0)
    _vm = _von_mises_1d(kappa, L)

    def _rho_xj(xj):
        _diffs = _periodic_diffs(xj, xs0, L)
        return jnp.mean(_vm(_diffs))

    def _j_xj(xj):
        _diffs = _periodic_diffs(xj, xs0, L)
        return jnp.mean(vs0 * _vm(_diffs))

    if use_k:

        def _k_xj(xj):
            _diffs = _periodic_diffs(xj, xs0, L)
            return jnp.mean((vs0 - vbar0) ** 2 * _vm(_diffs))

        return jax.vmap(_rho_xj)(grid), jax.vmap(_j_xj)(grid), jax.vmap(_k_xj)(grid)
    else:
        return jax.vmap(_rho_xj)(grid), jax.vmap(_j_xj)(grid)


def density_from_particles_2d(particles, grid, kappa, Lx, Ly, use_k=False):
    """
    Generate a two-dimensional density given initial particle positions by
    summing Von-Mises kernels
    """
    xs0, vs0 = particles
    vbar0 = jnp.mean(vs0, axis=0)
    nx, ny, _ = grid.shape
    _vm_2d = _von_mises_2d(kappa, Lx, Ly)

    def _rho_xj(xj):
        _diffs = _periodic_diffs(xj, xs0, jnp.array([Lx, Ly]))
        _weights = jax.vmap(_vm_2d)(_diffs)
        return jnp.mean(_weights)

    def _j_xj(xj):
        _diffs = _periodic_diffs(xj, xs0, jnp.array([Lx, Ly]))
        _weights = jax.vmap(_vm_2d)(_diffs)
        return jnp.mean(_weights[:, None] * vs0, axis=0)

    if use_k:

        def _k_xj(xj):
            _diffs = _periodic_diffs(xj, xs0, jnp.array([Lx, Ly]))
            _weights = jax.vmap(_vm_2d)(_diffs)
            return jnp.mean(_weights[:, None] * (vs0 - vbar0) ** 2, axis=0)

    _grid_flat = grid.reshape(-1, 2)
    _rho_flat = jax.vmap(_rho_xj)(_grid_flat)
    _j_flat = jax.vmap(_j_xj)(_grid_flat)
    rho = _rho_flat.reshape(nx, ny)
    j = _j_flat.reshape(nx, ny, 2)

    if use_k:
        _k_flat = jax.vmap(_k_xj)(_grid_flat)
        k = _k_flat.reshape(nx, ny, 2)
        return rho, j, k
    else:
        return rho, j


def density_from_particles_3d(particles, grid, kappa, Lx, Ly, Lz, use_k=False):
    """
    Generate a three-dimensional density given initial particle positions by
    summing Von-Mises kernels
    """
    xs0, vs0 = particles
    vbar0 = jnp.mean(vs0, axis=0)
    nx, ny, nz, _ = grid.shape
    _vm_3d = _von_mises_3d(kappa, Lx, Ly, Lz)

    def _rho_xj(xj):
        _diffs = _periodic_diffs(xj, xs0, jnp.array([Lx, Ly, Lz]))
        _weights = jax.vmap(_vm_3d)(_diffs)
        return jnp.mean(_weights)

    def _j_xj(xj):
        _diffs = _periodic_diffs(xj, xs0, jnp.array([Lx, Ly, Lz]))
        _weights = jax.vmap(_vm_3d)(_diffs)
        return jnp.mean(_weights[:, None] * vs0, axis=0)

    if use_k:

        def _k_xj(xj):
            _diffs = _periodic_diffs(xj, xs0, jnp.array([Lx, Ly, Lz]))
            _weights = jax.vmap(_vm_3d)(_diffs)
            return jnp.mean(_weights[:, None] * (vs0 - vbar0) ** 2, axis=0)

    _grid_flat = grid.reshape(-1, 3)
    _rho_flat = jax.vmap(_rho_xj)(_grid_flat)
    _j_flat = jax.vmap(_j_xj)(_grid_flat)
    rho = _rho_flat.reshape(nx, ny, nz)
    j = _j_flat.reshape(nx, ny, nz, 3)

    if use_k:
        _k_flat = jax.vmap(_k_xj)(_grid_flat)
        k = _k_flat.reshape(nx, ny, 3)
        return rho, j, k
    else:
        return rho, j


def _von_mises(kappa, L):
    """
    Returns a log-stabilised von-mises kernel
    """

    def _vm(x):
        exponent = jnp.cos(jnp.pi * x / L)
        return jnp.exp(kappa * jnp.sum(exponent))

    _Z = jnp.prod(2 * L * i0(kappa))
    return lambda x: _vm(x) / _Z


def density_from_particles(particles, grid, kappa, L, d):
    xs, vs = particles
    _vm = _von_mises(kappa, L)
    if d == 1:
        vs = vs[:, None]

    def _rho_j(xj):
        _diffs = _periodic_diffs(xj, xs, L)
        _diracs = jax.vmap(_vm)(_diffs)
        _rho = jnp.mean(_diracs)
        _j = jnp.mean(_diracs[:, None] * vs, axis=0)
        return _rho, _j

    _grid_flat = grid.reshape(-1, d)
    _rho_flat, _j_flat = jax.vmap(_rho_j)(_grid_flat)
    rho = _rho_flat.reshape(grid.shape[:-1])
    if d == 1:
        j = _j_flat.reshape(grid.shape)[..., 0]
    return rho, j
