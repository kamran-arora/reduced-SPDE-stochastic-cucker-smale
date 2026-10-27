import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import flocking as fl

jax.config.update("jax_enable_x64", True)

"""
Script to demonstrate how the addition of environmental noise can drive flocking
"""


def make_ic_particle(key, N):
    k1, k2, k3, k4 = jr.split(key, 4)
    xs_left = jr.uniform(k1, shape=(N - 1,), minval=-12.0, maxval=-10.0)
    vs_left = jr.uniform(k2, shape=(N - 1,), minval=-0.1, maxval=1.2)
    x_right = jr.uniform(k3, shape=(1,), minval=10.0, maxval=12.0)
    v_right = jr.uniform(k4, shape=(1,), minval=0.05, maxval=0.15)

    xs0 = jnp.concatenate([xs_left, x_right])
    vs0 = jnp.concatenate([vs_left, v_right])

    return xs0, vs0


def make_ic_spde(particles, grid, x0, x_final, Lx, kappa):
    rho0, j0 = fl.density_from_particles_1d(particles, grid, kappa, Lx)
    return fl.ScalarField(x0, x_final, rho0), fl.ScalarField(x0, x_final, j0)


def main(sigma, key):
    N = 100
    lambda_ = 10.0
    r = 4.0
    kappa = 64.0
    Lx = 200.0
    x0 = (-Lx / 2.0,)
    x_final = (Lx / 2.0,)
    nx = (512,)
    grid = fl.make_grid(x0, x_final, nx)
    h = Lx / nx[0]
    domain = "unbounded"

    ic_key, solve_key = jr.split(key, 2)

    xs0, vs0 = make_ic_particle(ic_key, N)
    rho0, j0 = make_ic_spde((xs0, vs0), grid, x0, x_final, Lx, kappa)

    vbar0 = jnp.mean(vs0)
    args = fl.CuckerSmaleArgs(
        L=jnp.array([Lx]), lambda_=lambda_, r=r, sigma=sigma, vbar0=vbar0
    )

    T = 10.0
    dt0 = 0.005
    ts = jnp.linspace(0.0, T, int(T / dt0) + 1)
    saveat = dfx.SaveAt(ts=ts)

    xs, vs = fl.solve_CS_SDE(
        t0=0.0,
        t1=T,
        dt0=dt0,
        saveat=saveat,
        state0=(xs0, vs0),
        noise_type="vanishing-common",
        key=solve_key,
        args=args,
        domain=domain,
        kernel="full",
    )

    rho_eps, j_eps = eqx.filter_vmap(
        lambda state: fl.density_from_particles_1d(state, grid, kappa, Lx)
    )((xs, vs))

    flock_particle = jnp.sum(h * (vbar0 * rho_eps - j_eps) ** 2, axis=-1)

    rhos, js = fl.solve_spde_common_noise(
        t0=0.0,
        t1=T,
        dt0=dt0,
        saveat=saveat,
        state0=(rho0, j0),
        args=args,
        dim=1,
        kernel_type="full",
        key=solve_key,
        conv_type="padded",
    )

    flock_spde = jnp.sum(h * (vbar0 * rhos.vals - js.vals) ** 2, axis=-1)

    return ts, flock_particle, flock_spde


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 0.8,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        }
    )

    ts, flock_particle_no_noise, flock_spde_no_noise = main(0.0, jr.key(123))

    fig, ax = plt.subplots()
    ax.plot(
        ts, flock_particle_no_noise, label="particles", color="black", alpha=0.8, lw=1.8
    )
    ax.plot(ts, flock_spde_no_noise, label="SPDE", color="tab:red", alpha=0.8, lw=1.8)
    ax.set_ylim(
        [0.0, 1.1 * jnp.max(jnp.stack([flock_particle_no_noise, flock_spde_no_noise]))]
    )
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$ \lVert j - \overline{v} \rho \rVert^2$")
    handles = [
        Line2D([0], [0], color="black", lw=2.4, label="Particles"),
        Line2D([0], [0], color="tab:red", lw=2.4, label="SPDE"),
    ]
    ax.legend(handles=handles, frameon=False)

    fig.tight_layout()
    fig.savefig("figures/noise_influence_vc_no_noise.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    _, flock_particle_noise, flock_spde_noise = main(1.5, jr.key(123))

    fig, ax = plt.subplots()
    ax.plot(
        ts, flock_particle_noise, label="particles", color="black", alpha=0.8, lw=1.8
    )
    ax.plot(ts, flock_spde_noise, label="SPDE", color="tab:red", alpha=0.8, lw=1.8)
    ax.set_ylim(
        [0.0, 1.1 * jnp.max(jnp.stack([flock_particle_noise, flock_spde_noise]))]
    )
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$ \lVert j - \overline{v} \rho \rVert^2$")

    tail = ts >= 0.55 * ts[-1]
    tail_max = jnp.max(jnp.stack([flock_particle_noise[tail], flock_spde_noise[tail]]))
    inset = ax.inset_axes([0.36, 0.36, 0.42, 0.42])
    inset.plot(ts, flock_particle_noise, color="black", alpha=0.8, lw=1.5)
    inset.plot(ts, flock_spde_noise, color="tab:red", alpha=0.8, lw=1.5)
    inset.set_xlim(ts[tail][0], ts[-1])
    inset.set_ylim([0.0, 1.1 * tail_max])
    inset.tick_params(labelsize=8)
    ax.indicate_inset_zoom(inset, edgecolor="0.35")

    handles = [
        Line2D([0], [0], color="black", lw=2.4, label="Particles"),
        Line2D([0], [0], color="tab:red", lw=2.4, label="SPDE"),
    ]
    ax.legend(handles=handles, frameon=False)

    fig.tight_layout()
    fig.savefig("figures/noise_influence_vc_noise.pdf", dpi=300, bbox_inches="tight")
    plt.show()
