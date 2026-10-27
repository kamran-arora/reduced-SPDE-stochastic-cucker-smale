from pathlib import Path
from timeit import default_timer as timer

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import flocking as fl

jax.config.update("jax_enable_x64", True)

"""Compare rho and j between the common-noise particle system and SPDE in d=1."""


def _make_initial_particles(N, key):
    """Asymmetric clusters with random spatial shifts for the common-noise experiment."""
    if N < 3:
        raise ValueError("N must be at least 3")

    sizes = ((N + 2) // 3, (N + 1) // 3, N // 3)
    # Asymmetric widths/locations distinguish this setup from the i.i.d. experiment.
    x_ranges = ((-27.0, -18.0), (-4.0, 2.0), (16.0, 28.0))
    # Left-moving, right-moving, and approximately stationary clusters.
    v_ranges = ((-3.0, -1.0), (3.0, 5.0), (-1.0, 1.0))
    keys = jr.split(key, 10)

    xs, vs = [], []
    for i, (size, xr, vr) in enumerate(zip(sizes, x_ranges, v_ranges)):
        shift = jr.uniform(keys[7 + i], (), minval=-2.0, maxval=2.0)
        xs.append(
            jr.uniform(
                keys[2 * i],
                (size,),
                minval=xr[0] + shift,
                maxval=xr[1] + shift,
            )
        )
        vs.append(
            jr.uniform(
                keys[2 * i + 1],
                (size,),
                minval=vr[0],
                maxval=vr[1],
            )
        )

    perm = jr.permutation(keys[6], N)
    return jnp.concatenate(xs)[perm], jnp.concatenate(vs)[perm]


def _make_spde_ic(particles, grid, kappa, x0, x_final, Lx):
    rho0, j0 = fl.density_from_particles_1d(particles, grid, kappa, Lx)
    return (
        fl.ScalarField(x0, x_final, rho0),
        fl.ScalarField(x0, x_final, j0),
    )


@eqx.filter_jit
def _terminal_density(particles0, N, nx, kappa, particle_key, spde_key):
    Lx = 100.0
    x0 = (-Lx / 2.0,)
    x_final = (Lx / 2.0,)
    grid = fl.make_grid(x0, x_final, (nx,))
    vbar0 = jnp.mean(particles0[1], axis=0)

    args = fl.CuckerSmaleArgs(
        L=jnp.array([Lx]),
        lambda_=50,
        r=0.5,
        sigma=1.0,
        vbar0=vbar0,
    )

    spde0 = _make_spde_ic(particles0, grid, kappa, x0, x_final, Lx)
    t0 = 0.0
    t1 = 2.0
    dt0 = 0.0005
    save_ts = jnp.array([0.5, t1])
    saveat = dfx.SaveAt(ts=save_ts)

    xs, vs = fl.solve_CS_SDE(
        t0=t0,
        t1=t1,
        dt0=dt0,
        saveat=saveat,
        state0=particles0,
        noise_type="vanishing-common",
        key=particle_key,
        args=args,
        kernel="full",
    )

    rho_eps, j_eps = eqx.filter_vmap(
        lambda state: fl.density_from_particles_1d(state, grid, kappa, Lx)
    )((xs, vs))

    rho, j = fl.solve_spde_common_noise(
        t0=t0,
        t1=t1,
        dt0=dt0,
        saveat=saveat,
        state0=spde0,
        args=args,
        dim=1,
        kernel_type="full",
        key=spde_key,
    )

    # Shape (2 times, 4 fields, nx): rho, j, rho_particles, j_particles
    return jnp.stack([rho.vals, j.vals, rho_eps, j_eps], axis=1)


def run(realisations, N, nx, SEED, resample_ic=False):
    Lx = 100.0
    x0 = (-Lx / 2.0,)
    x_final = (Lx / 2.0,)
    grid = fl.make_grid(x0, x_final, (nx,))[:, 0]
    kappa = 64.0

    key = jr.key(SEED)
    ic_key, run_key = jr.split(key, 2)
    fixed_particles0 = _make_initial_particles(N, ic_key)

    means = jnp.zeros((2, 4, nx))
    means_squared = jnp.zeros((2, 4, nx))

    run_keys = jr.split(run_key, realisations)
    time0 = timer()

    for idx in range(realisations):
        ic_run_key, particle_spde_key = jr.split(run_keys[idx], 2)
        particles0 = (
            _make_initial_particles(N, ic_run_key) if resample_ic else fixed_particles0
        )

        densities = _terminal_density(
            particles0, N, nx, kappa, particle_spde_key, particle_spde_key
        )

        n = idx + 1
        delta = densities - means
        means = means + delta / n
        means_squared = means_squared + delta * (densities - means)

    if realisations < 2:
        raise ValueError(
            "At least 2 realisations are required to estimate a standard deviation."
        )
    var = means_squared / (realisations - 1)
    std = jnp.sqrt(jnp.maximum(var, 0.0))
    sem = std / jnp.sqrt(realisations)

    print(f"Time elapsed = {timer() - time0:.2f}s")
    return {
        "mean": means,
        "std": std,
        "sem": sem,
        "grid": grid,
        "times": jnp.array([0.5, 2.0]),
    }


def plot_results(data, shaded="std"):
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

    x = data["grid"]
    times = data["times"]
    mean = data["mean"]
    spread = 1.96 * data["sem"] if shaded == "sem" else data["std"]

    # Corresponding SPDE and particle field indices in mean/std.
    pairs = [
        (0, 2, r"$\rho$"),
        (1, 3, r"$j$"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True, sharey="col")

    # spde_color = "#0072B2"
    # particle_color = "#D55E00"
    spde_color = "black"
    particle_color = "tab:red"

    for row in range(2):
        for col, (spde_idx, particle_idx, title) in enumerate(pairs):
            ax = axs[row, col]

            # SPDE: blue mean and shaded one-standard-deviation band.
            spde_y = mean[row, spde_idx]
            spde_s = spread[row, spde_idx]
            ax.fill_between(
                x,
                spde_y - spde_s,
                spde_y + spde_s,
                color=spde_color,
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
            ax.plot(x, spde_y, color=spde_color, lw=2.0, zorder=3)

            # Particles: orange dashed mean and translucent standard-deviation band.
            particle_y = mean[row, particle_idx]
            particle_s = spread[row, particle_idx]
            ax.fill_between(
                x,
                particle_y - particle_s,
                particle_y + particle_s,
                color=particle_color,
                alpha=0.22,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                x,
                particle_y,
                color=particle_color,
                linestyle="--",
                linewidth=2.0,
                zorder=4,
            )

            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(rf"$t={float(times[row]):g}$")
            if row == 1:
                ax.set_xlabel(r"$x$")

            # Uncomment to focus on the occupied region of the domain.
            # ax.set_xlim(-15, 40)

    handles = [
        Line2D([0], [0], color=spde_color, lw=2.4, label="SPDE"),
        Line2D(
            [0],
            [0],
            color=particle_color,
            lw=2.4,
            linestyle="--",
            label="Particles",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(
        "figures/vc_comparison_1d_two_time.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    return fig, axs


if __name__ == "__main__":
    RUN = False
    SAVE = False
    PLOT = True

    if RUN:
        results = run(
            realisations=100,
            N=250,
            nx=512,
            SEED=123,
            resample_ic=True,
        )

    if SAVE:
        Path("data").mkdir(exist_ok=True)
        jnp.savez("data/approximation_vc_1d", **results)

    if PLOT:
        if not RUN:
            results = jnp.load("data/approximation_vc_1d.npz")
        fig, axs = plot_results(results, shaded="std")
        plt.show()
