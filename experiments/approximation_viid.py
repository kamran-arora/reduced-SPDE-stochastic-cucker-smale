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

"""
Script to compare rho, rho_eps and j, j_eps and k, k_eps for vanishing-i.i.d 
noise in d=1
"""


def _make_initial_particles(N, key):
    """Three clusters with independently randomised spatial shifts."""
    if N < 3:
        raise ValueError("N must be at least 3")

    sizes = ((N + 2) // 3, (N + 1) // 3, N // 3)
    x_ranges = ((-5.0, 2.0), (8.0, 15.0), (23.0, 30.0))
    v_ranges = ((1.0, 3.0), (2.0, 4.0), (-4.0, -2.0))
    keys = jr.split(key, 10)

    xs, vs = [], []
    for i, (size, xr, vr) in enumerate(zip(sizes, x_ranges, v_ranges)):
        # Random shift of the entire cluster
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
    rho0, j0, k0 = fl.density_from_particles_1d(particles, grid, kappa, Lx, use_k=True)
    vbar0 = jnp.mean(particles[1], axis=0)
    # jax.debug.print(
    #     "Initial discrepancy: {x:.6e}",
    #     x=(100 / 64) * jnp.sum((j0 - vbar0 * rho0) ** 2, axis=-1),
    # )
    return (
        fl.ScalarField(x0, x_final, rho0),
        fl.ScalarField(x0, x_final, j0),
        fl.ScalarField(x0, x_final, k0),
        vbar0,
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
        noise_type="vanishing-iid",
        key=particle_key,
        args=args,
        kernel="full",
    )

    rho_eps, j_eps, k_eps = eqx.filter_vmap(
        lambda state: fl.density_from_particles_1d(state, grid, kappa, Lx, use_k=True)
    )((xs, vs))

    # rho_eps, j_eps, k_eps = fl.density_from_particles_1d(
    #     (xs[0], vs[0]), grid, 25.0, Lx, use_k=True
    # )

    rho, j, k, _ = fl.solve_spde_viid(
        t0=t0,
        t1=t1,
        dt0=dt0,
        saveat=saveat,
        state0=spde0,
        args=args,
        dim=1,
        kernel_type="full",
        N=N,
        key=spde_key,
    )
    return jnp.stack([rho.vals, j.vals, k.vals, rho_eps, j_eps, k_eps], axis=1)


def run(realisations, N, nx, SEED, resample_ic=False):
    Lx = 100.0
    x0 = (-Lx / 2.0,)
    x_final = (Lx / 2.0,)
    grid = fl.make_grid(x0, x_final, (nx,))[:, 0]
    kappa = 64.0

    key = jr.key(SEED)
    ic_key, run_key = jr.split(key, 2)

    fixed_particles0 = _make_initial_particles(N, ic_key)

    means = jnp.zeros((2, 6, nx))
    means_squared = jnp.zeros((2, 6, nx))

    run_keys = jr.split(run_key, realisations)
    time0 = timer()
    for idx in range(realisations):
        if resample_ic:
            k1, particle_key, spde_key = jr.split(run_keys[idx], 3)
            particles0 = _make_initial_particles(N, k1)
        elif not resample_ic:
            _, particle_key, spde_key = jr.split(run_keys[idx], 3)
            particles0 = fixed_particles0

        densities = _terminal_density(particles0, N, nx, kappa, particle_key, spde_key)

        n = idx + 1
        delta = densities - means
        means = means + delta / n
        means_squared = means_squared + delta * (densities - means)

    var = means_squared / (realisations - 1)
    std = jnp.sqrt(jnp.maximum(var, 0.0))
    sem = std / jnp.sqrt(realisations)

    time1 = timer()
    print(f"Time elapsed = {time1 - time0:.2f}s")

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
    # spread = data[shaded]
    if shaded == "sem":
        spread = 1.96 * data[shaded]
    else:
        spread = data[shaded]

    pairs = [
        (0, 3, r"$\rho$", r"$\rho_\varepsilon$", r"$\rho$"),
        (1, 4, r"$j$", r"$j_\varepsilon$", r"$j$"),
        (2, 5, r"$k$", r"$k_\varepsilon$", r"$k$"),
    ]

    fig, axs = plt.subplots(
        2,
        3,
        figsize=(12.0, 6.0),
        sharex=True,
        sharey="col",
    )

    # spde_color = "#0072B2"
    # particle_color = "#D55E00"
    spde_color = "black"
    particle_color = "tab:red"

    for row in range(2):
        for col, (
            spde_idx,
            particle_idx,
            spde_label,
            particle_label,
            title,
        ) in enumerate(pairs):
            ax = axs[row, col]

            for idx, label, color, linestyle in [
                (spde_idx, spde_label, spde_color, "-"),
                (particle_idx, particle_label, particle_color, "--"),
            ]:
                y = mean[row, idx]
                s = spread[row, idx]

                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    lw=2.0,
                    label=label,
                )
                ax.fill_between(
                    x,
                    y - s,
                    y + s,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )

            ax.grid(True, alpha=0.25, linewidth=0.7)

            if row == 0:
                ax.set_title(title)

            if col == 0:
                ax.set_ylabel(rf"$t={float(times[row]):g}$")

            if row == 1:
                ax.set_xlabel(r"$x$")

    handles = [
        Line2D(
            [0],
            [0],
            color=spde_color,
            lw=2.4,
            linestyle="-",
            label="SPDE",
        ),
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

    # Zoom inset for the bottom-right k panel, where k is close to zero.
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    ax = axs[1, 2]
    row = 1
    spde_idx = 2
    particle_idx = 5

    axins = inset_axes(
        ax,
        width="43%",
        height="43%",
        loc="upper right",
        borderpad=1.0,
    )

    for idx, color, linestyle in [
        (spde_idx, spde_color, "-"),
        (particle_idx, particle_color, "--"),
    ]:
        y = mean[row, idx]
        s = spread[row, idx]

        axins.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            lw=1.5,
        )
        axins.fill_between(
            x,
            y - s,
            y + s,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

    # Choose a spatial window to zoom in on.
    x_min, x_max = -50.0, 50.0
    mask = (x >= x_min) & (x <= x_max)

    y_vals = []
    for idx in [spde_idx, particle_idx]:
        y = mean[row, idx]
        s = spread[row, idx]
        y_vals.append(y[mask] - s[mask])
        y_vals.append(y[mask] + s[mask])

    y_vals = jnp.concatenate(y_vals)
    y_min = float(jnp.min(y_vals))
    y_max = float(jnp.max(y_vals))
    y_pad = 0.12 * (y_max - y_min)

    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min - y_pad, y_max + y_pad)

    axins.grid(True, alpha=0.2, linewidth=0.5)
    axins.tick_params(labelsize=8)

    mark_inset(
        ax,
        axins,
        loc1=2,
        loc2=4,
        fc="none",
        ec="0.45",
        lw=0.8,
        alpha=0.8,
    )

    fig.tight_layout()
    fig.savefig("figures/viid_comparison_1d_two_time.pdf", dpi=300, bbox_inches="tight")
    return fig, axs


if __name__ == "__main__":
    RUN = False
    SAVE = False
    PLOT = True

    if RUN:
        results = run(
            realisations=100,
            N=250,
            nx=256,
            SEED=123,
            resample_ic=True,
        )

    if SAVE:
        jnp.savez("data/approximation_viid_1d", **results)

    if PLOT:
        if not RUN:
            results = jnp.load("data/approximation_viid_1d.npz")
        figs, axs = plot_results(results, shaded="std")
        plt.show()
