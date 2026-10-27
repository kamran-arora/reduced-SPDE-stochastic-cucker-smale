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

Compare the vanishing-common-noise particle system against the corresponding SPDE in 1D at a terminal time T.

For each realisation this script:

1. samples an initial condition drawn from a common distribution
2. constructs the SPDE initial condition by mollification
3. evolves the particle system and SPDE with the same driving noise
4. computes the terminal discrete L2 error

    ||(rho_eps, j_eps) - (rho, j)||_{L^2}^2 ~= h * sum(vbar^2 * (rho_eps - rho)^2 + (j_eps - j)^2).

It then estimates
    
    E ||(rho_eps, j_eps) - (rho, j)||_{L^2}^2

for each requested pair (N, kappa).

"""


def _make_initial_particles(N, key, ic_type):

    if ic_type == "uniform":
        xmin, xmax = -50.0, 50.0
        vmin, vmax = -20.0, 25.0
        k1, k2 = jr.split(key, 2)
        xs0 = jr.uniform(key=k1, shape=(N,), minval=xmin, maxval=xmax)
        vs0 = jr.uniform(key=k2, shape=(N,), minval=vmin, maxval=vmax)
        return xs0, vs0

    elif ic_type == "cluster":
        k1, k2 = jr.split(key, 2)
        Lx = 100.0
        m = 3
        delta = 0.5
        # Three smooth spatial density peaks
        q = jr.uniform(k1, (N,), minval=-Lx / 2, maxval=Lx / 2)
        xs0 = q - delta * Lx / (2 * jnp.pi * m) * jnp.sin(2 * jnp.pi * m * q / Lx)
        # Original independent velocities
        vs_uniform = jr.uniform(k2, (N,), minval=-20.0, maxval=25.0)
        # Preferred velocities for the three spatial groups
        cluster_v = jnp.where(
            xs0 < -Lx / 6,
            -2.0,
            jnp.where(xs0 < Lx / 6, 4.0, 0.0),
        )
        # position-velocity correlation
        eta = 0.25
        mean_v = jnp.mean(vs_uniform)
        std_v = jnp.std(vs_uniform)
        z_uniform = (vs_uniform - mean_v) / std_v
        z_cluster = (cluster_v - jnp.mean(cluster_v)) / jnp.maximum(
            jnp.std(cluster_v), 1e-12
        )
        z = (1.0 - eta) * z_uniform + eta * z_cluster
        vs0 = mean_v + std_v * (z - jnp.mean(z)) / jnp.std(z)
        return xs0, vs0

    else:
        raise ValueError(f"Unknown initial condition: {ic_type}")


def _make_spde_initial_state(particles, grid, kappa, x0, x_final, Lx):
    rho0, j0 = fl.density_from_particles_1d(particles, grid, kappa, Lx)
    return fl.ScalarField(x0, x_final, rho0), fl.ScalarField(x0, x_final, j0)


@eqx.filter_jit
def _terminal_l2_error(
    particles0,
    grid,
    kappa,
    Lx,
    T,
    dt0,
    sigma,
    lambda_,
    r,
    noise_key,
):

    x0 = (-Lx / 2.0,)
    x_final = (Lx / 2.0,)
    h = Lx / grid.shape[0]
    vbar0 = jnp.mean(particles0[1], axis=0)
    args = fl.CuckerSmaleArgs(
        L=jnp.array([Lx]),
        lambda_=lambda_,
        r=r,
        sigma=sigma,
        vbar0=vbar0,
    )
    spde_state0 = _make_spde_initial_state(particles0, grid, kappa, x0, x_final, Lx)
    saveat = dfx.SaveAt(t1=True)
    xsT, vsT = fl.solve_CS_SDE(
        t0=0.0,
        t1=T,
        dt0=dt0,
        saveat=saveat,
        state0=particles0,
        noise_type="vanishing-common",
        key=noise_key,
        args=args,
        kernel="full",
    )
    rho_eps_T, j_eps_T = fl.density_from_particles_1d((xsT, vsT), grid, kappa, Lx)
    rho_T, j_T = fl.solve_spde_common_noise(
        t0=0.0,
        t1=T,
        dt0=dt0,
        saveat=saveat,
        state0=spde_state0,
        args=args,
        dim=1,
        kernel_type="full",
        key=noise_key,
    )

    rho_diff = rho_eps_T - rho_T.vals
    j_diff = j_eps_T - j_T.vals

    err1 = h * jnp.sum(rho_diff**2 + j_diff**2)
    err2 = h * jnp.sum(vbar0**2 * rho_diff**2 + j_diff**2)
    err3 = h * jnp.sum(rho_diff**2)
    err4 = h * jnp.sum(j_diff**2)
    return jnp.array([err1, err2, err3, err4])


def run_experiment(
    realisations,
    Ns,
    kappas,
    initial_conditions,
    T,
    dt,
    sigma,
    lambda_,
    r,
    nx,
    Lx,
    seed,
):
    x0 = (-Lx / 2.0,)
    x_final = (Lx / 2.0,)
    grid = fl.make_grid(x0, x_final, (nx,))
    grid = grid[:, 0]

    # Axes: N, kappa, initial condition, realisation, error component.
    pathwise_errors = jnp.zeros(
        (len(Ns), len(kappas), len(initial_conditions), realisations, 4)
    )
    master_key = jr.key(seed)
    combo_keys = jr.split(master_key, len(Ns) * len(kappas) * len(initial_conditions))

    combo_idx = 0
    for i, N in enumerate(Ns):
        for j, kappa in enumerate(kappas):
            for k, ic_type in enumerate(initial_conditions):
                combo_key = combo_keys[combo_idx]
                combo_idx += 1
                ic_key, realisation_key = jr.split(combo_key, 2)
                noise_keys = jr.split(realisation_key, realisations)
                ic_keys = jr.split(ic_key, realisations)

                time0 = timer()
                combo_errors = []
                for realisation_idx in range(realisations):
                    particles0 = _make_initial_particles(
                        N, ic_keys[realisation_idx], ic_type
                    )
                    error = _terminal_l2_error(
                        particles0=particles0,
                        grid=grid,
                        kappa=kappa,
                        Lx=Lx,
                        T=T,
                        dt0=dt,
                        sigma=sigma,
                        lambda_=lambda_,
                        r=r,
                        noise_key=noise_keys[realisation_idx],
                    )
                    combo_errors.append(error)

                combo_errors = jnp.stack(combo_errors)
                pathwise_errors = pathwise_errors.at[i, j, k, :, :].set(combo_errors)
                time1 = timer()
                print(
                    f"N={N}, kappa={float(kappa):g}, IC={ic_type} "
                    f"finished in {time1 - time0:.4f}s, "
                    f"weighted MSE={jnp.mean(combo_errors[:, 1]):.6e}"
                )

    mean_squared_errors = jnp.mean(pathwise_errors, axis=3)
    return {
        "Ns": jnp.asarray(Ns),
        "kappas": jnp.asarray(kappas),
        "pathwise_errors": pathwise_errors,
        "mean_squared_errors": mean_squared_errors,
        "T": jnp.asarray(T),
        "dt": jnp.asarray(dt),
        "sigma": jnp.asarray(sigma),
        "lambda_": jnp.asarray(lambda_),
        "r": jnp.asarray(r),
        "nx": jnp.asarray(nx),
        "Lx": jnp.asarray(Lx),
        "seed": jnp.asarray(seed),
    }


def main():
    RUN = True
    PLOT = True

    sigma = 0.5
    initial_conditions = ("uniform", "cluster")
    data_file = "data/compare_L2_common_1d_250_twoIC.npz"

    if RUN:
        results = run_experiment(
            realisations=250,
            Ns=jnp.array([2**k for k in range(5, 12)]),
            kappas=jnp.array([25.0, 64.0]),
            initial_conditions=initial_conditions,
            T=2.0,
            dt=0.01,
            sigma=sigma,
            lambda_=50,
            r=0.5,
            nx=256,
            Lx=100,
            seed=456,
        )
        jnp.savez(data_file, **results)

    if PLOT:
        data = jnp.load(data_file)

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

        fig, ax = plt.subplots()
        colors = plt.cm.plasma(jnp.linspace(0.1, 0.85, len(data["kappas"])))
        linestyles = ("-", "--")  # smooth, separated

        for j, color in enumerate(colors):
            for k, linestyle in enumerate(linestyles):
                ax.plot(
                    data["Ns"],
                    data["mean_squared_errors"][:, j, k, 1],
                    color=color,
                    marker="s",
                    linestyle=linestyle,
                )

        # One N^{-1} visual guide, anchored near the first smooth-data point.
        Ns = data["Ns"]
        reference = 0.5 * data["mean_squared_errors"][0, 0, 0, 1] * Ns[0] / Ns
        ax.plot(Ns, reference, color="black", lw=1.2, linestyle=":")

        handles = [
            Line2D(
                [0],
                [0],
                color=colors[j],
                lw=2.4,
                label=rf"$\varepsilon = {1.0 / float(kappa) ** 0.5:.2g}$",
            )
            for j, kappa in enumerate(data["kappas"])
        ] + [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                lw=2.4,
                label="Uniform",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                lw=2.4,
                label="Clusters",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=":",
                lw=1.2,
                label=r"$\propto N^{-1}$",
            ),
        ]

        ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel(
            r"$\mathbb{E} \lVert X_{\mathrm{PDE}}(T) - X_{\mathrm{CS}}(T) \rVert^2$"
        )
        ax.legend(handles=handles, frameon=False)
        fig.tight_layout()
        fig.savefig("figures/L2_common_1d_250_twoIC.pdf", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
