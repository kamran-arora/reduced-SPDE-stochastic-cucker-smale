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

Compare 2D particle and reduced SPDE flocking quantities:

common noise: E ||j - vbar*rho||_L2^2;

i.i.d. noise: E[||j - vbar*rho||_h^2 + ||k||_h].
"""


def _make_initial_particles(N, key):
    """Three asymmetric 2D particle clusters, matching the 1D experiment."""
    if N < 3:
        raise ValueError("N must be at least 3")

    sizes = ((N + 2) // 3, (N + 1) // 3, N // 3)
    x_ranges = ((-27.0, -18.0), (-4.0, 2.0), (16.0, 28.0))
    v_ranges = ((-3.0, -1.0), (3.0, 5.0), (-1.0, 1.0))
    keys = jr.split(key, 10)

    xs, vs = [], []
    for i, (size, xr, vr) in enumerate(zip(sizes, x_ranges, v_ranges)):
        # Shift both spatial coordinates independently for each cluster.
        shift = jr.uniform(keys[7 + i], (2,), minval=-2.0, maxval=2.0)
        xs.append(
            jr.uniform(
                keys[2 * i],
                (size, 2),
                minval=xr[0] + shift,
                maxval=xr[1] + shift,
            )
        )
        vs.append(
            jr.uniform(
                keys[2 * i + 1],
                (size, 2),
                minval=vr[0],
                maxval=vr[1],
            )
        )

    perm = jr.permutation(keys[6], N)
    return jnp.concatenate(xs, axis=0)[perm], jnp.concatenate(vs, axis=0)[perm]


def main_particle(
    realisations,
    N,
    n,
    kappa,
    noise_type,
    SEED,
):
    """
    realisations: number of Monte-Carlo realisations to use
    N: number of particles
    n: number of gridpoints (tuple of length 2)
    kappa: Von-Mises scale parameter
    SEED: seed for jax.random `master` key
    """

    t0, t1 = 0.0, 0.5
    dt0 = 0.005
    ts = jnp.linspace(t0, t1, int((t1 - t0) / dt0) + 1)
    saveat = dfx.SaveAt(ts=ts)

    if noise_type == "vanishing-common":
        sigmas = jnp.array([0.1, 1.0, 2.5, 3.0])

    elif noise_type == "vanishing-iid":
        sigmas = jnp.array([0.1, 1.0, 2.5, 5.0])

    nx, ny = n
    Lx = Ly = 100
    x0 = (-Lx // 2, -Ly // 2)
    x_final = (Lx // 2, Ly // 2)
    hs = jnp.array([Lx / nx, Ly / ny])
    grid = fl.make_grid(x0, x_final, n)
    master_key = jr.key(SEED)
    ic_keys, s_keys = jr.split(master_key, 2)

    @eqx.filter_jit
    def _run_with_sigma(sigma, ic_key, key):
        """
        Runs a single realisation of the Cucker-Smale SDE.
        """

        particles = _make_initial_particles(N, ic_key)
        vbar0 = jnp.mean(particles[1], axis=0)

        args = fl.CuckerSmaleArgs(
            L=jnp.array([Lx, Ly]),
            lambda_=50,
            r=0.5,
            sigma=sigma,
            vbar0=vbar0,
        )

        xs, vs = fl.solve_CS_SDE(
            t0=t0,
            t1=t1,
            dt0=dt0,
            saveat=saveat,
            state0=particles,
            noise_type=noise_type,
            key=key,
            args=args,
        )

        if noise_type == "vanishing-common":
            rhos_eps, js_eps = eqx.filter_vmap(
                lambda state: fl.density_from_particles_2d(state, grid, kappa, Lx, Ly)
            )((xs, vs))

            _diff = js_eps - rhos_eps[..., None] * vbar0

            # _diff has shape (time, nx, ny, 2)
            flock = jnp.prod(hs) * jnp.sum(_diff**2, axis=(1, 2, 3))

            return flock, None

        elif noise_type == "vanishing-iid":
            vbars = jnp.mean(vs, axis=1)

            rhos_eps, js_eps, ks_eps = eqx.filter_vmap(
                lambda state: fl.density_from_particles_2d(
                    state, grid, kappa, Lx, Ly, use_k=True
                )
            )((xs, vs))

            _diff = js_eps - rhos_eps[..., None] * vbars[:, None, None, :]

            flock = jnp.prod(hs) * jnp.sum(_diff**2, axis=(1, 2, 3))
            flock += jnp.sqrt(jnp.prod(hs) * jnp.sum(ks_eps**2, axis=(1, 2, 3)))

            k_L1 = jnp.prod(hs) * jnp.sum(jnp.abs(ks_eps), axis=(1, 2, 3))

            return flock, k_L1

    results = {}
    ic_keys_split = jr.split(ic_keys, len(sigmas))
    s_keys_split = jr.split(s_keys, len(sigmas))

    for idx, sig in enumerate(sigmas):
        keys = jr.split(s_keys_split[idx], realisations)
        initial_condition_keys = jr.split(ic_keys_split[idx], realisations)
        time0 = timer()
        mean_flock = 0.0
        mean_k_L1 = 0.0

        for i in range(realisations):
            flock, k_L1 = _run_with_sigma(sig, initial_condition_keys[i], keys[i])
            mean_flock += flock / realisations
            if k_L1 is not None:
                mean_k_L1 += k_L1 / realisations

        time1 = timer()
        print(f"sigma = {sig} completed in {time1 - time0:.4f} seconds")
        results[f"{sig!s}_flock"] = mean_flock
        results[f"{sig!s}_k_L1"] = mean_k_L1

    return results


def main_spde(
    realisations,
    N,
    n,
    kappa,
    noise_type,
    SEED,
):
    """
    realisations: number of Monte-Carlo realisations to use
    N: number of particles
    n: number of gridpoints
    kappa: Von-Mises scale parameter
    SEED: seed for jax.random `master` key
    """

    t0, t1 = 0.0, 0.5
    dt0 = 0.005
    ts = jnp.linspace(t0, t1, int((t1 - t0) / dt0) + 1)
    saveat = dfx.SaveAt(ts=ts)

    if noise_type == "vanishing-common":
        sigmas = jnp.array([0.1, 1.0, 2.5, 3.0])

    elif noise_type == "vanishing-iid":
        sigmas = jnp.array([0.1, 1.0, 2.5, 5.0])

    nx, ny = n
    Lx = Ly = 100
    x0 = (-Lx // 2, -Ly // 2)
    x_final = (Lx // 2, Ly // 2)
    hs = jnp.array([Lx / nx, Ly / ny])
    grid = fl.make_grid(x0, x_final, n)
    lambda_ = 50
    r = 0.5
    master_key = jr.key(SEED)
    ic_keys, s_keys = jr.split(master_key, 2)

    @eqx.filter_jit
    def _run_with_sigma(sigma, ic_key, key):

        particles = _make_initial_particles(N, ic_key)
        vbar0 = jnp.mean(particles[1], axis=0)
        rho0, j0, k0 = fl.density_from_particles_2d(
            particles, grid, kappa, Lx, Ly, use_k=True
        )
        rho0_field = fl.ScalarField(x0, x_final, rho0)
        j0_field = fl.VectorField(x0, x_final, j0)
        k0_field = fl.VectorField(x0, x_final, k0)

        if noise_type == "vanishing-common":
            state0 = (rho0_field, j0_field)

        elif noise_type == "vanishing-iid":
            state0 = (rho0_field, j0_field, k0_field, vbar0)

        args = fl.CuckerSmaleArgs(
            L=jnp.array([Lx, Ly]),
            lambda_=lambda_,
            r=r,
            sigma=sigma,
            vbar0=vbar0,
        )

        if noise_type == "vanishing-common":
            rhos, js = fl.solve_spde_common_noise(
                t0=t0,
                t1=t1,
                dt0=dt0,
                saveat=saveat,
                state0=state0,
                args=args,
                dim=2,
                kernel_type="full",
                key=key,
            )

            _diff = js.vals - rhos.vals[..., None] * vbar0

            # this has shape (time, nx, ny, 2)
            flock = hs.prod() * jnp.sum(_diff**2, axis=(1, 2, 3))

            return flock, None

        elif noise_type == "vanishing-iid":
            rhos, js, ks, vbars = fl.solve_spde_viid(
                t0=t0,
                t1=t1,
                dt0=dt0,
                saveat=saveat,
                state0=state0,
                args=args,
                dim=2,
                kernel_type="full",
                N=N,
                key=key,
            )

            _diff = js.vals - rhos.vals[..., None] * vbars[:, None, None, :]

            flock = hs.prod() * jnp.sum(_diff**2, axis=(1, 2, 3))
            flock += jnp.sqrt(jnp.prod(hs) * jnp.sum(ks.vals**2, axis=(1, 2, 3)))

            k_L1 = jnp.prod(hs) * jnp.sum(jnp.abs(ks.vals), axis=(1, 2, 3))

            return flock, k_L1

    results = {}
    ic_keys_split = jr.split(ic_keys, len(sigmas))
    s_keys_split = jr.split(s_keys, len(sigmas))

    for idx, sig in enumerate(sigmas):
        keys = jr.split(s_keys_split[idx], realisations)
        initial_condition_keys = jr.split(ic_keys_split[idx], realisations)
        time0 = timer()
        mean_flock = 0.0
        mean_k_L1 = 0.0

        for i in range(realisations):
            flock, k_L1 = _run_with_sigma(sig, initial_condition_keys[i], keys[i])
            mean_flock += flock / realisations
            if k_L1 is not None:
                mean_k_L1 += k_L1 / realisations

        time1 = timer()
        print(f"Sigma {sig} completed in {time1 - time0:.4f} seconds")
        results[f"{sig!s}_flock"] = mean_flock
        results[f"{sig!s}_k_L1"] = mean_k_L1

    return results


if __name__ == "__main__":
    PARTICLE = False
    SAVE_PARTICLE = False
    SPDE = False
    SAVE_SPDE = False
    PLOT = True

    if PARTICLE:
        all_results_particle = {}

        for noise_type, SEED, kappa in zip(
            ["vanishing-common", "vanishing-iid"],
            (3, 3),
            (25.0, 25.0),
        ):
            results_particle = main_particle(
                100, 250, (64, 64), kappa, noise_type, SEED
            )

            all_results_particle[noise_type] = results_particle

    if SPDE:
        all_results_spde = {}

        for noise_type, SEED, kappa in zip(
            ["vanishing-common", "vanishing-iid"],
            (3, 3),
            (25.0, 25.0),
        ):
            results_spde = main_spde(100, 250, (64, 64), kappa, noise_type, SEED)

            all_results_spde[noise_type] = results_spde

    if SAVE_PARTICLE:
        jnp.savez("data/all_flocking_particle_2d_random_ic", **all_results_particle)

    if SAVE_SPDE:
        jnp.savez("data/all_flocking_spde_2d_random_ic", **all_results_spde)

    if PLOT:
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

        t0, t1 = 0.0, 0.5
        dt0 = 0.005
        ts = jnp.linspace(t0, t1, int((t1 - t0) / dt0) + 1)
        noise_types = ["vanishing-common", "vanishing-iid"]
        noise_types = ["vanishing-common", "vanishing-iid"]
        sigmas_by_noise_type = {
            "vanishing-common": jnp.array([0.1, 1.0, 2.5, 3.0]),
            "vanishing-iid": jnp.array([0.1, 1.0, 2.5, 5.0]),
        }
        all_sigmas = sorted(
            {
                float(sigma)
                for sigmas in sigmas_by_noise_type.values()
                for sigma in sigmas
            }
        )

        global_colors = plt.cm.plasma(jnp.linspace(0.1, 0.85, len(all_sigmas)))
        sigma_to_color = {sigma: global_colors[i] for i, sigma in enumerate(all_sigmas)}

        for noise_type in noise_types:
            fig, ax = plt.subplots(figsize=(6.0, 4.0))

            sigmas = sigmas_by_noise_type[noise_type]

            data_particle = (
                jnp.load(
                    "data/all_flocking_particle_2d_random_ic.npz", allow_pickle=True
                )
            )[noise_type].item()

            data_spde = (
                jnp.load("data/all_flocking_spde_2d_random_ic.npz", allow_pickle=True)
            )[noise_type].item()

            spde_style = dict(alpha=1.0, zorder=3)
            particle_style = dict(linestyle="--", zorder=2)

            for sigma in sigmas:
                color = sigma_to_color[float(sigma)]
                key1 = str(sigma) + "_flock"
                ax.plot(ts, data_particle[key1], color=color, **particle_style)
                ax.plot(ts, data_spde[key1], color=color, **spde_style)

            ax.set_yscale("log")
            ax.set_xlabel(r"$t$")

            if noise_type == "vanishing-common":
                ax.set_ylabel(r"$\mathbb{E} \lVert j - \overline{v} \rho \rVert^2$")
            else:
                ax.set_ylabel(
                    r"$\mathbb{E}(\lVert j-\overline{v}\rho\rVert_h^2+\lVert k\rVert_h)$"
                )

            ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
            ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)

            sigma_handles = [
                Line2D(
                    [0],
                    [0],
                    color=sigma_to_color[float(sigma)],
                    lw=2.4,
                    label=rf"$\sigma={float(sigma):g}$",
                )
                for sigma in sigmas
            ]

            model_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    lw=2.4,
                    label="SPDE",
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    lw=2.4,
                    linestyle="--",
                    label="Particles",
                ),
            ]

            if noise_type == "vanishing-common":
                leg1 = ax.legend(
                    handles=sigma_handles,
                    # title=r"$\sigma$",
                    loc="lower left",
                    frameon=False,
                    bbox_to_anchor=(0.0, -0.045),
                )

                ax.add_artist(leg1)

                ax.legend(
                    handles=model_handles,
                    loc="upper left",
                    frameon=False,
                )

            elif noise_type == "vanishing-iid":
                leg1 = ax.legend(
                    handles=sigma_handles,
                    # title=r"$\sigma$",
                    loc="upper left",
                    frameon=False,
                )

                ax.add_artist(leg1)

                ax.legend(
                    handles=model_handles,
                    loc="center right",
                    frameon=False,
                )

            fig.tight_layout()

            fig.savefig(
                f"figures/flocking_2d_random_ic_{noise_type}.pdf",
                dpi=300,
                bbox_inches="tight",
            )

            plt.show()
