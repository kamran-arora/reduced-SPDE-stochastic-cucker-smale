import timeit

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import flocking as fl

jax.config.update("jax_enable_x64", True)


"""
Script to compute runtime performance for particle systems and reduced SPDE

We define runtime performance as the average (CPU) time to compute one timestep
"""


def _make_ic_particle(N, dim, key):
    k1, k2 = jr.split(key)
    xmin = 0.0
    xmax = 2 * jnp.pi
    xs_shape = (N,) if dim == 1 else (N, dim)
    xs0 = jr.uniform(key=k1, shape=xs_shape, minval=xmin, maxval=xmax)

    if dim == 1:
        vmin = -2.0
        vmax = 5.0
        vs_shape = (N,)
    else:
        vmins = jnp.array([-2.0, -0.5, -1.0])
        vmaxs = jnp.array([5.0, 1.5, 3.0])
        vmin = vmins[:dim]
        vmax = vmaxs[:dim]
        vs_shape = (N, dim)

    vs0 = jr.uniform(key=k2, shape=vs_shape, minval=vmin, maxval=vmax)

    return xs0, vs0


@eqx.filter_jit
def run_particle(state0, noise_type, key):
    """
    Run CS SDE for a given initial state with a given noise type

    For the purposes of the performance comparision, we hardcode the kernel parameters and noise intensity. We perform 250 timesteps, saving only at the final time.
    """

    args = fl.CuckerSmaleArgs(2 * jnp.pi, 10, 0.5, 0.1)
    xs, vs = fl.solve_CS_SDE(
        t0=0.0,
        t1=0.1,
        dt0=0.001,
        saveat=dfx.SaveAt(t1=True),
        state0=state0,
        noise_type=noise_type,
        key=key,
        args=args,
    )
    return xs, vs


def _make_ic_spde(N, dim, grid, key, use_k=False):
    L = 2 * jnp.pi
    x0 = tuple(0 for _ in range(dim))
    x_final = tuple(L for _ in range(dim))

    k1, k2 = jr.split(key)

    xs_shape = (N,) if dim == 1 else (N, dim)
    xs0 = jr.uniform(k1, xs_shape, minval=0.0, maxval=L)

    if dim == 1:
        vs_shape = (N,)
        vmin = -2.0
        vmax = 5.0
    else:
        vmins = jnp.array([-2.0, -0.5, -1.0])
        vmaxs = jnp.array([5.0, 1.5, 3.0])
        vs_shape = (N, dim)
        vmin = vmins[:dim]
        vmax = vmaxs[:dim]

    vs0 = jr.uniform(k2, vs_shape, minval=vmin, maxval=vmax)
    vbar0 = jnp.mean(vs0, axis=0)

    particles = (xs0, vs0)

    if use_k:
        if dim == 1:
            rho0, j0, k0 = fl.density_from_particles_1d(particles, grid, 50.0, L, use_k)
            j0_field = fl.ScalarField(x0, x_final, j0)
            k0_field = fl.ScalarField(x0, x_final, k0)

        elif dim == 2:
            rho0, j0, k0 = fl.density_from_particles_2d(
                particles, grid, 50.0, L, L, use_k
            )
            j0_field = fl.VectorField(x0, x_final, j0)
            k0_field = fl.VectorField(x0, x_final, k0)

        else:
            rho0, j0, k0 = fl.density_from_particles_3d(
                particles, grid, 50.0, L, L, L, use_k
            )
            j0_field = fl.VectorField(x0, x_final, j0)
            k0_field = fl.VectorField(x0, x_final, k0)

        rho0_field = fl.ScalarField(x0, x_final, rho0)
        del xs0, vs0, rho0, j0, k0
        state0 = (rho0_field, j0_field, k0_field, vbar0)

    else:
        if dim == 1:
            rho0, j0 = fl.density_from_particles_1d(particles, grid, 50.0, L)
            j0_field = fl.ScalarField(x0, x_final, j0)

        elif dim == 2:
            rho0, j0 = fl.density_from_particles_2d(particles, grid, 50.0, L, L)
            j0_field = fl.VectorField(x0, x_final, j0)

        else:
            rho0, j0 = fl.density_from_particles_3d(particles, grid, 50.0, L, L, L)
            j0_field = fl.VectorField(x0, x_final, j0)

        rho0_field = fl.ScalarField(x0, x_final, rho0)
        del xs0, vs0, rho0, j0
        state0 = (rho0_field, j0_field)

    return state0, vbar0


@eqx.filter_jit
def run_spde(state0, vbar0, dim, noise_type, N, key):
    if dim == 1:
        args = fl.CuckerSmaleArgs(
            L=jnp.array([2 * jnp.pi]), lambda_=10, r=0.5, sigma=0.1, vbar0=vbar0
        )
    elif dim == 2:
        args = fl.CuckerSmaleArgs(
            L=jnp.array([2 * jnp.pi, 2 * jnp.pi]),
            lambda_=10,
            r=0.5,
            sigma=0.1,
            vbar0=vbar0,
        )
    elif dim == 3:
        args = fl.CuckerSmaleArgs(
            L=jnp.array([2 * jnp.pi, 2 * jnp.pi, 2 * jnp.pi]),
            lambda_=10,
            r=0.5,
            sigma=0.1,
            vbar0=vbar0,
        )
    if noise_type == "vanishing-common":
        rhos, js = fl.solve_spde_common_noise(
            t0=0.0,
            t1=0.1,
            dt0=0.001,
            saveat=dfx.SaveAt(t1=True),
            state0=state0,
            args=args,
            dim=int(dim),
            kernel_type="full",
            key=key,
        )
        return rhos, js
    elif noise_type == "vanishing-iid":
        rhos, js, ks, vbars = fl.solve_spde_viid(
            t0=0.0,
            t1=0.1,
            dt0=0.001,
            saveat=dfx.SaveAt(t1=True),
            state0=state0,
            args=args,
            dim=int(dim),
            kernel_type="full",
            N=N,
            key=key,
        )
        return rhos, js, ks, vbars


if __name__ == "__main__":
    PARTICLE = False
    SPDE = False
    PLOT = True

    Ns = jnp.array([2**k for k in range(3, 13)])
    N_runs = 100
    N_steps = 100
    SEED = 123

    noise_types = ["vanishing-common", "vanishing-iid"]

    spde_grids = {
        1: (256,),
        2: (64, 64),
    }

    if PARTICLE:
        for dim in [1, 2]:
            for noise_type in noise_types:
                all_run_times_particle = {}

                print(f"Particle system: dim={dim}, noise_type={noise_type}")

                for N in Ns:
                    # run once for JIT
                    state0 = _make_ic_particle(N, dim, jr.key(0))
                    _ = jax.block_until_ready(
                        run_particle(state0, noise_type, jr.key(0))
                    )

                    run_times = jnp.zeros(N_runs)
                    keys = jr.split(jr.key(SEED), N_runs)

                    for k in range(N_runs):
                        k1, k2 = jr.split(keys[k], 2)
                        state0 = _make_ic_particle(N, dim, k1)
                        execution_time = timeit.timeit(
                            lambda: jax.block_until_ready(
                                run_particle(state0, noise_type, k2)
                            ),
                            number=1,
                        )
                        run_times = run_times.at[k].set(execution_time)

                    avg_time_per_timestep = jnp.mean(run_times) / N_steps

                    print(f"N={N}, time per step={avg_time_per_timestep:.6e}")

                    all_run_times_particle[str(N)] = avg_time_per_timestep

                fname = f"data/particle_run_times_{noise_type}_{dim}d.npz"
                jnp.savez(fname, **all_run_times_particle)

    if SPDE:
        for dim in [1, 2]:
            n = spde_grids[dim]

            L = 2 * jnp.pi
            x0 = tuple(0 for _ in range(dim))
            x_final = tuple(L for _ in range(dim))
            grid = fl.make_grid(x0, x_final, n)

            for noise_type in noise_types:
                if noise_type == "vanishing-common":
                    use_k = False
                elif noise_type == "vanishing-iid":
                    use_k = True
                all_run_times_spde = {}

                print(f"SPDE: dim={dim}, noise_type={noise_type}")

                for N in Ns:
                    # run once for JIT
                    state0, vbar0 = _make_ic_spde(N, dim, grid, jr.key(0), use_k=use_k)
                    _ = jax.block_until_ready(
                        run_spde(state0, vbar0, dim, noise_type, N, jr.key(0))
                    )

                    run_times = jnp.zeros(N_runs)
                    keys = jr.split(jr.key(SEED), N_runs)

                    for k in range(N_runs):
                        k1, k2 = jr.split(keys[k], 2)

                        # Build IC outside the timed solve.
                        state0, vbar0 = jax.block_until_ready(
                            _make_ic_spde(N, dim, grid, k1, use_k=use_k)
                        )

                        execution_time = timeit.timeit(
                            lambda: jax.block_until_ready(
                                run_spde(state0, vbar0, dim, noise_type, N, k2)
                            ),
                            number=1,
                        )

                        run_times = run_times.at[k].set(execution_time)

                    avg_time_per_timestep = jnp.mean(run_times) / N_steps

                    print(f"N={N}, time per step={avg_time_per_timestep:.6e}")

                    all_run_times_spde[str(N)] = avg_time_per_timestep

                n_label = "x".join(str(m) for m in n)
                fname = f"data/spde_run_times_{noise_type}_{dim}d_{n_label}.npz"
                jnp.savez(fname, **all_run_times_spde)

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

        Ns = jnp.array([2**k for k in range(3, 13)])

        spde_grid = {
            1: "256",
            2: "64x64",
        }

        def load_costs(path):
            data = jnp.load(path)
            return jnp.array([data[str(int(N))] for N in Ns])

        def make_plot(dim):
            fig, ax = plt.subplots(figsize=(6.0, 4.0))

            n_label = spde_grid[dim]

            particle_common = load_costs(
                f"data/particle_run_times_vanishing-common_{dim}d.npz"
            )
            particle_iid = load_costs(
                f"data/particle_run_times_vanishing-iid_{dim}d.npz"
            )

            spde_common = load_costs(
                f"data/spde_run_times_vanishing-common_{dim}d_{n_label}.npz"
            )
            spde_iid = load_costs(
                f"data/spde_run_times_vanishing-iid_{dim}d_{n_label}.npz"
            )

            ax.plot(
                Ns,
                particle_common,
                color="black",
                marker="o",
                lw=2.2,
                markersize=4,
                label="Particle (common)",
                zorder=4,
            )

            ax.plot(
                Ns,
                particle_iid,
                color="0.45",
                marker="o",
                lw=2.2,
                markersize=4,
                linestyle="--",
                label="Particle (i.i.d.)",
                zorder=4,
            )

            # SPDEs: coloured, square markers
            ax.plot(
                Ns,
                spde_common,
                color=plt.cm.plasma(0.25),
                marker="s",
                lw=2.0,
                markersize=4,
                label="SPDE (common)",
                zorder=3,
            )

            ax.plot(
                Ns,
                spde_iid,
                color=plt.cm.plasma(0.75),
                marker="s",
                lw=2.0,
                markersize=4,
                linestyle="--",
                label="SPDE (i.i.d.)",
                zorder=3,
            )

            ax.set_xlabel(r"$N$")
            ax.set_ylabel(r"Average CPU time per $\Delta t$ (seconds)")

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
            ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)

            ax.legend(frameon=False, loc="best")

            fig.tight_layout()

            fig.savefig(
                f"figures/performance_both_{dim}d.pdf", dpi=300, bbox_inches="tight"
            )

            plt.show()

        make_plot(dim=1)
        make_plot(dim=2)
