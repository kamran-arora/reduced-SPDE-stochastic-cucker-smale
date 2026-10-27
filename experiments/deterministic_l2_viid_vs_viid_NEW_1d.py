import math
from timeit import default_timer as timer

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import flocking as fl

jax.config.update("jax_enable_x64", True)

"""
Compare the original and improved reduced VIID models in their deterministic
limit as the initial velocity spread is varied.

For each random initial condition, the evolution is deterministic: the particle system is solved as an ODE and the two VIID models are solved with sigma=0. At time T, terminal particle fields are mollified on the PDE grid and the pathwise errors

    ||rho_eps - rho||_{L2}^2,
    ||j_eps - j||_{L2}^2,
    ||k_eps - k||_{L2}^2

are computed separately for both reduced models.
"""


def _make_initial_particles(N, velocity_width, key, vbar_target=-6.0):
    position_key, velocity_key = jr.split(key)
    xs0 = jr.uniform(key=position_key, shape=(N,), minval=-20.0, maxval=20.0)
    vs0 = jr.uniform(
        key=velocity_key,
        shape=(N,),
        minval=vbar_target - velocity_width / 2.0,
        maxval=vbar_target + velocity_width / 2.0,
    )
    return xs0, vs0


def _make_initial_particles(N, velocity_width, key, vbar_target=-6.0):
    if N < 3:
        raise ValueError("N must be at least 3")

    sizes = ((N + 2) // 3, (N + 1) // 3, N // 3)

    x_ranges = ((-27.0, -18.0), (-4.0, 2.0), (16.0, 28.0))
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
    xs0 = jnp.concatenate(xs)[perm]
    vs0 = jnp.concatenate(vs)[perm]

    # Preserve the cluster structure while varying the velocity spread.
    centered = vs0 - jnp.mean(vs0)
    initial_std = jnp.sqrt(jnp.mean(centered**2))

    vs0 = vbar_target + (velocity_width / jnp.sqrt(12.0)) * centered / initial_std

    return xs0, vs0


def _make_pde_initial_state(particles, grid, kappa, x0, x_final, Lx):
    rho0, j0, k0 = fl.density_from_particles_1d(particles, grid, kappa, Lx, use_k=True)
    vbar0 = jnp.mean(particles[1], axis=0)
    return (
        fl.ScalarField(x0, x_final, rho0),
        fl.ScalarField(x0, x_final, j0),
        fl.ScalarField(x0, x_final, k0),
        vbar0,
    )


@eqx.filter_jit
def _terminal_l2_errors(
    particles0,
    grid,
    kappa,
    Lx,
    T,
    dt0,
    lambda_,
    r,
    N,
    pde_key,
):
    x0 = (-Lx / 2.0,)
    x_final = (Lx / 2.0,)
    h = Lx / grid.shape[0]
    vbar0 = jnp.mean(particles0[1], axis=0)
    state0 = _make_pde_initial_state(particles0, grid, kappa, x0, x_final, Lx)
    initial_velocity_variance = jnp.mean((particles0[1] - vbar0) ** 2)

    args = fl.CuckerSmaleArgs(
        L=jnp.array([Lx]),
        lambda_=lambda_,
        r=r,
        sigma=0.0,
        vbar0=vbar0,
    )
    saveat = dfx.SaveAt(t1=True)

    xs, vs = fl.solve_CS_ODE(
        t0=0.0,
        t1=T,
        dt0=dt0,
        saveat=saveat,
        state0=particles0,
        args=args,
        kernel="full",
    )
    rho_eps, j_eps, k_eps = fl.density_from_particles_1d(
        (xs[-1], vs[-1]), grid, kappa, Lx, use_k=True
    )
    particle_fields = jnp.stack([rho_eps, j_eps, k_eps])

    rho, j, k, _ = fl.solve_spde_viid(
        t0=0.0,
        t1=T,
        dt0=dt0,
        saveat=saveat,
        state0=state0,
        args=args,
        dim=1,
        kernel_type="full",
        N=N,
        key=pde_key,
    )
    viid_fields = jnp.stack([rho.vals[-1], j.vals[-1], k.vals[-1]])

    rho_new, j_new, k_new, _ = fl.solve_spde_viid_NEW(
        t0=0.0,
        t1=T,
        dt0=dt0,
        saveat=saveat,
        state0=state0,
        args=args,
        dim=1,
        kernel_type="full",
        N=N,
        key=pde_key,
    )
    viid_new_fields = jnp.stack([rho_new.vals[-1], j_new.vals[-1], k_new.vals[-1]])

    viid_errors = h * jnp.sum((particle_fields - viid_fields) ** 2, axis=-1)
    viid_new_errors = h * jnp.sum((particle_fields - viid_new_fields) ** 2, axis=-1)
    return initial_velocity_variance, viid_errors, viid_new_errors


def run_experiment(
    realisations,
    velocity_widths,
    N,
    kappa,
    T,
    dt,
    lambda_,
    r,
    nx,
    Lx,
    seed,
    vbar_target=-6.0,
):
    x0 = (-Lx / 2.0,)
    x_final = (Lx / 2.0,)
    grid = fl.make_grid(x0, x_final, (nx,))[:, 0]
    velocity_widths = jnp.asarray(velocity_widths)
    shape = (velocity_widths.shape[0], realisations, 3)

    initial_velocity_variances = jnp.zeros((velocity_widths.shape[0], realisations))
    viid_pathwise_errors = jnp.zeros(shape)
    viid_new_pathwise_errors = jnp.zeros(shape)

    run_keys = jr.split(jr.key(seed), realisations)

    for width_idx, velocity_width in enumerate(velocity_widths):
        width_results = []
        time0 = timer()
        for realisation_idx in range(realisations):
            ic_key, pde_key = jr.split(run_keys[realisation_idx])
            particles0 = _make_initial_particles(
                N, velocity_width, ic_key, vbar_target=vbar_target
            )
            width_results.append(
                _terminal_l2_errors(
                    particles0=particles0,
                    grid=grid,
                    kappa=kappa,
                    Lx=Lx,
                    T=T,
                    dt0=dt,
                    lambda_=lambda_,
                    r=r,
                    N=N,
                    pde_key=pde_key,
                )
            )

        variance_values, viid_errors, viid_new_errors = [
            jnp.stack(values) for values in zip(*width_results)
        ]
        initial_velocity_variances = initial_velocity_variances.at[width_idx].set(
            variance_values
        )
        viid_pathwise_errors = viid_pathwise_errors.at[width_idx].set(viid_errors)
        viid_new_pathwise_errors = viid_new_pathwise_errors.at[width_idx].set(
            viid_new_errors
        )
        time1 = timer()
        print(
            f"velocity width={float(velocity_width):g}, "
            f"estimated initial spread={jnp.mean(variance_values):.6e} "
            f"finished in {time1 - time0:.4f}s\n"
            f"  VIID mean squared L2 error (rho, j, k)="
            f"{jnp.mean(viid_errors, axis=0)}\n"
            f"  VIID NEW mean squared L2 error (rho, j, k)="
            f"{jnp.mean(viid_new_errors, axis=0)}"
        )

    mean_initial_velocity_variances = jnp.mean(initial_velocity_variances, axis=1)
    mean_viid_errors = jnp.mean(viid_pathwise_errors, axis=1)
    mean_viid_new_errors = jnp.mean(viid_new_pathwise_errors, axis=1)

    if realisations > 1:
        scale = jnp.sqrt(realisations)
        initial_velocity_variance_sem = (
            jnp.std(initial_velocity_variances, axis=1, ddof=1) / scale
        )
        viid_error_sem = jnp.std(viid_pathwise_errors, axis=1, ddof=1) / scale
        viid_new_error_sem = jnp.std(viid_new_pathwise_errors, axis=1, ddof=1) / scale
    else:
        initial_velocity_variance_sem = jnp.full_like(
            mean_initial_velocity_variances, jnp.nan
        )
        viid_error_sem = jnp.full_like(mean_viid_errors, jnp.nan)
        viid_new_error_sem = jnp.full_like(mean_viid_new_errors, jnp.nan)

    return {
        "velocity_widths": velocity_widths,
        "initial_velocity_variances": initial_velocity_variances,
        "mean_initial_velocity_variances": mean_initial_velocity_variances,
        "initial_velocity_variance_sem": initial_velocity_variance_sem,
        "viid_pathwise_errors": viid_pathwise_errors,
        "viid_new_pathwise_errors": viid_new_pathwise_errors,
        "mean_viid_errors": mean_viid_errors,
        "mean_viid_new_errors": mean_viid_new_errors,
        "viid_error_sem": viid_error_sem,
        "viid_new_error_sem": viid_new_error_sem,
        "N": jnp.asarray(N),
        "kappa": jnp.asarray(kappa),
        "T": jnp.asarray(T),
        "dt": jnp.asarray(dt),
        "lambda_": jnp.asarray(lambda_),
        "r": jnp.asarray(r),
        "nx": jnp.asarray(nx),
        "h": jnp.asarray(Lx / nx),
        "Lx": jnp.asarray(Lx),
        "seed": jnp.asarray(seed),
        "vbar_target": jnp.asarray(vbar_target),
    }


def _set_plot_style(use_tex=True):
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.linewidth": 0.8,
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        }
    )


def _log_tick_with_plain_one(value, _position):
    if value <= 0.0:
        return ""
    if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return "1.0"
    if math.isclose(value, 10.0, rel_tol=0.0, abs_tol=1e-12):
        return "10.0"

    exponent = math.log10(value)
    rounded_exponent = round(exponent)
    if math.isclose(exponent, rounded_exponent, rel_tol=0.0, abs_tol=1e-10):
        return rf"$10^{{{int(rounded_exponent)}}}$"
    return ""


def _include_log_y_ticks(ax, tick_values):
    ymin, ymax = ax.get_ylim()
    tick_values = [tick for tick in tick_values if tick > 0.0]
    ymin = min(ymin, min(tick_values) / 1.05)
    ymax = max(ymax, max(tick_values) * 1.05)
    ax.set_ylim(ymin, ymax)

    ticks = [tick for tick in ax.get_yticks() if ymin <= tick <= ymax]
    ticks.extend(tick_values)
    ax.set_yticks(sorted(set(ticks)))


def _log_padded_ylim(values, tick_values=(1.0, 10.0), pad_fraction=0.05):
    values = jnp.asarray(values).reshape(-1)
    tick_values = jnp.asarray(tick_values)
    tick_values = tick_values[tick_values > 0.0]
    values = values[jnp.isfinite(values) & (values > 0.0)]
    values = jnp.concatenate([values, tick_values])

    log_ymin = float(jnp.log10(jnp.min(values)))
    log_ymax = float(jnp.log10(jnp.max(values)))
    log_span = max(log_ymax - log_ymin, 1.0)
    return (
        10 ** (log_ymin - pad_fraction * log_span),
        10 ** (log_ymax + pad_fraction * log_span),
    )


def _shared_improvement_ylim(data, field_indices=(0, 1, 2)):
    factors, _factor_sem = _improvement_factor_summary(data)
    return _log_padded_ylim(factors[:, field_indices])


def _approx_symbol(field_symbol, is_new):
    if is_new:
        return rf"{field_symbol}_{{\varepsilon}}^{{(\mathrm{{new}})}}"
    return rf"{field_symbol}_{{\varepsilon}}"


def _set_legend_linewidth(legend, linewidth=2.4):
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    for handle in handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(linewidth)


def plot_results(data, use_tex=True):
    _set_plot_style(use_tex=use_tex)

    x = data["mean_initial_velocity_variances"]
    fields = [(r"$\rho$", "rho", r"\rho"), (r"$j$", "j", r"j"), (r"$k$", "k", r"k")]
    models = [
        (False, "mean_viid_errors", "black", "-"),
        (
            True,
            "mean_viid_new_errors",
            "tab:cyan",
            "-",
        ),
    ]
    figs = []

    for field_idx, (_field_label, save_label, field_symbol) in enumerate(fields):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for is_new, error_key, color, linestyle in models:
            estimate = data[error_key][:, field_idx]
            ax.plot(
                x,
                estimate,
                marker="o",
                markersize=2.5,
                color=color,
                linestyle=linestyle,
                label=rf"${_approx_symbol(field_symbol, is_new)}$",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_with_plain_one))
        ax.set_xlabel(r"$\mathbb{E}\left[(v_i(0)-\overline{v}(0))^2\right]$")
        ax.set_ylabel(
            rf"$\mathbb{{E}}\Vert {field_symbol}"
            rf"-{field_symbol}_{{\varepsilon}}\Vert^2$"
        )
        ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                marker="s",
                markersize=2.5,
                lw=2.4,
                label=rf"${_approx_symbol(field_symbol, is_new)}$",
            )
            for is_new, _error_key, color, linestyle in models
        ]
        ax.legend(handles=legend_handles, frameon=False, handlelength=3.0)
        fig.tight_layout()
        figs.append((fig, save_label))

    return figs


def _improvement_factor_summary(data):
    old_errors = jnp.asarray(data["viid_pathwise_errors"])
    new_errors = jnp.asarray(data["viid_new_pathwise_errors"])
    old_means = jnp.mean(old_errors, axis=1)
    new_means = jnp.mean(new_errors, axis=1)
    factors = old_means / new_means

    realisations = old_errors.shape[1]
    if realisations > 1:
        # Paired delta-method SEM for the ratio of expected errors.
        residuals = old_errors - factors[:, None, :] * new_errors
        factor_sem = jnp.std(residuals, axis=1, ddof=1) / (
            jnp.sqrt(realisations) * new_means
        )
    else:
        factor_sem = jnp.full_like(factors, jnp.nan)
    return factors, factor_sem


def plot_improvement_factors(data, field_indices=(0,), use_tex=True, ylim=None):
    _set_plot_style(use_tex=use_tex)

    x = data["mean_initial_velocity_variances"]
    factors, _factor_sem = _improvement_factor_summary(data)
    fields = [
        (r"$\rho$", r"\rho", "black", "s"),
        (r"$j$", r"j", "black", "s"),
        (r"$k$", r"k", "black", "s"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for field_idx in field_indices:
        field_label, _field_symbol, color, marker = fields[field_idx]
        estimate = factors[:, field_idx]
        ax.plot(
            x,
            estimate,
            marker=marker,
            color=color,
            linestyle="-",
            label=field_label,
        )

    ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1.8, label="equal error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_with_plain_one))
    if ylim is not None:
        ax.set_ylim(*ylim)
    _include_log_y_ticks(ax, (1.0, 10.0))
    ax.set_xlabel(r"$\mathbb{E}\left[(v_i(0)-\overline{v}(0))^2\right]$")
    field_symbols = [fields[field_idx][1] for field_idx in field_indices]
    field_symbol = field_symbols[0] if len(field_symbols) == 1 else r"f"
    ax.set_ylabel(
        rf"$\mathbb{{E}}\Vert {field_symbol}"
        rf"-{_approx_symbol(field_symbol, False)}\Vert^2"
        rf"/\mathbb{{E}}\Vert {field_symbol}"
        rf"-{_approx_symbol(field_symbol, True)}\Vert^2$"
    )
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)
    legend = ax.legend(frameon=False, loc="upper left")
    _set_legend_linewidth(legend)
    fig.tight_layout()
    return fig


def main():
    RUN = True
    SAVE = True
    PLOT = True
    filename = "data/deterministic_l2_viid_vs_viid_NEW_1d_1024"

    if RUN:
        results = run_experiment(
            realisations=100,
            # Deliberately includes the extreme-spread regime so that the
            # improvement and the eventual closure breakdown are both visible.
            velocity_widths=jnp.geomspace(1.0, 50.0, 20),
            N=1024,
            kappa=64.0,
            T=2.0,
            dt=0.005,
            lambda_=50.0,
            r=0.5,
            nx=256,
            Lx=100.0,
            seed=123,
            vbar_target=0.0,
        )

    if SAVE:
        jnp.savez(filename, **results)

    if PLOT:
        if not RUN:
            results = jnp.load(filename + ".npz")
        for fig, save_label in plot_results(results):
            fig.savefig(
                f"figures/deterministic_l2_viid_vs_viid_NEW_1d_{save_label}_1024.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        improvement_ylim = _shared_improvement_ylim(results)
        for field_idx, save_label in [(0, "rho"), (1, "j"), (2, "k")]:
            improvement_fig = plot_improvement_factors(
                results, field_indices=(field_idx,), ylim=improvement_ylim
            )
            improvement_fig.savefig(
                "figures/"
                f"deterministic_l2_viid_vs_viid_NEW_1d_{save_label}"
                "_improvement_factor_1024.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()


if __name__ == "__main__":
    main()
