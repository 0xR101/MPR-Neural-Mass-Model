from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


Delta = 1.0
J = 15.0
eta = -5.0
tau = 1.0
h_min = -8.0
h_max = 1.0
base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
out = base / "mpr_numerical_results"


def rhs(t, y, current):
    r, v = y
    h = eta + current(t)
    dr = Delta / (np.pi * tau**2) + 2 * r * v / tau
    dv = (v**2 - (np.pi * tau * r)**2 + J * tau * r + h) / tau
    return [dr, dv]


def integrate(current, start, stop, y0, rtol=1e-8, atol=1e-10,
              max_step=None):
    if max_step is None:
        max_step = 0.05 * tau
    n = int(round((stop - start) / (0.01 * tau))) + 1
    times = np.linspace(start, stop, n)
    sol = solve_ivp(rhs, (start, stop), y0, args=(current,),
                    method="RK45", t_eval=times, rtol=rtol, atol=atol,
                    max_step=max_step)
    if not sol.success:
        raise RuntimeError(sol.message)
    if not np.all(np.isfinite(sol.y)) or np.min(sol.y[0]) < -10 * atol:
        raise RuntimeError("Nonfinite solution or negative firing rate.")
    return sol.t, sol.y


def no_input(t):
    return 0.0


def pulse_on(t):
    return 3.0


def sinusoidal_input(t):
    h_mid = (h_min + h_max) / 2
    h_amplitude = (h_max - h_min) / 2
    h = h_mid + h_amplitude * np.sin(np.pi * t / (20 * tau))
    return h - eta


def equilibria(h):
    roots = np.roots([1, -J / np.pi, -h, 0, -Delta**2 / 4])
    x_eq = np.sort(roots.real[(np.abs(roots.imag) < 1e-10)
                              & (roots.real > 0)])
    return np.column_stack((x_eq / (np.pi * tau), -Delta / (2 * x_eq)))


def run_cases(**options):
    # Positive roots give the physical equilibria for I = 0.
    eq = equilibria(eta)
    low0 = eq[0] + [0.1 * eq[0, 0], 0.1]
    high0 = eq[-1] + [0.1 * eq[-1, 0], 0.1]

    t, low = integrate(no_input, 0, 40 * tau, low0, **options)
    _, high = integrate(no_input, 0, 40 * tau, high0, **options)

    # Restart at each input jump, keeping r and v continuous.
    t1, y1 = integrate(no_input, 0, 10 * tau, eq[0], **options)
    t2, y2 = integrate(pulse_on, 10 * tau, 40 * tau, y1[:, -1], **options)
    t3, y3 = integrate(no_input, 40 * tau, 80 * tau, y2[:, -1], **options)
    tp = np.concatenate((t1, t2[1:], t3[1:]))
    pulse = np.concatenate((y1, y2[:, 1:], y3[:, 1:]), axis=1)

    # Match the initial state to the sinusoidal drive's initial h = -3.5.
    sine0 = equilibria(eta + sinusoidal_input(0))[0]
    ts, sine = integrate(sinusoidal_input, 0, 160 * tau, sine0, **options)
    return eq, t, low, high, tp, pulse, ts, sine


def format_axes(axes):
    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True,
                       width=0.7, length=3)
        ax.margins(x=0)


def save_figure(fig, name):
    fig.savefig(out / f"{name}.png", dpi=600)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.svg")
    # Leave the figure open so plt.show() can display all three cases.


def main():
    out.mkdir(exist_ok=True)
    eq, t, low, high, tp, pulse, ts, sine = run_cases()

    # Recompute every trajectory with tighter settings on the same output grid.
    reference = run_cases(rtol=1e-10, atol=1e-12, max_step=0.025 * tau)
    labels = ["Constant: low", "Constant: high", "Pulse", "Sinusoidal"]
    solutions = [low, high, pulse, sine]
    refined = [reference[2], reference[3], reference[5], reference[7]]
    print(f"{'Case':<20} {'max |dx|':>15} {'max |dv|':>15}")
    for name, y, yr in zip(labels, solutions, refined):
        difference = np.max(np.abs(y - yr), axis=1)
        print(f"{name:<20} {np.pi*tau*difference[0]:15.3e} {difference[1]:15.3e}")

    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 11, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.fonttype": "path"
    })
    blue = "#36648b"

    # Static saddle-node boundaries at the chosen J and Delta.
    # These are reference values for h, not exact switching times under forcing.
    kappa = J / np.pi
    roots = np.roots([4, -2 * kappa, 0, 0, Delta**2])
    xf = roots.real[(np.abs(roots.imag) < 1e-10) & (roots.real > 0)]
    fold_h = np.sort(-xf**2 - 3 * Delta**2 / (4 * xf**2))
    print("Static saddle-node h values:", fold_h)

    def drive_panel(ax, times, values):
        ax.plot(times, values, color=blue, lw=1.1)
        for value in fold_h:
            ax.axhline(value, color="0.55", lw=0.7, ls="--")
        ax.set_ylabel(r"$h(t)$")
        ax.set_xlabel(r"$t/\tau$")
        ax.set_ylim(h_min, h_max)
        ax.set_yticks([-8, -6, -4, -2, 0, 1])

    fig, ax = plt.subplots(3, 1, figsize=(3.5, 4.5), sharex=True,
                           layout="constrained")
    ax[0].plot(t / tau, np.pi * tau * low[0], color="0.15", lw=1.1,
               label="Near low-rate state")
    ax[0].plot(t / tau, np.pi * tau * high[0], color=blue, lw=1.1,
               label="Near high-rate state")
    ax[1].plot(t / tau, low[1], color="0.15", lw=1.1)
    ax[1].plot(t / tau, high[1], color=blue, lw=1.1)
    ax[0].set_ylabel(r"$x(t)=\pi\tau r(t)$")
    ax[1].set_ylabel(r"$v(t)$")
    drive_panel(ax[2], t / tau, np.full_like(t, eta))
    ax[0].legend(frameon=False, fontsize=8, loc="center right")
    format_axes(ax)
    save_figure(fig, "mpr_constant")

    fig, ax = plt.subplots(3, 1, figsize=(3.5, 4.5), sharex=True,
                           layout="constrained")
    ax[0].plot(tp / tau, np.pi * tau * pulse[0], color="0.15", lw=0.9)
    ax[1].plot(tp / tau, pulse[1], color="0.15", lw=0.9)
    drive_panel(ax[2], [0, 10, 10, 40, 40, 80],
                eta + np.array([0, 0, 3, 3, 0, 0]))
    ax[0].set_ylabel(r"$x(t)=\pi\tau r(t)$")
    ax[1].set_ylabel(r"$v(t)$")
    format_axes(ax)
    save_figure(fig, "mpr_pulse")

    fig, ax = plt.subplots(3, 1, figsize=(6.8, 4.5), sharex=True,
                           layout="constrained")
    ax[0].plot(ts / tau, np.pi * tau * sine[0], color="0.15", lw=0.8)
    ax[1].plot(ts / tau, sine[1], color="0.15", lw=0.8)
    drive_panel(ax[2], ts / tau, eta + sinusoidal_input(ts))
    ax[0].set_ylabel(r"$x(t)=\pi\tau r(t)$")
    ax[1].set_ylabel(r"$v(t)$")
    format_axes(ax)
    save_figure(fig, "mpr_sinusoidal")

    np.savetxt(out / "equilibria.csv", eq, delimiter=",",
               header="r_star,v_star", comments="")
    np.savetxt(out / "constant.csv",
               np.column_stack((t, low.T, high.T, np.full_like(t, eta))),
               delimiter=",", header="t,r_low,v_low,r_high,v_high,h", comments="")
    hp = eta + np.where((tp >= 10 * tau) & (tp < 40 * tau), 3.0, 0.0)
    np.savetxt(out / "pulse.csv", np.column_stack((tp, pulse.T, hp)),
               delimiter=",", header="t,r,v,h", comments="")
    np.savetxt(out / "sinusoidal.csv",
               np.column_stack((ts, sine.T, eta + sinusoidal_input(ts))),
               delimiter=",", header="t,r,v,h", comments="")
    print(f"Results saved in {out}")
    plt.show()


if __name__ == "__main__":
    main()
