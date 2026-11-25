from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Parameters
Delta = 1.0
J = 15.0
kappa = J / np.pi

h_min = -8.0
h_max = 1.0


# Node-focus transition
x_nf = kappa / 2
h_nf = -kappa**2 / 4 - Delta**2 / kappa**2


# Equilibrium curve, with extra resolution near the node-focus point
x = np.sort(np.unique(np.concatenate((
    np.linspace(0.03, 5.5, 30000),
    np.linspace(x_nf - 0.08, x_nf + 0.08, 5000),
    [x_nf],
))))

h = x**2 - kappa * x - Delta**2 / (4 * x**2)


# Calculate both eigenvalues
root = np.lib.scimath.sqrt(2 * kappa * x - 4 * x**2)

lambda_plus = -Delta / x + root
lambda_minus = -Delta / x - root

stable = (lambda_plus.real < 0) & (lambda_minus.real < 0)

node = stable & (root.imag == 0)
focus = stable & (root.imag != 0)
saddle = (lambda_plus.real > 0) & (lambda_minus.real < 0)


# Saddle-node points
fold_roots = np.roots([4, -2 * kappa, 0, 0, Delta**2])

physical = (
    (np.abs(fold_roots.imag) < 1e-9)
    & (fold_roots.real > 0)
)

fold_x = np.sort(fold_roots.real[physical])

fold_h = (
    fold_x**2
    - kappa * fold_x
    - Delta**2 / (4 * fold_x**2)
)

lower_index = np.argmin(fold_h)
upper_index = np.argmax(fold_h)

h_lower = fold_h[lower_index]
h_upper = fold_h[upper_index]


# Scientific plotting style
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "svg.fonttype": "path",
})


def point_label(name, h_value, x_value):
    return (
        name + "\n"
        + rf"$h={h_value:.6f}$" + "\n"
        + rf"$x_*={x_value:.6f}$"
    )


def draw_branches(ax):
    ax.plot(
        np.where(node, h, np.nan), x,
        color="0.15",
        lw=1.2,
        label="Stable node"
    )

    ax.plot(
        np.where(focus, h, np.nan), x,
        color="#36648b",
        lw=1.2,
        label="Stable focus"
    )

    ax.plot(
        np.where(saddle, h, np.nan), x,
        color="0.45",
        lw=1.1,
        ls="--",
        label="Saddle"
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        width=0.7,
        length=3
    )

    ax.grid(False)


def draw_closeup(ax, fontsize=7):
    draw_branches(ax)

    # Lower saddle-node marker
    ax.scatter(
        h_lower, fold_x[lower_index],
        s=28,
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=4
    )

    # Node-focus marker
    ax.scatter(
        h_nf, x_nf,
        marker="^",
        s=35,
        facecolors="#36648b",
        edgecolors="white",
        linewidths=0.4,
        zorder=5
    )

    ax.set(
        xlim=(h_lower - 0.00005, h_lower + 0.0008),
        ylim=(2.338, 2.404),
        xlabel=r"$h$",
        ylabel=r"$x_*$"
    )

    ax.set_xticks([-5.7435, -5.7432, -5.7429])
    ax.set_yticks([2.34, 2.36, 2.38, 2.40])

    ax.ticklabel_format(
        axis="x",
        style="plain",
        useOffset=False
    )

    # Node-focus coordinates
    ax.annotate(
        point_label("Node-focus", h_nf, x_nf),
        xy=(h_nf, x_nf),
        xytext=(0.06, 0.98),
        textcoords="axes fraction",
        fontsize=fontsize,
        ha="left",
        va="top",
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.6,
            "color": "0.4"
        }
    )

    # Lower saddle-node coordinates
    ax.annotate(
        point_label(
            "Saddle-node",
            h_lower,
            fold_x[lower_index]
        ),
        xy=(h_lower, fold_x[lower_index]),
        xytext=(0.42, 0.41),
        textcoords="axes fraction",
        fontsize=fontsize,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.6,
            "color": "0.4"
        }
    )


# Main figure
fig, ax = plt.subplots(
    figsize=(8, 4),
    layout="constrained"
)

draw_branches(ax)

ax.scatter(
    fold_h, fold_x,
    s=24,
    facecolors="white",
    edgecolors="black",
    linewidths=0.8,
    label="Saddle-node",
    zorder=4
)

ax.scatter(
    h_nf, x_nf,
    marker="^",
    s=30,
    facecolors="#36648b",
    edgecolors="white",
    linewidths=0.4,
    label="Node-focus",
    zorder=5
)

ax.set(
    xlim=(h_min, h_max),
    ylim=(0, 5.2),
    xlabel=r"$h$",
    ylabel=r"$x_*$"
)

ax.set_xticks([-8, -6, -4, -2, 0])
ax.set_yticks([0, 1, 2, 3, 4, 5])

ax.legend(
    frameon=False,
    loc="upper left",
    fontsize=8,
    handlelength=2,
    labelspacing=0.35
)

# Upper saddle-node coordinates
ax.annotate(
    point_label(
        "Saddle-node",
        h_upper,
        fold_x[upper_index]
    ),
    xy=(h_upper, fold_x[upper_index]),
    xytext=(-4.5, 0.65),
    fontsize=8,
    ha="center",
    va="center",
    arrowprops={
        "arrowstyle": "-",
        "lw": 0.6,
        "color": "0.4"
    }
)

# Magnified inset
zoom = ax.inset_axes([0.62, 0.17, 0.35, 0.50])
draw_closeup(zoom)


# Standalone zoom
fig_zoom, ax_zoom = plt.subplots(
    figsize=(4, 3),
    layout="constrained"
)

draw_closeup(ax_zoom, fontsize=9)


# Output directory: works in scripts and notebooks
out = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

# Save main figure
fig.savefig(out / "mpr_bifurcation.pdf")
fig.savefig(out / "mpr_bifurcation.svg")
fig.savefig(out / "mpr_bifurcation.png", dpi=600)

# Save standalone zoom
fig_zoom.savefig(out / "mpr_node_focus_zoom.pdf")
fig_zoom.savefig(out / "mpr_node_focus_zoom.svg")
fig_zoom.savefig(out / "mpr_node_focus_zoom.png", dpi=600)


# Print special-point coordinates
print(f"{'Special point':<26} {'h':>14} {'x_star':>14}")

special_points = [
    ("Saddle-node (lower h)", h_lower, fold_x[lower_index]),
    ("Saddle-node (upper h)", h_upper, fold_x[upper_index]),
    ("Node-focus", h_nf, x_nf),
]

for name, h_value, x_value in special_points:
    print(f"{name:<26} {h_value:14.9f} {x_value:14.9f}")

plt.show()