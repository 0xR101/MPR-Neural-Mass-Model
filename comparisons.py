import csv
import numpy as np
from scipy.signal import find_peaks
from mpr_numerical import Delta, J, tau, out


# Read the two constant-input trajectories saved by the simulation.
data = np.genfromtxt(out / "constant.csv", delimiter=",", names=True)
s = data["t"] / tau
x_low = np.pi * tau * data["r_low"]
x_high = np.pi * tau * data["r_high"]
h = float(data["h"][0])
if not np.allclose(data["h"], h):
    raise ValueError("This comparison requires constant h.")


# Analytical equilibrium and eigenvalue formulas.
kappa = J / np.pi
roots = np.roots([1, -kappa, -h, 0, -Delta**2 / 4])
x_eq = np.sort(roots.real[(np.abs(roots.imag) < 1e-10)
                          & (roots.real > 0)])
if len(x_eq) != 3:
    raise ValueError("This table is for the three-equilibrium example.")

x_low_star = x_eq[0]
x_high_star = x_eq[-1]
D_high = 2 * kappa * x_high_star - 4 * x_high_star**2
if D_high >= 0:
    raise ValueError("The high-rate equilibrium must be a stable focus.")

gamma_analytical = Delta / x_high_star
omega_analytical = np.sqrt(-D_high)
period_analytical = 2 * np.pi / omega_analytical


# Numerical equilibrium estimates: values at the final saved time.
x_low_numerical = x_low[-1]
x_high_numerical = x_high[-1]


# Numerical period: average spacing between successive maxima.
peaks, _ = find_peaks(x_high)
peak_times = s[peaks]
period_window = (peak_times > 2) & (peak_times < 15)
selected_times = peak_times[period_window]
if len(selected_times) < 3:
    raise ValueError("Not enough peaks in 2 < t/tau < 15.")
period_numerical = np.mean(np.diff(selected_times))


# Numerical damping: log(peak amplitude) = intercept - gamma*s.
# Amplitudes are measured relative to the predicted high equilibrium.
amplitudes = x_high[peaks] - x_high_star
decay_window = ((peak_times > 3) & (peak_times < 15)
                & (amplitudes > 0))
if np.count_nonzero(decay_window) < 3:
    raise ValueError("Not enough positive peaks in 3 < t/tau < 15.")
slope, intercept = np.polyfit(
    peak_times[decay_window], np.log(amplitudes[decay_window]), 1
)
gamma_numerical = -slope


rows = [
    ("Low-state equilibrium x*", x_low_star, x_low_numerical),
    ("High-state equilibrium x*", x_high_star, x_high_numerical),
    ("Period T/tau", period_analytical, period_numerical),
    ("Decay coefficient gamma", gamma_analytical, gamma_numerical),
]

table = ["| Quantity | Analytical | Numerical | Relative difference (%) |",
         "|---|---:|---:|---:|"]
csv_rows = []
for quantity, analytical, numerical in rows:
    difference = 100 * abs(numerical - analytical) / abs(analytical)
    table.append(f"| {quantity} | {analytical:.8f} | {numerical:.8f} "
                 f"| {difference:.6g} |")
    csv_rows.append([quantity, analytical, numerical, difference])

text = "\n".join(table) + "\n"
print(text)

with (out / "comparison_table.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Quantity", "Analytical", "Numerical",
                     "Relative difference (%)"])
    writer.writerows(csv_rows)

notes = ("\nConstant h = " + str(h)
         + ". Parameters must match the saved simulation.\n\n"
         "Period: mean peak spacing for 2 < t/tau < 15. "
         "Decay: straight-line fit of log(x_peak - x_high_star) "
         "for 3 < t/tau < 15. The decay estimate uses the analytical "
         "equilibrium as its baseline.\n\n"
         "Relative differences include finite-time relaxation, "
         "finite-amplitude effects and sampled-peak estimation; "
         "they are not solely integration errors.\n")
(out / "comparison_table.md").write_text(text + notes)
