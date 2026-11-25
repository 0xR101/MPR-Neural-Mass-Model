# MPR Model — Running the Code

## Requirements

Keep all Python scripts in the same folder and install:

```bash
python3 -m pip install numpy scipy matplotlib
```

## Scripts and Results

Run these commands from the folder containing the scripts:

| Command | Results |
|---|---|
| `python3 mpr_bifurcation.py` | Bifurcation diagram and node–focus zoom |
| `python3 mpr_numerical.py` | Time responses to constant, pulsed, and sinusoidal inputs |
| `python3 mpr_comparison_table.py` | Analytical–numerical comparison table |
| `python3 mpr_3d_dynamics.py` | 3D trajectories near equilibria and special points |
| `python3 mpr_equilibrium_dynamics.py` | Optional 2D time plots near equilibria |
| `python3 mpr_solver_comparison.py` | Optional comparison of five numerical integration methods |

**Run `mpr_numerical.py` before `mpr_comparison_table.py`.** The other scripts run independently.

## Output Locations

- Bifurcation figures: beside the script.
- Input responses and comparison table: `mpr_numerical_results/`.
- 3D trajectories: `mpr_3d_results/`.
- Optional 2D plots: `mpr_equilibrium_dynamics_results/`.
- Solver comparison: `mpr_solver_comparison_results/`.

Figures are saved as PNG, PDF, and SVG. Close the plot windows to finish a run. Use PDF figures for LaTeX.
