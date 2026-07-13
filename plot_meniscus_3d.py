"""
3D plot of meniscus rise/fall vs tau_g and tau_f, from param_study_phase1_tau_results.csv.

Usage: python plot_meniscus_3d.py
Reads param_study_phase1_tau_results.csv (must be in the same folder) and
writes meniscus_vs_tau_g_tau_f.png.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'param_study_phase1_tau_results.csv')
OUT_PATH = os.path.join(SCRIPT_DIR, 'meniscus_vs_tau_g_tau_f.png')


def load_rows():
    with open(CSV_PATH, 'r', newline='') as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()

    tau_g_vals = sorted({float(r['tau_g']) for r in rows})
    tau_f_vals = sorted({float(r['tau_f']) for r in rows})

    grid = np.full((len(tau_g_vals), len(tau_f_vals)), np.nan)
    failed_points = []

    for r in rows:
        i = tau_g_vals.index(float(r['tau_g']))
        j = tau_f_vals.index(float(r['tau_f']))
        if r['result'] == 'DONE' and r['meniscus_rise'] not in ('', None):
            grid[i, j] = float(r['meniscus_rise'])
        else:
            failed_points.append((float(r['tau_g']), float(r['tau_f'])))

    n_missing = np.isnan(grid).sum()
    if n_missing:
        print(f"Warning: {n_missing}/{grid.size} combos missing or failed "
              f"(NaN in plot): {failed_points}")

    TG, TF = np.meshgrid(tau_g_vals, tau_f_vals, indexing='ij')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(TG, TF, grid, cmap='viridis', edgecolor='k',
                            linewidth=0.3, antialiased=True)

    # mark failed/missing combos explicitly so they're not silently blank
    for tg, tf in failed_points:
        ax.scatter([tg], [tf], [0], color='red', marker='x', s=60, depthshade=False)

    ax.set_xlabel('tau_g')
    ax.set_ylabel('tau_f')
    ax.set_zlabel('meniscus rise/fall (lattice units)')
    ax.set_title('Meniscus rise/fall vs tau_g and tau_f (vf_theta=60)')
    fig.colorbar(surf, shrink=0.6, aspect=12, label='meniscus rise/fall')

    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
