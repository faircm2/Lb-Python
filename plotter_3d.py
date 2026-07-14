import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


class Plotter3D:
    """
    Class to handle post-processing plots for a param_study.py sweep:
    - 3D surface of a result metric (e.g. meniscus_rise) vs two swept params

    (Superimposed profile-curve overlays moved to Plotter2D.plot_profile_overlays
    -- those are 2D line plots, not 3D.)
    """
    def __init__(self, script_dir, results_csv, debug_log=None):
        self.script_dir = script_dir
        self.results_csv = results_csv
        self.debug_log = debug_log or (lambda *a, **k: None)

    def load_rows(self):
        with open(self.results_csv, 'r', newline='') as f:
            return list(csv.DictReader(f))

    def plot_metric_surface(self, x_param, y_param, z_field,
                             out_path='metric_vs_params.png',
                             title=None, z_label=None):
        """3D surface of z_field vs (x_param, y_param), built from the
        results CSV. Failed/missing combos are marked with a red X at z=0
        instead of being silently blank."""
        rows = self.load_rows()

        x_vals = sorted({float(r[x_param]) for r in rows})
        y_vals = sorted({float(r[y_param]) for r in rows})

        grid = np.full((len(x_vals), len(y_vals)), np.nan)
        failed_points = []

        for r in rows:
            i = x_vals.index(float(r[x_param]))
            j = y_vals.index(float(r[y_param]))
            if r['result'] == 'DONE' and r[z_field] not in ('', None):
                grid[i, j] = float(r[z_field])
            else:
                failed_points.append((float(r[x_param]), float(r[y_param])))

        n_missing = np.isnan(grid).sum()
        if n_missing:
            self.debug_log('WARN', f"{n_missing}/{grid.size} combos missing or "
                            f"failed (NaN in plot): {failed_points}")

        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, grid, cmap='viridis', edgecolor='k',
                                linewidth=0.3, antialiased=True)

        for xv, yv in failed_points:
            ax.scatter([xv], [yv], [0], color='red', marker='x', s=60, depthshade=False)

        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_zlabel(z_label or z_field)
        ax.set_title(title or f'{z_field} vs {x_param} and {y_param}')
        fig.colorbar(surf, shrink=0.6, aspect=12, label=z_label or z_field)

        full_out_path = os.path.join(self.script_dir, out_path)
        fig.savefig(full_out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        self.debug_log('INIT', f'Saved 3D surface: {full_out_path}')
        return full_out_path
