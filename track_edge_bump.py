"""
Tracks the meniscus edge-bump size across multiple phi_matrix_<iteration>.txt
snapshots (dumped via save_phi_results at iterationsOfInterest checkpoints),
so its evolution over the run can be seen directly instead of comparing only
two endpoints.

Usage: python track_edge_bump.py
Reads all phi_matrix_*.txt in this folder, prints edge-bump size (peak minus
far-field baseline) for each, sorted by iteration.
"""
import glob
import os
import re

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_field(path):
    with open(path, 'r') as f:
        f.readline()  # header: phi_star_G phi_star_L (reference only, field is normalized 0-1)
        data = np.loadtxt(f)
    return data[::-1, :].T  # full[x, y]


def edge_bump_size(full, edge_x, baseline_x_offset=20):
    """Interface y-position at edge_x minus the baseline a bit further in,
    using the same phi=0.5 crossing logic as the meniscus profile work."""
    x_ext, y_ext = full.shape
    y_idx = np.arange(y_ext)

    def interface_y(col):
        sign = np.sign(col - 0.5)
        changes = np.where(np.diff(sign) != 0)[0]
        if len(changes) == 0:
            return None
        idx = int(changes[0])
        y0, y1 = y_idx[idx], y_idx[idx + 1]
        p0, p1 = col[idx], col[idx + 1]
        if p1 == p0:
            return float(y0)
        return float(y0 + (0.5 - p0) * (y1 - y0) / (p1 - p0))

    y_edge = interface_y(full[edge_x, :])
    baseline_x = min(edge_x + baseline_x_offset, x_ext - 1) if edge_x < x_ext // 2 else max(edge_x - baseline_x_offset, 0)
    y_baseline = interface_y(full[baseline_x, :])
    if y_edge is None or y_baseline is None:
        return None
    return y_edge - y_baseline


def main():
    files = glob.glob(os.path.join(SCRIPT_DIR, 'phi_matrix_*'))
    entries = []
    for path in files:
        m = re.search(r'phi_matrix_(\d+)$', os.path.basename(path))
        if not m:
            continue
        iteration = int(m.group(1))
        entries.append((iteration, path))
    entries.sort()

    if not entries:
        print("No phi_matrix_<iteration>.txt files found in this folder.")
        return

    print(f"{'iteration':>10}  {'left edge bump':>15}  {'right edge bump':>16}")
    for iteration, path in entries:
        full = load_field(path)
        x_ext, _ = full.shape
        left_bump = edge_bump_size(full, edge_x=0)
        right_bump = edge_bump_size(full, edge_x=x_ext - 1)
        print(f"{iteration:>10}  {left_bump!s:>15}  {right_bump!s:>16}")


if __name__ == '__main__':
    main()
