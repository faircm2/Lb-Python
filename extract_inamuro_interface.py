"""
Extracts the phi=0.5 interface profile from the Inamuro script's
phi_results.txt (written every iteration by save_phi_results(), so after a
run finishes it holds the final iteration's normalized phi field).

Usage: python extract_inamuro_interface.py
Reads phi_results.txt (same folder), writes interface_profile_inamuro.csv,
and prints the last ~10 columns near each edge.
"""
import csv
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHI_RESULTS_FILE = os.path.join(SCRIPT_DIR, 'phi_results.txt')
OUT_CSV = os.path.join(SCRIPT_DIR, 'interface_profile_inamuro.csv')


def interface_y(col, y_idx):
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


def main():
    with open(PHI_RESULTS_FILE, 'r') as f:
        f.readline()  # header: "phi_star_G phi_star_L" (reference only, field is already normalized 0-1)
        data = np.loadtxt(f)

    full = data[::-1, :].T  # full[x, y], y=0 bottom .. y=Yn+1 top
    x_ext, y_ext = full.shape
    y_idx = np.arange(y_ext)

    xs, ys = [], []
    for x in range(x_ext):
        y = interface_y(full[x, :], y_idx)
        xs.append(x)
        ys.append(y)

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'interface_y'])
        for x, y in zip(xs, ys):
            writer.writerow([x, '' if y is None else y])

    print(f"Saved: {OUT_CSV}")
    print(f"\nFirst 10 columns (left edge):")
    for x, y in list(zip(xs, ys))[:10]:
        print(f"  x={x:3d}  interface_y={y}")
    print(f"\nLast 10 columns (right edge):")
    for x, y in list(zip(xs, ys))[-10:]:
        print(f"  x={x:3d}  interface_y={y}")


if __name__ == '__main__':
    main()
