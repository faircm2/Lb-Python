"""
Quick look at the floor of a run, from its plane_yz dumps (run on the cluster, numpy and scipy are in the lbm env):

    python floor_check_system.py <run_folder> [max_iteration]

For every dump it prints
    min phi on the floor row z = 0 (liquid side, y = 3..190) and where,
    the number of floor cells with phi < 0.999,
    the strength of the 4-cell ripple along y in the strip y = 2..40 (RMS of the 4-cell-periodic part of 1 - phi on the rows z = 0 and z = 1),
    the number of gas bubbles (gas cells enclosed by liquid) and their total size.
Reference values, same parameters (sigma 0.05, tau_f = tau_g = 0.65, alpha 90):
    A_periodic   (theta 60):   4-cell ripple 6e-5 at it 100, 1.0e-4 at it 450;  min phi 0.9989 at it 450, 0.981 at it 9000;  bubbles from it 36,003
    T90_periodic (theta 90):   4-cell ripple 1e-6 at it 100, 1.1e-5 at it 500 (10x weaker);  min phi 0.9997 at it 450, 0.9988 at it 48,000;  no bubbles
"""
import glob
import os
import re
import sys

import numpy as np

folder = sys.argv[1]
max_it = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None


def bubbles(p):
    if ndi is None:
        return None
    lab, n = ndi.label(p < 0.5)
    top = set(np.unique(lab[:, -1])) - {0}
    return [int((lab == k).sum()) for k in range(1, n + 1) if k not in top]


def ripple(p):
    """RMS of the component with period 4 cells along y, in the strip y = 2..40, rows z = 0 and 1."""
    out = []
    y = np.arange(2, 42)
    for z in (0, 1):
        d = 1.0 - p[2:42, z]
        d = d - d.mean()
        c = np.cos(2 * np.pi * y / 4.0)
        s = np.sin(2 * np.pi * y / 4.0)
        out.append(2.0 / len(y) * np.hypot((d * c).sum(), (d * s).sum()))
    return float(np.mean(out))


files = sorted(glob.glob(os.path.join(folder, "plane_yz_iter_*.npz")))
print("%8s | %10s %4s | %6s | %10s | %s" % ("iter", "min phi", "y", "<0.999", "4-cell rip", "bubbles (count, cells)"))
for f in files:
    it = int(re.search(r"iter_(\d+)", f).group(1))
    if it > max_it:
        break
    p = np.load(f)["phi"]
    row = p[3:190, 0]
    j = int(row.argmin()) + 3
    b = bubbles(p)
    print("%8d | %10.5f %4d | %6d | %10.2e | %s" % (it, row.min(), j, int((p[3:190, 0] < 0.999).sum()), ripple(p),
                                                    "n/a (no scipy)" if b is None else "%d, %d" % (len(b), sum(b))))
