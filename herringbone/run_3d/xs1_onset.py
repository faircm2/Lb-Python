"""Onset of the NaN in the dense fields3d dumps (iterations 9800..11000, every 40) of the fill9.4 periodicx MULT=1 run.
For every dump: NaN cells and their x-range, or, if the dump is clean, the fastest cell, the fastest GAS cell (phi<0.05) and the phi range.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_onset.py
"""
import glob
import numpy as np

import sys
folder = glob.glob(f"FreesurfaceImages/*{sys.argv[1]}*fill9.4_periodicx_hbs*")[0]     # e.g. xs1.0_nu, xs1.01875, xs1.848528
print(folder)
print(f"{'iter':>5s} | {'NaN cells':>9s} | {'NaN x-range':>11s} | {'max|u|':>7s} | {'fastest (x,y,z)':<16s} | {'phi':>5s} | "
      f"{'gas max|u|':>10s} | {'gas cell (x,y,z)':<16s} | {'phi':>5s} | {'phi min':>7s} | {'phi max':>7s}")
for fn in sorted(glob.glob(folder + "/fields3d_iter_*.npz")):
    it = int(fn.split("_")[-1].split(".")[0])
    if it < (int(sys.argv[2]) if len(sys.argv) > 2 else 9800):
        continue
    d = np.load(fn)
    phi = d["phi"].astype(float)
    u = d["u"].astype(float)
    bad = ~np.isfinite(phi) | ~np.isfinite(u).all(axis=0)
    if bad.any():
        xs = np.where(bad.any(axis=(1, 2)))[0]
        print(f"{it:5d} | {int(bad.sum()):9d} | {xs.min():4d}..{xs.max():<4d}  |")
        continue
    speed = np.sqrt((u ** 2).sum(0))
    i = np.unravel_index(speed.argmax(), speed.shape)
    gs = np.where(phi < 0.05, speed, 0.0)
    j = np.unravel_index(gs.argmax(), gs.shape)
    print(f"{it:5d} | {0:9d} | {'-':>11s} | {speed.max():7.4f} | {str(tuple(int(a) for a in i)):<16s} | {phi[i]:5.2f} | "
          f"{gs.max():10.4f} | {str(tuple(int(a) for a in j)):<16s} | {phi[j]:5.3f} | {phi.min():7.3f} | {phi.max():7.3f}")
