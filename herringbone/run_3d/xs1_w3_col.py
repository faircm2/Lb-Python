"""W=3 run: what surrounds the fastest gas cell (phi<0.05) in the dense dumps 2150..2220?
For every dump: the fastest gas cell and the z-column through it (z-4..z+4): phi and |u|.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_w3_col.py
"""
import glob
import numpy as np

folder = glob.glob("FreesurfaceImages/*W3_xs1.0*fill9.4_periodicx_hbs*")[0]
for it in range(2150, 2221, 10):
    d = np.load(f"{folder}/fields3d_iter_{it:06d}.npz")
    phi = d["phi"].astype(float)
    u = d["u"].astype(float)
    speed = np.sqrt((u ** 2).sum(0))
    gs = np.where(phi < 0.05, speed, 0.0)
    x, y, z = (int(a) for a in np.unravel_index(gs.argmax(), gs.shape))
    zs = range(z - 4, z + 5)
    print(f"\n{it}: fastest gas cell ({x},{y},{z}) |u|={speed[x, y, z]:.4f}")
    print("  z   : " + " ".join(f"{k:7d}" for k in zs))
    print("  phi : " + " ".join(f"{phi[x, y, k]:7.3f}" for k in zs))
    print("  |u| : " + " ".join(f"{speed[x, y, k]:7.4f}" for k in zs))
