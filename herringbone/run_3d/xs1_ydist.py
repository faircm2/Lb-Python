"""Where does the liquid leave G3-G5?  Baseline run (xs1.0, fill9.4, rho_G=0.01), x = 118..192 (grooves G3, G4, G5 of train 1).
Per y (every 2nd cell), mean over that x range at 8400 / 9800 / 10560:
  above = liquid thickness above the floor in cells (sum of phi over z>=15)
  total = liquid in the whole column incl. the grooves (sum of phi over the open cells)
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_ydist.py
"""
import os, sys, glob
import numpy as np
os.environ["TP"] = "4.9"
sys.path.insert(0, "/work/classic/fr_jn194-lbm")
from herringbone.herringbone_idealised import Herringbone, herringbone_mask

XN, YN, ZN = 736, 64, 64
X0, X1 = 118, 192
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
folder = glob.glob("FreesurfaceImages/*xs1.0_nu*fill9.4_periodicx_hbs*")[0]
dumps = (8400, 9800, 10560)
above, total = [], []
for it in dumps:
    phi = np.load(f"{folder}/fields3d_iter_{it:06d}.npz")["phi"].astype(float)[X0:X1 + 1]
    above.append(phi[:, :, 15:].sum(axis=2).mean(axis=0))                 # (Y,)
    total.append((phi * ~solid[X0:X1 + 1]).sum(axis=2).mean(axis=0))
print(f"x = {X0}..{X1};  film thickness against y (cells), mean over x")
print(f"{'y':>3s} | " + " ".join(f"{'above ' + str(it):>11s}" for it in dumps) + " | " + " ".join(f"{'total ' + str(it):>11s}" for it in dumps))
for y in range(0, YN, 2):
    print(f"{y:3d} | " + " ".join(f"{a[y]:11.2f}" for a in above) + " | " + " ".join(f"{t[y]:11.2f}" for t in total))
