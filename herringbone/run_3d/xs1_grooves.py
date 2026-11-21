"""Do all grooves thin like groove 4 of train 1, or only that one?  Baseline run (xs1.0, fill9.4, rho_G=0.01), plane y=25.
Grooves = runs of open cells along x at y=25, z=10 (12 grooves: 6 per train).  For every groove and dump:
  fill  = mean phi over the groove cells (x in the run, z=4..14)
  above = liquid thickness above the floor in cells, sum of phi over z>=15, mean over the x of the run
The groove containing the onset cell (x=157) is marked with *.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_grooves.py
"""
import os, sys, glob
import numpy as np
os.environ["TP"] = "4.9"
sys.path.insert(0, "/work/classic/fr_jn194-lbm")
from herringbone.herringbone_idealised import Herringbone, herringbone_mask

XN, YN, ZN = 736, 64, 64
Y = 25
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
xs = np.where(~solid[:, Y, 10])[0]
cut = np.where(np.diff(xs) > 1)[0] + 1
runs = [(int(r[0]), int(r[-1])) for r in np.split(xs, cut)]
names = [f"G{n + 1}{'*' if a <= 157 <= b else ''}" for n, (a, b) in enumerate(runs)]
print("grooves at y=25:", ", ".join(f"{nm} x{a}-{b}" for nm, (a, b) in zip(names, runs)))

folder = glob.glob("FreesurfaceImages/*xs1.0_nu*fill9.4_periodicx_hbs*")[0]
dumps = [it for it in (8400, 9800, 10000, 10200, 10400, 10480, 10520, 10560) if os.path.exists(f"{folder}/fields3d_iter_{it:06d}.npz")]
fill, above = [], []
for it in dumps:
    phi = np.load(f"{folder}/fields3d_iter_{it:06d}.npz")["phi"].astype(float)
    fill.append([phi[a:b + 1, Y, 4:15].mean() for a, b in runs])
    above.append([phi[a:b + 1, Y, 15:].sum(axis=1).mean() for a, b in runs])
for title, tab in (("fill (mean phi in the groove, z=4..14)", fill), ("above (liquid thickness above the floor, cells)", above)):
    print(f"\n{title}")
    print(f"{'iter':>6s} | " + " ".join(f"{nm:>6s}" for nm in names))
    for it, row in zip(dumps, tab):
        print(f"{it:6d} | " + " ".join(f"{v:6.2f}" for v in row))
