"""Q, Re, h(x) and fastest cell for every dump (>= 1285) of the xs1.0 runs. Run on NEMO2 from xs1_status.sh."""
import os, sys, glob
import numpy as np
os.environ["TP"] = "4.9"                       # must be set before the import (1.0 mm gap between the trains)
sys.path.insert(0, "/work/classic/fr_jn194-lbm")
from herringbone.herringbone_idealised import Herringbone, herringbone_mask

RUNS = (("14", 2.9e-6, 1.0), ("9", 1e-6, 0.65))            # (fill, nu, tau_f); the target is 1 ml/min for both
XN, YN, ZN = 736, 64, 64
D, DX = 1.2e-3, 1.2e-3 / 64
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
HEADER = (f"{'fill':>4s} | {'iter':>5s} | {'Q mean':>6s} | {'Q min':>6s} | {'Q max':>6s} | {'Re':>5s} | "
          f"{'h min':>5s} | {'h max':>5s} | {'Ma max':>6s} | {'fastest cell (x,y,z)':<20s} | {'phi there':>9s}")
print("Q = ml/min over the x-planes (target 1.0), h = liquid column height in cells, Ma max = sqrt(3)*max|u_lattice|")
print(HEADER)
for fill, nu, tau in RUNS:
    q = DX ** 3 / ((1 / 3) * (tau - 0.5) * DX ** 2 / nu) * 6e7           # lattice flux -> ml/min
    for d in glob.glob(f"FreesurfaceImages/*xs1.0*_fill{fill}_*"):
        for fn in sorted(glob.glob(d + "/fields3d_iter_*.npz")):
            it = int(fn.split("_")[-1].split(".")[0])
            if it < 1285: continue
            dd = np.load(fn); u = dd["u"].astype(float); phi = dd["phi"].astype(float)
            ux = np.where(solid, 0.0, u[0])
            flux = (phi * ux * ~solid).sum(axis=(1, 2)) * q
            h = phi.sum(axis=2).mean(axis=1)
            speed = np.sqrt((u ** 2).sum(0)); speed = np.where(np.isfinite(speed), speed, 0)
            i = np.unravel_index(speed.argmax(), speed.shape)
            cell = str(tuple(int(a) for a in i))
            print(f"{fill:>4s} | {it:5d} | {flux.mean():6.3f} | {flux.min():6.3f} | {flux.max():6.3f} | {flux.mean() / 6e7 / (D * nu):5.2f} | "
                  f"{h.min():5.1f} | {h.max():5.1f} | {np.sqrt(3) * speed.max():6.3f} | {cell:<20s} | {phi[i]:9.2f}")
