"""W=3 (per-link q) run: column through the gas-tail cell that has the largest gas |u| (phi<0.05, z>=20) at each dump 2150, 2180, 2200, 2220.
For z = (cell z - 8) .. (cell z + 4): phi, rho, |u|, J = rho*|u| (momentum density, what streaming carries).
Question: does J in the tail fall smoothly with height (rho*u decaying), or is there a bump/floor at the failing cell?
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_w3_jcol.py
"""
import glob
import numpy as np

RHO_L, RHO_G = 10.0, 0.01
folder = glob.glob("FreesurfaceImages/*_W3_xs1.0_*fill9.4_periodicx_hbs*")[0]
for it in (2150, 2180, 2200, 2220):
    d = np.load(f"{folder}/fields3d_iter_{it:06d}.npz")
    phi = d["phi"].astype(float)
    sp = np.sqrt((d["u"].astype(float) ** 2).sum(0))
    g = np.where((phi < 0.05) & (np.arange(phi.shape[2])[None, None, :] >= 20), sp, 0.0)
    x, y, z0 = np.unravel_index(g.argmax(), g.shape)
    print(f"\niter {it}: column x={x} y={y} (gas max |u| {g.max():.4f} at z={z0})")
    print(f"{'z':>3s} | {'phi':>8s} | {'rho':>8s} | {'|u|':>7s} | {'J=rho|u|':>9s}")
    for z in range(z0 - 8, z0 + 5):
        p = phi[x, y, z]
        rho = p * RHO_L + (1 - p) * RHO_G
        print(f"{z:3d} | {p:8.5f} | {rho:8.4f} | {sp[x, y, z]:7.4f} | {rho * sp[x, y, z]:9.5f}")
