"""W=3 trace run: does the advection by u remove phi from the first diverging gas cells?  Passes m = 2212..2223.
Per cell and pass m:
    phi_old / phi_new = phi of trace row m-1 / m          dphi = phi_new - phi_old
    adv  = -u . grad(phi)   (central differences, u and phi of row m-1, lattice units per iteration)
    rest = dphi - adv       (diffusion/Cahn-Hilliard terms, clipping, dt scaling)
    raw  = raw_phi of row m (before the clip), rho, |u| of row m-1
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_w3_phibudget.py
"""
import glob
import numpy as np

folder = glob.glob("FreesurfaceImages/*_W3_xs1.0_*fill9.4_periodicx_hbs*")[0]
T = np.load(glob.glob(folder + "/trace_x*.npz")[0])
it_t, raw, phi, rho, u, o = T["it"], T["raw_phi"], T["phi"], T["rho"], T["u"], T["origin"]
print(f"{'m':>5s} | {'rho':>7s} {'|u|':>7s} | {'phi_old':>9s} {'phi_new':>9s} {'raw_new':>9s} | {'dphi':>9s} {'adv':>9s} {'rest':>9s}")
for cell in ((415, 51, 27), (415, 53, 28), (415, 52, 28), (416, 53, 27)):
    b = tuple(cell[a] - o[a] for a in range(3))
    i, j, k = b
    print(f"\ncell {cell}")
    for m in range(2212, 2224):
        n1 = int(np.where(it_t == m)[0][0])
        n0 = n1 - 1
        p0 = phi[n0]
        grad = np.array([(p0[i + 1, j, k] - p0[i - 1, j, k]) / 2.0, (p0[i, j + 1, k] - p0[i, j - 1, k]) / 2.0, (p0[i, j, k + 1] - p0[i, j, k - 1]) / 2.0])
        uu = u[n0][(slice(None),) + b]
        adv = -float(uu @ grad)
        dphi = float(phi[n1][b] - p0[b])
        print(f"{m:5d} | {rho[n0][b]:7.4f} {np.sqrt((uu ** 2).sum()):7.4f} | {p0[b]:9.5f} {phi[n1][b]:9.5f} {raw[n1][b]:9.5f} | {dphi:9.5f} {adv:9.5f} {dphi - adv:9.5f}")
