"""Baseline run (xs1.0, W=4, fill9.4, trace 10500..10620, box x155-160, y24-27, z12-16): does the advection by u remove phi from the draining
groove-4 cell?  Same budget as xs1_w3_phibudget.py, cells (157,25,13) and (158,25,13), passes m = 10575..10600.
    dphi = phi(row m) - phi(row m-1);  adv = -u . grad(phi) (central differences, u and phi of row m-1);  rest = dphi - adv
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_phibudget_base.py
"""
import glob
import numpy as np

fn = glob.glob("FreesurfaceImages/*_sigma0.1_xs1.0_nu*fill9.4_periodicx_hbs*/trace_x*.npz")[0]
print(fn)
T = np.load(fn)
it_t, raw, phi, rho, u, mom, o = T["it"], T["raw_phi"], T["phi"], T["rho"], T["u"], T["mom"], T["origin"]
print("rows", it_t[0], "..", it_t[-1], " origin", o, " shape", rho.shape)
print(f"{'m':>6s} | {'rho':>7s} {'|u|':>8s} {'|mom|':>8s} | {'phi_old':>9s} {'phi_new':>9s} {'raw_new':>9s} | {'dphi':>9s} {'adv':>9s} {'rest':>9s}")
for cell in ((157, 25, 13), (158, 25, 13)):
    b = tuple(cell[a] - o[a] for a in range(3))
    i, j, k = b
    print(f"\ncell {cell}")
    for m in range(10575, 10601):
        w = np.where(it_t == m)[0]
        if len(w) == 0:
            continue
        n1 = int(w[0])
        n0 = n1 - 1
        p0 = phi[n0]
        grad = np.array([(p0[i + 1, j, k] - p0[i - 1, j, k]) / 2.0, (p0[i, j + 1, k] - p0[i, j - 1, k]) / 2.0, (p0[i, j, k + 1] - p0[i, j, k - 1]) / 2.0])
        uu = u[n0][(slice(None),) + b]
        adv = -float(uu @ grad)
        dphi = float(phi[n1][b] - p0[b])
        mg = float(np.sqrt((mom[n0][(slice(None),) + b] ** 2).sum()))
        print(f"{m:6d} | {rho[n0][b]:7.4f} {np.sqrt((uu ** 2).sum()):8.4f} {mg:8.5f} | {p0[b]:9.5f} {phi[n1][b]:9.5f} {raw[n1][b]:9.5f} | {dphi:9.5f} {adv:9.5f} {dphi - adv:9.5f}")
