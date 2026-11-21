"""Pressure budget without the product rule, at x=157 (onset cell) and x=158 (y=25, z=13), fill9.4 periodicx MULT=1 trace, iterations 10580..10596.
    predicted dp = -Cs2*div(rho*u + Fh) + Cs2*u.grad(rho) + d(term),  Fh = rho*u - mom,  term = 0.5*Cs2*u.grad(rho)
All derivatives are central differences inside the trace box.
Columns: observed dp, -Cs2*div(rho u + Fh), +Cs2 u.grad(rho), d(term), residual = obs - sum of the three.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_trace7.py
"""
import glob
import numpy as np

fn = glob.glob("FreesurfaceImages/*fill9.4_periodicx_hbs*/trace_x*.npz")[0]
d = np.load(fn)
o = d["origin"]
its = d["it"]
p, rho, u, mom = d["p"], d["rho"], d["u"], d["mom"]
mstar = rho[:, None] * u + (rho[:, None] * u - mom)      # post-collision first moment / Cs2 = rho*u + Fh
Cs2 = 1.0 / 3.0
j, k = 25 - o[1], 13 - o[2]


def div(vec, n, i):
    return ((vec[n, 0, i + 1, j, k] - vec[n, 0, i - 1, j, k]) + (vec[n, 1, i, j + 1, k] - vec[n, 1, i, j - 1, k])
            + (vec[n, 2, i, j, k + 1] - vec[n, 2, i, j, k - 1])) / 2.0


def u_grad_rho(n, i):
    gr = np.array([(rho[n, i + 1, j, k] - rho[n, i - 1, j, k]) / 2.0, (rho[n, i, j + 1, k] - rho[n, i, j - 1, k]) / 2.0,
                   (rho[n, i, j, k + 1] - rho[n, i, j, k - 1]) / 2.0])
    return float(u[n, :, i, j, k] @ gr)


hdr = f"{'it':>6s} |"
for x in (157, 158):
    hdr += f" x={x}: {'dp obs':>9s} {'-Cs2 div m*':>11s} {'+Cs2 u.grho':>11s} {'d term':>9s} {'resid':>9s} |"
print(hdr)
for n, it in enumerate(its):
    if it < 10580 or it > 10596:
        continue
    row = f"{it:6d} |"
    for x in (157, 158):
        i = x - o[0]
        obs = p[n + 1, i, j, k] - p[n, i, j, k]
        a = -Cs2 * div(mstar, n, i)
        b = Cs2 * u_grad_rho(n, i)
        c = 0.5 * Cs2 * (u_grad_rho(n + 1, i) - u_grad_rho(n, i))
        row += f"        {obs:9.5f} {a:11.5f} {b:11.5f} {c:9.5f} {obs - a - b - c:9.5f} |"
    print(row)
