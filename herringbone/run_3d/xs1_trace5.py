"""Pressure budget at the onset cell (157,25,13) and its x-neighbours in the fill9.4 periodicx MULT=1 trace (iterations 10580..10597).
Continuum prediction of the pressure-form scheme:  dp/dt = -rho * Cs2 * div(u)      (collision adds Cs2*u.grad(rho), streaming carries the
matching flux of Cs2*rho*u, so the u.grad(rho) parts cancel).
Per cell: observed dp = p(n+1)-p(n), predicted -Cs2*rho*div(u) (central differences inside the trace box), residual = observed - predicted.
Also: div(u) and rho.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_trace5.py
"""
import glob
import numpy as np

fn = glob.glob("FreesurfaceImages/*fill9.4_periodicx_hbs*/trace_x*.npz")[0]
d = np.load(fn)
o = d["origin"]
its = d["it"]
p, rho, u = d["p"], d["rho"], d["u"]
Cs2 = 1.0 / 3.0
i0, j, k = 157 - o[0], 25 - o[1], 13 - o[2]
hdr = f"{'it':>6s} |"
for x in (156, 157, 158):
    hdr += f" x={x}: {'rho':>7s} {'div u':>8s} {'dp obs':>9s} {'dp pred':>9s} {'resid':>9s} |"
print(hdr)
for n, it in enumerate(its):
    if it < 10580 or it > 10597:
        continue
    row = f"{it:6d} |"
    for di in (-1, 0, 1):
        i = i0 + di
        div = ((u[n, 0, i + 1, j, k] - u[n, 0, i - 1, j, k]) + (u[n, 1, i, j + 1, k] - u[n, 1, i, j - 1, k])
               + (u[n, 2, i, j, k + 1] - u[n, 2, i, j, k - 1])) / 2.0
        pred = -Cs2 * rho[n, i, j, k] * div
        obs = p[n + 1, i, j, k] - p[n, i, j, k]
        row += f"        {rho[n, i, j, k]:7.4f} {div:8.4f} {obs:9.5f} {pred:9.5f} {obs - pred:9.5f} |"
    print(row)
