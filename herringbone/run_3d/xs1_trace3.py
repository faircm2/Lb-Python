"""Momentum budget at the onset cell (157,25,13) of the fill9.4 periodicx MULT=1 trace (iterations 10570..10598).
Per component: change of the f-moment per iteration  dmom = mom(n+1) - mom(n)  versus  -dp (central difference of p, neighbours at distance 1)
and  rhoF = rho*u - mom  (= (Fs*dt + body)/2).  Also p at x-1, x, x+1.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_trace3.py
"""
import glob
import numpy as np

fn = glob.glob("FreesurfaceImages/*fill9.4_periodicx_hbs*/trace_x*.npz")[0]
d = np.load(fn)
o = d["origin"]
cell = (157, 25, 13)
i, j, k = (cell[0] - o[0], cell[1] - o[1], cell[2] - o[2])
its = d["it"]
p, mom, rho, u = d["p"], d["mom"], d["rho"], d["u"]
print(f"cell {cell}: dmom = mom(n+1)-mom(n);  -dp = -(p[+1]-p[-1])/2;  rhoF = rho*u - mom")
print(f"{'it':>6s} | {'p x-1':>8s} {'p x':>8s} {'p x+1':>8s} | {'dmom x':>8s} {'-dp x':>8s} {'rhoF x':>8s} | "
      f"{'dmom y':>8s} {'-dp y':>8s} {'rhoF y':>8s} | {'dmom z':>8s} {'-dp z':>8s} {'rhoF z':>8s}")
for n, it in enumerate(its):
    if it < 10570 or it > 10598:
        continue
    dm = mom[n + 1, :, i, j, k] - mom[n, :, i, j, k]
    dp = -np.array([(p[n, i + 1, j, k] - p[n, i - 1, j, k]) / 2.0,
                    (p[n, i, j + 1, k] - p[n, i, j - 1, k]) / 2.0,
                    (p[n, i, j, k + 1] - p[n, i, j, k - 1]) / 2.0])
    rf = rho[n, i, j, k] * u[n, :, i, j, k] - mom[n, :, i, j, k]
    print(f"{it:6d} | {p[n, i - 1, j, k]:8.5f} {p[n, i, j, k]:8.5f} {p[n, i + 1, j, k]:8.5f} | "
          f"{dm[0]:8.5f} {dp[0]:8.5f} {rf[0]:8.5f} | {dm[1]:8.5f} {dp[1]:8.5f} {rf[1]:8.5f} | {dm[2]:8.5f} {dp[2]:8.5f} {rf[2]:8.5f}")
