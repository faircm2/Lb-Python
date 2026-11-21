"""W=3 trace run (box x410-418, y50-57, z26-32, iterations 2150..2229).
Takes the fastest cell of the box in the last trace row and prints, per iteration (every 4th, all from 2200 on):
  raw_phi, phi, rho, |u|, |mom| (= |sum c f|/Cs2), p of that cell, and max |u| / min rho / max rho of its 6 neighbours.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_w3_trace.py
"""
import glob
import numpy as np

folder = glob.glob("FreesurfaceImages/*_W3_xs1.0_*fill9.4_periodicx_hbs*")[0]
T = np.load(glob.glob(folder + "/trace_x*.npz")[0])
it, rawphi, phi, rho, u, mom, p, o = T["it"], T["raw_phi"], T["phi"], T["rho"], T["u"], T["mom"], T["p"], T["origin"]
speed = np.sqrt((u ** 2).sum(1))                       # (n, X, Y, Z)
last = speed[-1][1:-1, 1:-1, 1:-1]
i, j, k = (int(a) + 1 for a in np.unravel_index(last.argmax(), last.shape))
print("rows", it[0], "..", it[-1], " box origin", o, " shape", rho.shape)
print(f"fastest interior cell of the last row: ({o[0] + i},{o[1] + j},{o[2] + k})")
nb = [(i + 1, j, k), (i - 1, j, k), (i, j + 1, k), (i, j - 1, k), (i, j, k + 1), (i, j, k - 1)]
print(f"{'it':>5s} | {'raw_phi':>8s} {'phi':>8s} {'rho':>8s} {'|u|':>8s} {'|mom|':>8s} {'p':>9s} | {'nb max|u|':>9s} {'nb rho min':>10s} {'nb rho max':>10s}")
for n in range(len(it)):
    if it[n] % 4 != 0 and it[n] < 2200:
        continue
    m = np.sqrt((mom[n, :, i, j, k] ** 2).sum())
    print(f"{it[n]:5d} | {rawphi[n, i, j, k]:8.5f} {phi[n, i, j, k]:8.5f} {rho[n, i, j, k]:8.5f} {speed[n, i, j, k]:8.5f} {m:8.5f} {p[n, i, j, k]:9.5f} | "
          f"{max(speed[n][c] for c in nb):9.5f} {min(rho[n][c] for c in nb):10.5f} {max(rho[n][c] for c in nb):10.5f}")
