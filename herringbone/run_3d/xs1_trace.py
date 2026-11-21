"""Per-iteration trace (iterations 10500..10619) of the box x155-160, y24-27, z12-16 in the fill9.4 periodicx MULT=1 run.
Prints, for the two onset cells, raw sum(g) before the clip, phi after clip + mass rescale, rho, u, f-moment and p per iteration.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_trace.py
"""
import glob
import numpy as np

fn = glob.glob("FreesurfaceImages/*fill9.4_periodicx_hbs*/trace_x*.npz")[0]
print(fn)
d = np.load(fn)
o = d["origin"]
pG, pL = d["phi_star"]
print("origin", o, " phi_star G/L", pG, pL, " rows", len(d["it"]), "iter", d["it"][0], "..", d["it"][-1])
for cell in [(157, 25, 13), (158, 26, 15)]:
    i, j, k = (cell[0] - o[0], cell[1] - o[1], cell[2] - o[2])
    print(f"\ncell {cell}   (row label = dump label)")
    print(f"{'it':>6s} | {'raw sum g':>10s} | {'clip(raw)':>9s} | {'phi':>9s} | {'rho':>8s} | {'u_x':>8s} | {'u_y':>8s} | {'u_z':>8s} | "
          f"{'mom_z':>8s} | {'p':>9s}")
    for n, it in enumerate(d["it"]):
        if it < 10560 and it % 10 != 0:
            continue
        raw = d["raw_phi"][n, i, j, k]
        u = d["u"][n, :, i, j, k]
        print(f"{it:6d} | {raw:10.4f} | {min(max(raw, pG), pL):9.4f} | {d['phi'][n, i, j, k]:9.5f} | {d['rho'][n, i, j, k]:8.4f} | "
              f"{u[0]:8.4f} | {u[1]:8.4f} | {u[2]:8.4f} | {d['mom'][n, 2, i, j, k]:8.4f} | {d['p'][n, i, j, k]:9.5f}")
