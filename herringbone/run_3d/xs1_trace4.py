"""Split the pressure at the onset cell (157,25,13) and its x-neighbours of the fill9.4 periodicx MULT=1 trace into its two parts (zp):
    p = sum_i f_i  +  (dt/2) * Cs2 * u . grad(rho)
Gradient of rho by central differences from the trace box (not the solver's isotropic stencil, so approximate). Cs2 = 1/3, dt = 1.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_trace4.py
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
print("per cell: rho | u_x | u.grad(rho) term of p | sum f part of p = p - term        (cells x=156, 157, 158 at y=25, z=13)")
hdr = f"{'it':>6s} |"
for x in (156, 157, 158):
    hdr += f" x={x}: {'rho':>7s} {'u_x':>8s} {'term':>8s} {'sum f':>8s} |"
print(hdr)
for n, it in enumerate(its):
    if it < 10570 or it > 10598 or (it < 10585 and it % 5 != 0):
        continue
    row = f"{it:6d} |"
    for di in (-1, 0, 1):
        i = i0 + di
        gr = np.array([(rho[n, i + 1, j, k] - rho[n, i - 1, j, k]) / 2.0,
                       (rho[n, i, j + 1, k] - rho[n, i, j - 1, k]) / 2.0,
                       (rho[n, i, j, k + 1] - rho[n, i, j, k - 1]) / 2.0])
        term = 0.5 * Cs2 * float(u[n, :, i, j, k] @ gr)
        row += f"        {rho[n, i, j, k]:7.4f} {u[n, 0, i, j, k]:8.4f} {term:8.5f} {p[n, i, j, k] - term:8.5f} |"
    print(row)
