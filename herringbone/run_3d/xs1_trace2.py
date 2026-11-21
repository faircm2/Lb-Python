"""Split u at the onset cell (157,25,13) of the fill9.4 periodicx MULT=1 trace into its two parts (zu_ckl):
    u = mom/rho + (u - mom/rho),   mom = sum_i c_i f_i / Cs2,   u - mom/rho = (Fs*dt + body_force)/(2 rho)
Prints per component: mom/rho and force part, for iterations 10560..10599.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_trace2.py
"""
import glob
import numpy as np

fn = glob.glob("FreesurfaceImages/*fill9.4_periodicx_hbs*/trace_x*.npz")[0]
d = np.load(fn)
o = d["origin"]
cell = (157, 25, 13)
i, j, k = (cell[0] - o[0], cell[1] - o[1], cell[2] - o[2])
print(f"cell {cell}: u = mom/rho + force part (x | y | z); rho*force part = (Fs*dt+body)/2")
print(f"{'it':>6s} | {'phi':>8s} | {'rho':>7s} | {'mom/rho x':>9s} {'force x':>9s} | {'mom/rho y':>9s} {'force y':>9s} | "
      f"{'mom/rho z':>9s} {'force z':>9s} | {'rho*F x':>9s} {'rho*F y':>9s} {'rho*F z':>9s}")
for n, it in enumerate(d["it"]):
    if it < 10560 or it > 10599:
        continue
    rho = d["rho"][n, i, j, k]
    u = d["u"][n, :, i, j, k]
    m = d["mom"][n, :, i, j, k] / rho
    f = u - m
    print(f"{it:6d} | {d['phi'][n, i, j, k]:8.5f} | {rho:7.4f} | {m[0]:9.4f} {f[0]:9.4f} | {m[1]:9.4f} {f[1]:9.4f} | "
          f"{m[2]:9.4f} {f[2]:9.4f} | {rho*f[0]:9.5f} {rho*f[1]:9.5f} {rho*f[2]:9.5f}")
