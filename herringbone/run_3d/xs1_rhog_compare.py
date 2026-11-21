"""Same groove segment (train-1 groove, onset cell A=(157,25,13), B=(158,26,15)) in the rho_G=0.1 run and in the rho_G=0.01 MULT=1 run.
Per fields3d dump (iteration >= 8000): phi, rho and speed at A and B, and the fastest gas cell (phi<0.05) in the window
x150..162, y24..26, z5..16 with its rho.  rho = phi*rho_L + (1-phi)*rho_G, rho_L = 10, rho_G from the folder name (rhoG0.1 -> 0.1, else 0.01).
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_rhog_compare.py
"""
import glob
import numpy as np

RHO_L = 10.0
A, B = (157, 25, 13), (158, 26, 15)
W = (slice(150, 163), slice(24, 27), slice(5, 17))

folders = [("rho_G=0.1 (x2 folder)", glob.glob("FreesurfaceImages/*tau_f_0.6__*rhoG0.1*fill9.4_periodicx_x2_hbs*")),
           ("rho_G=0.01 MULT=1", glob.glob("FreesurfaceImages/*fill9.4_periodicx_hbs*"))]
for label, fl in folders:
    print(f"\n=== {label}: {fl}")
    if not fl:
        continue
    folder = fl[0]
    rho_g = 0.1 if "rhoG0.1" in folder else 0.01
    print(f"{'iter':>6s} | {'phi A':>7s} {'rho A':>7s} {'|u| A':>8s} | {'phi B':>7s} {'rho B':>7s} {'|u| B':>8s} | "
          f"{'gas max|u| in W':>15s} {'(x,y,z)':<14s} {'rho':>7s}")
    for fn in sorted(glob.glob(folder + "/fields3d_iter_*.npz")):
        it = int(fn.split("_")[-1].split(".")[0])
        if it < 8000:
            continue
        d = np.load(fn)
        phi = d["phi"].astype(float)
        u = d["u"].astype(float)
        if not (np.isfinite(phi).all() and np.isfinite(u).all()):
            print(f"{it:6d} | NaN in dump")
            continue
        sp = np.sqrt((u ** 2).sum(0))
        rho = phi * RHO_L + (1.0 - phi) * rho_g
        gs = np.where(phi[W] < 0.05, sp[W], 0.0)
        g = np.unravel_index(gs.argmax(), gs.shape)
        gx, gy, gz = g[0] + 150, g[1] + 24, g[2] + 5
        print(f"{it:6d} | {phi[A]:7.4f} {rho[A]:7.4f} {sp[A]:8.4f} | {phi[B]:7.4f} {rho[B]:7.4f} {sp[B]:8.4f} | "
              f"{gs.max():15.4f} {str((gx, gy, gz)):<14s} {rho[gx, gy, gz]:7.4f}")
