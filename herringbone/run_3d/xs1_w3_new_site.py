"""W=3 run with the per-link q wall BC: the first runaway is now at x~150-157, y~7-14, z~26-30 (old W=3 run: x~412-417, y~50-57).
From the dense fields3d dumps (every 10 iterations, 2150..2230): max gas |u| (phi<0.05) in three regions
A = new site (x 148..160, y 5..16), B = old site (x 405..420, y 48..58), C = everywhere else, with the cell, phi, rho, and the local interface height
(highest z with phi>=0.5 in that column).  Also the gas |u|>0.15 cell count per dump.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_w3_new_site.py
"""
import glob
import numpy as np

RHO_L, RHO_G = 10.0, 0.01
folder = glob.glob("FreesurfaceImages/*_W3_xs1.0_*fill9.4_periodicx_hbs*")[0]
print(folder)
print(f"{'iter':>5s} | {'reg':>3s} | {'gas max|u|':>10s} | {'cell (x,y,z)':<14s} | {'phi':>7s} | {'rho':>7s} | {'z_iface':>7s} | {'n(gas|u|>.15)':>13s}")
for fn in sorted(glob.glob(folder + "/fields3d_iter_*.npz")):
    it = int(fn.split("_")[-1].split(".")[0])
    if it < 2150 or it > 2230:
        continue
    d = np.load(fn)
    phi = d["phi"].astype(float)
    u = d["u"].astype(float)
    speed = np.sqrt((u ** 2).sum(0))
    gas = phi < 0.05
    gs = np.where(gas, speed, 0.0)
    X, Y = phi.shape[0], phi.shape[1]
    xi = np.arange(X)[:, None, None]
    yi = np.arange(Y)[None, :, None]
    regA = (xi >= 148) & (xi <= 160) & (yi >= 5) & (yi <= 16)
    regB = (xi >= 405) & (xi <= 420) & (yi >= 48) & (yi <= 58)
    regC = ~(regA | regB)
    ncell = int((gs > 0.15).sum())
    for name, reg in (("A", regA), ("B", regB), ("C", regC)):
        g = np.where(reg, gs, 0.0)
        j = np.unravel_index(g.argmax(), g.shape)
        col = phi[j[0], j[1], :] >= 0.5
        zi = int(np.nonzero(col)[0].max()) if col.any() else -1
        rho = phi[j] * RHO_L + (1 - phi[j]) * RHO_G
        print(f"{it:5d} | {name:>3s} | {g.max():10.4f} | {str(tuple(int(a) for a in j)):<14s} | {phi[j]:7.4f} | {rho:7.4f} | {zi:7d} | {ncell if name == 'A' else '':>13}")
