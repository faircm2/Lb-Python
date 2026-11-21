"""Is the ~0.1 'gas' speed at the interface just the film's own surface velocity?  sigma0.01 W=4 run, dump 2150.
Column through the interface at the fastest gas cell (x~371,y~43) and at a second column (x~284,y~46): z, phi, rho, u (3 components), |u|.
Then the liquid bulk (phi>0.9, z>=16): mean |u| and max |u|, and the interface cells (0.3<phi<0.7): mean |u|.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_surface_vel.py
"""
import glob
import numpy as np

RHO_L, RHO_G = 10.0, 0.01
folder = glob.glob("FreesurfaceImages/*sigma0.01_xs1.0_*fill9.4_periodicx_hbs*")[0]
d = np.load(f"{folder}/fields3d_iter_002150.npz")
phi = d["phi"].astype(float)
u = d["u"].astype(float)
sp = np.sqrt((u ** 2).sum(0))
for (x, y) in ((371, 43), (284, 46)):
    print(f"\ncolumn x={x} y={y}")
    print(f"{'z':>3s} | {'phi':>7s} | {'rho':>7s} | {'ux':>8s} {'uy':>8s} {'uz':>8s} | {'|u|':>7s}")
    for z in range(16, 35):
        p = phi[x, y, z]
        print(f"{z:3d} | {p:7.4f} | {p * RHO_L + (1 - p) * RHO_G:7.4f} | {u[0, x, y, z]:8.4f} {u[1, x, y, z]:8.4f} {u[2, x, y, z]:8.4f} | {sp[x, y, z]:7.4f}")
liq = phi[:, :, 16:] > 0.9
itf = (phi[:, :, 16:] > 0.3) & (phi[:, :, 16:] < 0.7)
s = sp[:, :, 16:]
print(f"\nliquid bulk (phi>0.9, z>=16): mean|u| {s[liq].mean():.4f}  max|u| {s[liq].max():.4f}")
print(f"interface cells (0.3<phi<0.7): mean|u| {s[itf].mean():.4f}  max|u| {s[itf].max():.4f}")
print(f"mean u components in the liquid bulk: {u[0][:, :, 16:][liq].mean():.4f} {u[1][:, :, 16:][liq].mean():.4f} {u[2][:, :, 16:][liq].mean():.4f}")
