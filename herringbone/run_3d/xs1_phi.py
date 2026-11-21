"""Liquid fraction phi of the xs1.0 runs: y-z cuts at x = 150 and 420 and the x-z side view through the channel centre (y = 32).
Colorbar RdBu over the data range (standard phi plot: red = low phi = gas, blue = high = liquid), grey = solid. Data: fields3d/fields3d_iter_XXXXXX.npz (W=4 run, job 8815366).
Usage: python xs1_phi.py [iteration]    ->  xs1_phi_iterXXXXXX.png"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TP"] = "4.9"                      # train period 4.9 mm = 1.0 mm gap; must be set before the import
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 10200
XN, YN, ZN = 736, 64, 64
DX = 1.2 / YN                                                   # cell size (mm)
Z_FLOOR = 15
FILLS = {"W4": "W=4, sigma 0.1 (job 8815366)"}
X_CUTS = (150, 420)                                             # cell number x of the y-z cuts
Y_CENTRE = 32                                                   # cell number y of the side view

x = (np.arange(1, XN + 1) - 0.5) * DX
y = (np.arange(1, YN + 1) - 0.5) * DX
z = (np.arange(1, ZN + 1) - 0.5 - Z_FLOOR) * DX                 # mm above the floor
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]    # index = cell number - 1


def draw(ax, h, v, phi_cut, solid_cut, xlabel):
    """phi_cut and solid_cut are (z, h) arrays."""
    mesh = ax.pcolormesh(h, v, np.ma.masked_where(solid_cut, phi_cut), cmap="RdBu", vmin=VMIN, vmax=VMAX, shading="nearest")
    ax.pcolormesh(h, v, np.ma.masked_where(~solid_cut, solid_cut), cmap="Greys", vmin=0, vmax=2, shading="nearest")
    ax.contour(h, v, phi_cut, levels=[0.5], colors="k", linewidths=0.8)
    ax.set(ylim=(-0.22, 0.56), xlabel=xlabel, ylabel="z above floor (mm)")
    return mesh


fill, fill_name = next(iter(FILLS.items()))
phi = np.load(os.path.join(HERE, "fields3d", f"fields3d_iter_{ITER:06d}.npz"))["phi"].astype(float)
VMIN, VMAX = phi[~solid].min(), phi[~solid].max()              # colorbar range = data range, as in the standard phi plot
fig = plt.figure(figsize=(15, 11), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])           # the two y-z cuts on top, the side view underneath
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
for ax, xc in zip(axes[:2], X_CUTS):
    mesh = draw(ax, y, z, phi[xc - 1].T, solid[xc - 1].T, "y (mm)")
    ax.set(xlim=(0, 1.2), aspect="equal", title=f"y-z at x = {xc}")
draw(axes[2], x, z, phi[:, Y_CENTRE - 1].T, solid[:, Y_CENTRE - 1].T, "x (mm)")
axes[2].set(xlim=(0, x[-1]), title=f"x-z side view at y = {Y_CENTRE} (channel centre)")
fig.colorbar(mesh, ax=axes, pad=0.01, fraction=0.02, label="phi")   # standard phi colorbar: RdBu, data range
fig.suptitle(f"{fill_name}: liquid fraction phi at iteration {ITER} (black line: phi = 0.5)")
out = os.path.join(HERE, f"xs1_phi_iter{ITER:06d}.png")
fig.savefig(out, dpi=110)
print("wrote", out)
