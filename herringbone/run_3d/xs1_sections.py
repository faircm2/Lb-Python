"""Cross-section streamlines of the xs1.0 runs (offset, train 1, 1.0 mm gap, train 2, smooth run-out).

One panel per fill (rows) and section (columns):
  lines  = streamlines of the secondary flow (u_y, u_z), coloured by |u_yz| on one scale for all panels
  blue   = liquid fraction phi (colorbar RdBu, as in xs1_phi.py), black = free surface (phi = 0.5), grey = solid
Data: fields3d/fields3d_iter_XXXXXX.npz (W=4 run, job 8815366).
Usage: python xs1_sections.py [iteration]    ->  xs1_sections_iterXXXXXX.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

os.environ["TP"] = "4.9"                      # train period 4.9 mm = 1.0 mm gap; must be set before the import
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 10200
XN, YN, ZN = 736, 64, 64
DX = 1.2 / YN                                                   # cell size (mm)
TAU_F, NU = 0.6, 1.0e-6                                         # job 8815366: --tau_f 0.6 --nu 1e-6 (water)
TO_MM_S = (DX * 1e-3) / ((1 / 3) * (TAU_F - 0.5) * (DX * 1e-3) ** 2 / NU) * 1000   # lattice velocity -> mm/s (dx/dt, eq 25)
Z_FLOOR = 15                                                    # smooth floor height (cells)

FILLS = {"W4": "W=4, sigma 0.1 (job 8815366)"}
SECTIONS = {150: "middle of train 1", 420: "middle of train 2", 700: "smooth run-out"}    # cell number x

y = (np.arange(1, YN + 1) - 0.5) * DX                           # mm from the wall
z = (np.arange(1, ZN + 1) - 0.5 - Z_FLOOR) * DX                 # mm above the floor
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]    # index = cell number - 1


def load(fill):
    d = np.load(os.path.join(HERE, "fields3d", f"fields3d_iter_{ITER:06d}.npz"))
    return d["phi"].astype(float), d["u"].astype(float) * TO_MM_S


def section(phi, u, x):
    """y-z plane at cell number x: liquid mask, u_x, (u_y, u_z), in-plane speed; arrays are (z, y)."""
    i = x - 1
    liquid = ((phi[i] >= 0.5) & ~solid[i]).T
    u_x, u_y, u_z = (np.where(liquid, u[k, i].T, 0.0) for k in range(3))
    return liquid, u_x, u_y, u_z, np.hypot(u_y, u_z)


data = {fill: load(fill) for fill in FILLS}
VMIN, VMAX = min(d[0][~solid].min() for d in data.values()), max(d[0][~solid].max() for d in data.values())    # phi colorbar range = data range
speed_max = max(section(*data[f], x)[4].max() for f in FILLS for x in SECTIONS)

fig, axes = plt.subplots(len(FILLS) * len(SECTIONS), 1, figsize=(11, 4.6 * len(SECTIONS) * len(FILLS) + 0.8), layout="constrained", squeeze=False)    # sections stacked underneath each other
axes = axes.reshape(len(FILLS), len(SECTIONS))
for row, (fill, fill_name) in enumerate(FILLS.items()):
    phi, u = data[fill]
    u_mean = (u[0] * phi)[~solid].sum() / phi[~solid].sum()      # mean u_x of the liquid (mm/s)
    for ax, (x, section_name) in zip(axes[row], SECTIONS.items()):
        liquid, u_x, u_y, u_z, speed = section(phi, u, x)
        phi_mesh = ax.pcolormesh(y, z, np.ma.masked_where(solid[x - 1].T, phi[x - 1].T), cmap="RdBu", shading="nearest", vmin=VMIN, vmax=VMAX)    # background = phi, standard phi colorbar
        ax.pcolormesh(y, z, np.ma.masked_where(~solid[x - 1].T, solid[x - 1].T), cmap="Greys", vmin=0, vmax=2, shading="nearest")
        lines = ax.streamplot(y, z, np.ma.masked_where(~liquid, u_y), np.ma.masked_where(~liquid, u_z), color=speed,
                              cmap="turbo", norm=Normalize(0, speed_max), density=2.2, linewidth=1.0, arrowsize=0.7)
        ax.contour(y, z, phi[x - 1].T, levels=[0.5], colors="k", linewidths=1.0)
        ax.set(xlim=(0, 1.2), ylim=(-0.22, 0.56), aspect="equal", xlabel="y (mm)", ylabel="z above floor (mm)",
               title=f"{fill_name}: x = {x}, {section_name}\nmax |u_yz| = {speed.max():.1f} mm/s = {speed.max() / u_mean * 100:.0f} % of mean u_x")
        ax.title.set_fontsize(9)

fig.colorbar(lines.lines, ax=axes, pad=0.01, fraction=0.025, label="streamline colour: in-plane speed |u_yz| (mm/s)")
fig.colorbar(phi_mesh, ax=axes, pad=0.01, fraction=0.025, label="background: phi")
fig.suptitle(f"cross sections at iteration {ITER}")
out = os.path.join(HERE, f"xs1_sections_iter{ITER:06d}.png")
fig.savefig(out, dpi=110)
print("wrote", out)
