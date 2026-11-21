"""Fig. 4 of Saliba et al. (left image, 1 ml/min) for the herringbone simulation: streamlines along the film seen in the flow direction.
3D streamlines of the velocity field (liquid only) are started on the plane x = X0 and followed to x = X1 (default x100-260 = 3.0 mm, train 1);
they are drawn projected on the y-z plane, coloured by the integration time along each streamline.  White line = inner contour of the smooth
channel (side walls and floor); the streamlines below it are the flow in the grooves of the herringbone pattern.
Integration: arc-length steps of 0.4 cell with a midpoint (RK2) rule, trilinear interpolation of u; a streamline stops when it leaves the liquid
(phi < 0.5 or solid), reaches x = X1 or has no velocity.
Usage (PC, herringbone\\run_3d):  python xs1_fig4.py fields3d/fields3d_iter_010200.npz [label] [X0 X1]    ->  xs1_fig4_iterNNNNNN.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.ndimage import map_coordinates

os.environ["TP"] = "4.9"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "W=4, job 8815366"
X0, X1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (100, 260)      # cell numbers, 1-based
ITER = int(path[-10:-4])
d = np.load(path)
phi, u = d["phi"].astype(np.float64), d["u"].astype(np.float64)
XN, YN, ZN = phi.shape
DX = 1.2 / YN
Z_FLOOR = 15
TAU_F, NU = 0.6, 1.0e-6
DT_MS = (1 / 3) * (TAU_F - 0.5) * (DX * 1e-3) ** 2 / NU * 1e3                    # lattice time step in ms
U_UNIT = (DX * 1e-3) / (DT_MS * 1e-3) * 100                                       # lattice velocity -> cm/s
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
liq = (phi >= 0.5) & ~solid
Q = (u[0] * liq).sum(axis=(1, 2)) * U_UNIT * 1e-2 * (DX * 1e-3) ** 2 * 6e7        # ml/min per x layer
print(f"{path}: {label}; window x{X0}-{X1}; volume flow rate there {Q[X0 - 1:X1].mean():.2f} ml/min (paper figure: 1 ml/min)")

# seeds: every SEED cells (env SEED, default 1.5) on the plane x = X0 inside the liquid
SP = float(os.environ.get("SEED", 1.5))                                            # seed spacing in the grooves (z < 0), cells
SPF = float(os.environ.get("SEEDF", SP))                                           # seed spacing in the film above the smooth floor (z >= 0), cells
LW = float(os.environ.get("LW", 1.0))                                              # line width of the plot
def seeds_on(sp, zmask):
    sy, sz = np.meshgrid(np.arange(0, YN, sp), np.arange(0, ZN, sp), indexing="ij")
    ok = liq[X0 - 1][np.round(sy).astype(int), np.round(sz).astype(int)] & zmask(sz)
    return np.c_[np.full(ok.sum(), X0 - 1.0), sy[ok], sz[ok]]
pos = np.vstack([seeds_on(SP, lambda z: z < Z_FLOOR), seeds_on(SPF, lambda z: z >= Z_FLOOR)])
n = len(pos)
print(f"{n} seeds")


def vel(p):
    return np.stack([map_coordinates(u[k], p.T, order=1, mode="nearest") for k in range(3)], axis=1)


def inside(p):
    i = np.clip(np.round(p).astype(int), 0, [XN - 1, YN - 1, ZN - 1])
    return liq[i[:, 0], i[:, 1], i[:, 2]] & (p[:, 1] >= 0) & (p[:, 1] <= YN - 1) & (p[:, 2] >= 0) & (p[:, 2] <= ZN - 1)


STEP = 0.4
alive = np.ones(n, bool)
traj = [pos.copy()]
tt = [np.zeros(n)]
t = np.zeros(n)
for _ in range(4000):
    v1 = vel(pos)
    sp = np.linalg.norm(v1, axis=1)
    ok = alive & (sp > 1e-6)
    mid = pos + 0.5 * STEP * v1 / np.maximum(sp, 1e-30)[:, None]
    v2 = vel(mid)
    sp2 = np.linalg.norm(v2, axis=1)
    new = pos + STEP * v2 / np.maximum(sp2, 1e-30)[:, None]
    ok &= (sp2 > 1e-6) & inside(new) & (new[:, 0] <= X1 - 1)
    t = t + np.where(ok, STEP / np.maximum(sp2, 1e-30), 0.0)                    # lattice time steps
    pos = np.where(ok[:, None], new, pos)
    alive = ok
    traj.append(np.where(ok[:, None], pos, np.nan))
    tt.append(np.where(ok, t, np.nan))
    if not alive.any():
        break
traj, tt = np.stack(traj, axis=1), np.stack(tt, axis=1) * DT_MS                  # (n, steps, 3), (n, steps) in ms
length = np.isfinite(tt).sum(axis=1)
reached = np.nanmax(traj[:, :, 0], axis=1) >= X1 - 3
print(f"streamlines reaching x{X1}: {int(reached.sum())} of {n}; integration time median {np.nanmedian(np.nanmax(tt, axis=1)):.1f} ms, max {np.nanmax(tt):.1f} ms")

y_mm, z_mm = (traj[:, :, 1] + 0.5) * DX, (traj[:, :, 2] + 0.5 - Z_FLOOR) * DX
segs, cols = [], []
for k in range(n):
    m = np.isfinite(y_mm[k])
    pts = np.c_[y_mm[k][m], z_mm[k][m]]
    if len(pts) < 2:
        continue
    segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
    cols.append(tt[k][m][:-1])
segs, cols = np.concatenate(segs), np.concatenate(cols)
order = np.argsort(cols)

# ---- density metrics, the same as measured on Saliba's image (xs1_fig4_ref_density.py): window y 0..1.2 mm, z -0.135..+0.37 mm --------------------------
WZ0, WZ1 = -0.135, 0.37
mid = segs.mean(axis=1)
inwin = (mid[:, 0] > 0) & (mid[:, 0] < 1.2) & (mid[:, 1] > WZ0) & (mid[:, 1] < WZ1)
seglen = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
area = 1.2 * (WZ1 - WZ0)
print(f"seed spacing {SP} cells: projected line length in the window {seglen[inwin].sum():.1f} mm -> length density {seglen[inwin].sum() / area:.1f} mm per mm^2 (his: 35-45)")
for z0 in (-0.06, 0.04, 0.10):
    cross = ((segs[:, 0, 1] - z0) * (segs[:, 1, 1] - z0) < 0) & (mid[:, 0] > 0) & (mid[:, 0] < 1.2)
    print(f"  z = {z0:+.2f} mm: {int(cross.sum())} line crossings = {cross.sum() / 1.2:.0f} per mm of width (his: 12 / 22 / 40 for -0.06 / +0.04 / +0.10, merged runs)")

# ---- ink coverage on the pixel scale of his image (354.6 px/mm): the line width is chosen so that the coloured fraction of the window equals his 15.6 % -------
import matplotlib.colors as mcol
from matplotlib.colors import ListedColormap
CMAP = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.16, 1.0, 256)))        # turbo without its near-black start (invisible on the dark background; his blue is bright)
HIS_PX_MM, HIS_COVER = 354.6, 15.6


def coverage(lw_px):
    """coloured pixel fraction of the window y 0..1.2, z WZ0..WZ1 rendered at his scale with lines of lw_px pixels"""
    wpx, hpx = int(round(1.2 * HIS_PX_MM)), int(round((WZ1 - WZ0) * HIS_PX_MM))
    f = plt.figure(figsize=(wpx / 100, hpx / 100), dpi=100, facecolor="#10163a")
    a_ = f.add_axes([0, 0, 1, 1]); a_.set_facecolor("#10163a"); a_.axis("off")
    a_.add_collection(LineCollection(segs[order], array=cols[order], cmap=CMAP, linewidths=lw_px * 72 / 100, clim=(0, VMAX_)))
    a_.set(xlim=(0, 1.2), ylim=(WZ0, WZ1))
    f.canvas.draw()
    img = np.asarray(f.canvas.buffer_rgba())[:, :, :3] / 255.0
    plt.close(f)
    hsv = mcol.rgb_to_hsv(img)
    return float(((hsv[:, :, 1] > 0.45) & (hsv[:, :, 2] > 0.45)).mean() * 100)


VMAX_ = float(np.nanpercentile(cols, 95))
cands = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
cov = [coverage(w) for w in cands]
LW_PX = cands[int(np.argmin([abs(c - HIS_COVER) for c in cov]))]
print("ink coverage of the window (his: 15.6 %) for line widths in px at his scale: " + ", ".join(f"{w} px: {c:.1f} %" for w, c in zip(cands, cov)) + f"  -> using {LW_PX} px")

BG = "#10163a"
fig, ax = plt.subplots(figsize=(12, 6.4), facecolor=BG)
ax.set_facecolor(BG)
VMAX = float(np.nanpercentile(cols, 95))                                          # colour range up to the 95th percentile (the slow groove streamlines would stretch it)
PX_MAIN = 12 * 110 / 1.68                                                          # px per mm of the plot below (12 in wide at 110 dpi, x range 1.68 mm)
lc = LineCollection(segs[order], array=cols[order], cmap=CMAP, linewidths=(LW_PX / HIS_PX_MM) * PX_MAIN * 72 / 110, clim=(0, VMAX))
ax.add_collection(lc)
ax.plot([0, 0, 1.2, 1.2], [0.4, 0, 0, 0.4], color="white", lw=3, solid_joinstyle="miter")                      # inner contour of the smooth channel: walls and floor
ax.annotate("", xy=(0, 0.47), xytext=(1.2, 0.47), arrowprops=dict(arrowstyle="<->", color="white", lw=2.2))
ax.text(0.6, 0.495, "1.2 mm", color="white", fontsize=26, fontweight="bold", ha="center", va="bottom")
ax.annotate("", xy=(1.27, 0.0), xytext=(1.27, 0.4), arrowprops=dict(arrowstyle="<->", color="white", lw=2.2))
ax.text(1.30, 0.0, "0.4 mm", color="white", fontsize=26, fontweight="bold", ha="left", va="center")
ax.set(xlim=(-0.06, 1.62), ylim=(-0.22, 0.62), aspect="equal")
ax.axis("off")
cb = fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.5)
cb.set_label("integration time along the streamline (ms)", color="white")
cb.ax.xaxis.set_tick_params(color="white", labelcolor="white")
fig.suptitle(f"Simulation ({label}, iteration {ITER}): streamlines along the film in the flow direction, x{X0}-{X1} ({(X1 - X0 + 1) * DX:.1f} mm), {Q[X0 - 1:X1].mean():.2f} ml/min", color="white", fontsize=12)
out = os.path.join(HERE, f"xs1_fig4_iter{ITER:06d}_x{X0}-{X1}_seed{SP}_film{SPF}.png")
fig.savefig(out, dpi=110, facecolor=BG)
print("wrote", out)
