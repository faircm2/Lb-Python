"""Fig. 4 (a) of Saliba et al. (1 ml/min) next to the simulation, made the way the paper describes it (p. 5: "streamlines were plotted with the inlet plane as the starting point.
A streamwise projection of these streamlines is shown in Fig. 4"; caption: the colour is the integration time along each streamline):
  seeds on the inlet plane x = 1 (every SEED cells in the liquid of that plane), full 3-D streamlines (all three velocity components, RK2, 0.4 cell steps, trilinear u)
  followed downstream until they leave the liquid or reach the outlet x = XN, drawn projected on the y-z plane, coloured by the integration time.
Both panels have the same scale (pixels per mm), the same channel outline and dimension arrows (white), and the same colour map (turbo, as in his image; his image has no colour bar).
Usage (PC, herringbone\\run_3d):  python xs1_fig4_inlet_compare.py fields3d_W14long/fields3d_iter_520000.npz "W=14, job 8888085_1" [SEED]
Writes xs1_fig4_inlet_compare_iterNNNNNN.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.ndimage import map_coordinates

os.environ["TP"] = "4.9"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "W=14, job 8888085_1"
SEED = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5                    # seed spacing in cells; 3.5 (49 seeds) has the smallest summed difference to his ink coverage per height band (xs1_fig4_seedsweep2.py) and was picked by the user
ITER = int(path[-10:-4])
d = np.load(path)
phi, u = d["phi"].astype(np.float64), d["u"].astype(np.float64)
XN, YN, ZN = phi.shape
DX = 1.2 / YN
Z_FLOOR = 15
TAU_F, NU = 0.6, 1.0e-6
DT_MS = (1 / 3) * (TAU_F - 0.5) * (DX * 1e-3) ** 2 / NU * 1e3
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
liq = (phi >= 0.5) & ~solid
CMAP_PLOT = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.16, 0.90, 256)))
BG = "#10163a"
PX_MM = (450.1 - 34.7) / 1.2                                               # pixels per mm of his image (centres of his white side-wall lines)
ROW_FLOOR = 193                                                            # row of his floor line
H_WALL, Z_ARROW, Z_TEXT_TOP, V_ARROW, Z_TEXT_SIDE = 0.382, 0.429, 0.473, (-0.075, 0.424), -0.078   # elevations (mm above the floor) of his wall tops, arrow, lettering

# ---- streamlines from the inlet plane -----------------------------------------------------------------------------------------------------------------------
sy, sz = np.meshgrid(np.arange(0, YN, SEED), np.arange(0, ZN, SEED), indexing="ij")
ok = liq[0][np.clip(np.round(sy).astype(int), 0, YN - 1), np.clip(np.round(sz).astype(int), 0, ZN - 1)]
pos = np.c_[np.zeros(ok.sum()), np.clip(np.round(sy[ok]), 0, YN - 1), np.clip(np.round(sz[ok]), 0, ZN - 1)]
n = len(pos)


def vel(p):
    return np.stack([map_coordinates(u[k], p.T, order=1, mode="nearest") for k in range(3)], axis=1)


def inside(p):
    i = np.clip(np.round(p).astype(int), 0, [XN - 1, YN - 1, ZN - 1])
    return liq[i[:, 0], i[:, 1], i[:, 2]] & (p[:, 1] >= 0) & (p[:, 1] <= YN - 1) & (p[:, 2] >= 0) & (p[:, 2] <= ZN - 1)


STEP = 0.4
alive = np.ones(n, bool)
traj, tt, t = [pos.copy()], [np.zeros(n)], np.zeros(n)
for _ in range(8000):
    v1 = vel(pos)
    sp1 = np.linalg.norm(v1, axis=1)
    okk = alive & (sp1 > 1e-6)
    mid = pos + 0.5 * STEP * v1 / np.maximum(sp1, 1e-30)[:, None]
    v2 = vel(mid)
    sp2 = np.linalg.norm(v2, axis=1)
    new = pos + STEP * v2 / np.maximum(sp2, 1e-30)[:, None]
    okk &= (sp2 > 1e-6) & inside(new) & (new[:, 0] <= XN - 1)
    t = t + np.where(okk, STEP / np.maximum(sp2, 1e-30), 0.0)
    pos = np.where(okk[:, None], new, pos)
    alive = okk
    traj.append(np.where(okk[:, None], pos, np.nan))
    tt.append(np.where(okk, t, np.nan))
    if not alive.any():
        break
traj, tt = np.stack(traj, axis=1), np.stack(tt, axis=1) * DT_MS
reached = np.nanmax(traj[:, :, 0], axis=1) >= XN - 3
y_mm, z_mm = (traj[:, :, 1] + 0.5) * DX, (traj[:, :, 2] + 0.5 - Z_FLOOR) * DX
segs, cols = [], []
for k in range(n):
    m = np.isfinite(y_mm[k])
    p_ = np.c_[y_mm[k][m], z_mm[k][m]]
    if len(p_) < 2:
        continue
    segs.append(np.stack([p_[:-1], p_[1:]], axis=1))
    cols.append(tt[k][m][:-1])
segs, cols = np.concatenate(segs), np.concatenate(cols)
vmax = float(np.nanpercentile(cols, 95))
WZ0, WZ1 = -0.135, 0.37                                                    # the window in which his density was measured (xs1_fig4_ref_density.py)
mid_ = segs.mean(axis=1)
inwin = (mid_[:, 0] > 0) & (mid_[:, 0] < 1.2) & (mid_[:, 1] > WZ0) & (mid_[:, 1] < WZ1)
dens = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)[inwin].sum() / (1.2 * (WZ1 - WZ0))
print(f"{n} seeds on the inlet plane (spacing {SEED} cells), {int(reached.sum())} reach the outlet x = {XN}; integration time median {np.nanmedian(np.nanmax(tt, axis=1)):.0f} ms, "
      f"max {np.nanmax(tt):.0f} ms, colours up to {vmax:.0f} ms (95th percentile); projected line length density {dens:.0f} mm per mm^2 (his image: 35-45)")


def draw_sim(ax, pxmm, dpi, fs):
    """streamlines and channel outline in an axes with pxmm pixels per mm (figure dpi dpi); returns the line collection"""
    ax.set_facecolor(BG)
    o = np.argsort(cols)
    lc = LineCollection(segs[o], array=cols[o], cmap=CMAP_PLOT, linewidths=1.5 * (pxmm / PX_MM) * 72 / dpi, clim=(0, vmax))
    ax.add_collection(lc)
    # heights of the outline, the arrows and the lettering are those of his figure (measured on the comparison plot: floor at row 456.5, wall tops row 219, 1.2 mm arrow row 189,
    # "1.2mm" lettering centre row 161.5, vertical arrow rows 192-503, "0.4mm" lettering centre row 505; 623.2 px per mm)
    ax.plot([0, 0, 1.2, 1.2], [H_WALL, 0, 0, H_WALL], color="white", lw=3 * fs / 26, solid_joinstyle="miter")
    ax.annotate("", xy=(0, Z_ARROW), xytext=(1.2, Z_ARROW), arrowprops=dict(arrowstyle="<->", color="white", lw=2.2 * fs / 26))
    ax.text(0.6, Z_TEXT_TOP, "1.2 mm", color="white", fontsize=fs, fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(1.27, V_ARROW[0]), xytext=(1.27, V_ARROW[1]), arrowprops=dict(arrowstyle="<->", color="white", lw=2.2 * fs / 26))
    ax.text(1.30, Z_TEXT_SIDE, "0.4 mm", color="white", fontsize=fs, fontweight="bold", ha="left", va="center")
    ax.set(xlim=(-0.06, 1.62), ylim=(-0.22, 0.62), aspect="equal")
    ax.axis("off")
    return lc


# ---- side by side: his figure and the simulation at the same scale ---------------------------------------------------------------------------------------------
DPI, SC, M, GAP, TOPM = 100, 1.8, 20, 40, 70
PXMM = PX_MM * SC                                                           # pixels per mm in both panels
ref = mi.imread(os.path.join(HERE, "saliba_fig4a_reference.png"))[:, :, :3].astype(float)
if ref.max() > 1.5:
    ref /= 255.0
# remove the white picture frame: the white region connected to the border becomes the plot background, and the edge of his dark panel fades into it over FADE pixels
lab, _ = ndimage.label(ref.min(axis=2) > 0.9)
border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
frame = np.isin(lab, border[border > 0])
FADE = 22
alpha = np.clip(ndimage.distance_transform_edt(~frame) / FADE, 0, 1)[:, :, None]
alpha = alpha * alpha * (3 - 2 * alpha)                                     # smooth step
white = ndimage.binary_dilation((ref.min(axis=2) > 0.75) & ~frame, iterations=2)  # his white lines and lettering inside the panel keep their full brightness
alpha = np.where(white[:, :, None], 1.0, alpha)
bg_rgb = np.array(matplotlib.colors.to_rgb(BG))
ref = alpha * ref + (1 - alpha) * bg_rgb
rh, rw = ref.shape[:2]
ref_w, ref_h = rw * SC, rh * SC
sim_w, sim_h = 1.68 * PXMM, 0.84 * PXMM                                     # x range -0.06..1.62 mm, z range -0.22..0.62 mm of draw_sim
above = max(ROW_FLOOR * SC, 0.62 * PXMM)                                    # both floor lines are put at the same height
below = max((rh - ROW_FLOOR) * SC, 0.22 * PXMM + 110)                       # room for the colour bar under the simulation panel
W_PX, H_PX = M + ref_w + GAP + sim_w + M, TOPM + above + below + M
floor_y = M + below
fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor=BG)
ax0 = fig.add_axes([M / W_PX, (floor_y - (rh - ROW_FLOOR) * SC) / H_PX, ref_w / W_PX, ref_h / H_PX])
ax0.imshow(ref, interpolation="lanczos")
ax0.axis("off")
sim_left = M + ref_w + GAP
ax1 = fig.add_axes([sim_left / W_PX, (floor_y - 0.22 * PXMM) / H_PX, sim_w / W_PX, sim_h / H_PX])
lc = draw_sim(ax1, PXMM, DPI, 26 * PXMM / 786)
chan_mid = sim_left + (0.6 + 0.06) * PXMM                                   # middle of the channel (y = 0.6 mm; the axes start at y = -0.06 mm)
cax = fig.add_axes([(chan_mid - 0.25 * sim_w) / W_PX, (M + 25) / H_PX, 0.5 * sim_w / W_PX, 20 / H_PX])
cb = fig.colorbar(lc, cax=cax, orientation="horizontal")
cb.set_label("integration time along the streamline (ms)", color="white")
cb.ax.xaxis.set_tick_params(color="white", labelcolor="white")
fig.text((M + ref_w / 2) / W_PX, (H_PX - 25) / H_PX, "Saliba et al., Fig. 4 (a), 1 ml/min", color="white", ha="center", va="center", fontsize=13)
fig.text((sim_left + sim_w / 2) / W_PX, (H_PX - 28) / H_PX,
         f"Simulation ({label}, iteration {ITER}): streamlines seeded on the inlet plane x = 1 and followed to the outlet ({XN * DX:.1f} mm)\n1 ml/min, projected on the transverse plane",
         color="white", ha="center", va="center", fontsize=11)
out = os.path.join(HERE, f"xs1_fig4_inlet_compare_iter{ITER:06d}_{n}seeds.png")
fig.savefig(out, dpi=DPI, facecolor=BG)
print("wrote", out)
