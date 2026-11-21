"""Streamline plots at many cross-sections along the channel; which one looks most like Saliba's Fig. 4a (1 ml/min)?
For every start cell X0 (step STEP) streamlines are seeded on the plane x = X0 (grooves: every SEED cells, film above the smooth floor: every SEEDF cells) and followed
for L cells downstream (default 1.0 mm), projected on the y-z plane exactly as in xs1_fig4.py.  Each result is rasterised at the pixel scale of his image
(354.6 px/mm, window y 0..1.2 mm, z -0.137..+0.371 mm, lines 1.5 px) and compared with his image (saliba_fig4a_reference.png):
  score = Pearson correlation of the Gaussian-blurred (sigma 10 px = 0.028 mm) maps of coloured pixels  (where the streamline ink is)
  also printed: ink fraction, left/right ink ratio in the grooves (z < 0) and the y of the vortex divider (largest mean |u_z| at groove depth).
The best BEST candidates per snapshot (well separated: at least MINSEP cells apart) are drawn in the style of his figure.
Usage (PC, herringbone\\run_3d):  python xs1_fig4_scan.py fields3d_W7/fields3d_iter_019222.npz "W=7, job 8866565" [BEST]
Writes xs1_fig4_scan_iterNNNNNN.csv, xs1_fig4_scan_iterNNNNNN.png (score along x), xs1_fig4_best_iterNNNNNN_rankK_xX0.png and a comparison with his image.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcol
import matplotlib.image as mi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter, map_coordinates

os.environ["TP"] = "4.9"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "W=7, job 8866565"
BEST = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ITER = int(path[-10:-4])
SEED, SEEDF, L_CELLS, STEP, MINSEP = 2.4, 1.0, 54, 6, 40
d = np.load(path)
phi, u = d["phi"].astype(np.float64), d["u"].astype(np.float64)
XN, YN, ZN = phi.shape
DX = 1.2 / YN
Z_FLOOR = 15
TAU_F, NU = 0.6, 1.0e-6
DT_MS = (1 / 3) * (TAU_F - 0.5) * (DX * 1e-3) ** 2 / NU * 1e3
U_UNIT = (DX * 1e-3) / (DT_MS * 1e-3) * 100
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
liq = (phi >= 0.5) & ~solid
CMAP = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.16, 1.0, 256)))            # used for the ink score (unchanged)
CMAP_PLOT = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.16, 0.90, 256)))      # used for drawing: without turbo's dark-red end, which disappears on the dark background
BG = "#10163a"

# ---- his image: coloured-pixel mask inside his white frame --------------------------------------------------------------------------------------
PX_MM = (450.1 - 34.7) / 1.2                                                        # centres of his white side-wall lines (measured in saliba_fig4a_reference.png)
ROW_FLOOR, ROW_TOP, ROW_BOT, COL_L, COL_R = 193, 62, 241, 37, 458
img = mi.imread(os.path.join(HERE, "saliba_fig4a_reference.png"))[:, :, :3].astype(float)
if img.max() > 1.5:
    img /= 255.0
hsv_ = mcol.rgb_to_hsv(img)
his = ((hsv_[:, :, 1] > 0.45) & (hsv_[:, :, 2] > 0.45) & ~(img.min(axis=2) > 0.88))[ROW_TOP:ROW_BOT + 1, COL_L:COL_R + 1]
HPX, WPX = his.shape
Y0, Y1 = (COL_L - 34.7) / PX_MM, (COL_R + 1 - 34.7) / PX_MM
Z1, Z0 = (ROW_FLOOR - (ROW_TOP - 0.5)) / PX_MM, (ROW_FLOOR - (ROW_BOT + 0.5)) / PX_MM
his_blur = gaussian_filter(his.astype(float), 10)
print(f"his mask {WPX} x {HPX} px, window y {Y0:.3f}..{Y1:.3f} mm, z {Z0:.3f}..{Z1:.3f} mm, ink {his.mean() * 100:.1f} %")


def lines(X0):
    """segments (m, 2, 2) of (y, z) in mm and integration times (ms) of the streamlines seeded on the plane x = X0 (1-based cell) and followed L_CELLS cells"""
    X1 = X0 + L_CELLS
    pos = []
    for sp, zm in ((SEED, lambda z: z < Z_FLOOR), (SEEDF, lambda z: z >= Z_FLOOR)):
        sy, sz = np.meshgrid(np.arange(0, YN, sp), np.arange(0, ZN, sp), indexing="ij")
        ok = liq[X0 - 1][np.round(sy).astype(int), np.round(sz).astype(int)] & zm(sz)
        pos.append(np.c_[np.full(ok.sum(), X0 - 1.0), sy[ok], sz[ok]])
    pos = np.vstack(pos)
    n = len(pos)

    def vel(p):
        return np.stack([map_coordinates(u[k], p.T, order=1, mode="nearest") for k in range(3)], axis=1)

    def inside(p):
        i = np.clip(np.round(p).astype(int), 0, [XN - 1, YN - 1, ZN - 1])
        return liq[i[:, 0], i[:, 1], i[:, 2]] & (p[:, 1] >= 0) & (p[:, 1] <= YN - 1) & (p[:, 2] >= 0) & (p[:, 2] <= ZN - 1)

    STEPL = 0.4
    alive = np.ones(n, bool)
    traj, tt, t = [pos.copy()], [np.zeros(n)], np.zeros(n)
    for _ in range(1500):
        v1 = vel(pos)
        sp1 = np.linalg.norm(v1, axis=1)
        ok = alive & (sp1 > 1e-6)
        mid = pos + 0.5 * STEPL * v1 / np.maximum(sp1, 1e-30)[:, None]
        v2 = vel(mid)
        sp2 = np.linalg.norm(v2, axis=1)
        new = pos + STEPL * v2 / np.maximum(sp2, 1e-30)[:, None]
        ok &= (sp2 > 1e-6) & inside(new) & (new[:, 0] <= X1 - 1)
        t = t + np.where(ok, STEPL / np.maximum(sp2, 1e-30), 0.0)
        pos = np.where(ok[:, None], new, pos)
        alive = ok
        traj.append(np.where(ok[:, None], pos, np.nan))
        tt.append(np.where(ok, t, np.nan))
        if not alive.any():
            break
    traj, tt = np.stack(traj, axis=1), np.stack(tt, axis=1) * DT_MS
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
    return segs, cols


def raster(segs, cols, vmax):
    f = plt.figure(figsize=(WPX / 100, HPX / 100), dpi=100, facecolor=BG)
    a = f.add_axes([0, 0, 1, 1]); a.set_facecolor(BG); a.axis("off")
    o = np.argsort(cols)
    a.add_collection(LineCollection(segs[o], array=cols[o], cmap=CMAP, linewidths=1.5 * 72 / 100, clim=(0, vmax)))
    a.set(xlim=(Y0, Y1), ylim=(Z0, Z1))
    f.canvas.draw()
    im = np.asarray(f.canvas.buffer_rgba())[:, :, :3] / 255.0
    plt.close(f)
    h = mcol.rgb_to_hsv(im)
    return (h[:, :, 1] > 0.45) & (h[:, :, 2] > 0.45)


def divider(X0):
    y = (np.arange(YN) + 0.5) * DX
    z = (np.arange(ZN) + 0.5 - Z_FLOOR) * DX
    sl = slice(X0 - 1, X0 - 1 + L_CELLS)
    uz = np.nanmean(np.where(liq[sl], u[2][sl], np.nan), axis=0)
    kz = (z >= -0.2) & (z <= -0.02); jy = (y >= 0.2) & (y <= 1.0)
    sub = np.abs(np.where(np.isfinite(uz), uz, 0))[np.ix_(jy, kz)].mean(axis=1)
    return float(y[jy][np.argmax(sub)])


def draw_sim(ax, segs, cols, pxmm, dpi, fs):
    """streamlines and channel outline in an axes whose scale is pxmm pixels per mm (figure dpi dpi); returns the line collection"""
    vmax = float(np.nanpercentile(cols, 95))
    ax.set_facecolor(BG)
    o = np.argsort(cols)
    lc = LineCollection(segs[o], array=cols[o], cmap=CMAP_PLOT, linewidths=1.5 * (pxmm / PX_MM) * 72 / dpi, clim=(0, vmax))
    ax.add_collection(lc)
    ax.plot([0, 0, 1.2, 1.2], [0.4, 0, 0, 0.4], color="white", lw=3 * fs / 26, solid_joinstyle="miter")
    ax.annotate("", xy=(0, 0.47), xytext=(1.2, 0.47), arrowprops=dict(arrowstyle="<->", color="white", lw=2.2 * fs / 26))
    ax.text(0.6, 0.495, "1.2 mm", color="white", fontsize=fs, fontweight="bold", ha="center", va="bottom")
    ax.annotate("", xy=(1.27, 0.0), xytext=(1.27, 0.4), arrowprops=dict(arrowstyle="<->", color="white", lw=2.2 * fs / 26))
    ax.text(1.30, 0.0, "0.4 mm", color="white", fontsize=fs, fontweight="bold", ha="left", va="center")
    ax.set(xlim=(-0.06, 1.62), ylim=(-0.22, 0.62), aspect="equal")
    ax.axis("off")
    return lc


def plot(segs, cols, X0, score, out):
    fig, ax = plt.subplots(figsize=(12, 6.4), facecolor=BG)
    lc = draw_sim(ax, segs, cols, 12 * 110 / 1.68, 110, 26)
    cb = fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.5)
    cb.set_label("integration time along the streamline (ms)", color="white")
    cb.ax.xaxis.set_tick_params(color="white", labelcolor="white")
    fig.suptitle(f"Simulation ({label}, iteration {ITER}): streamlines along the film in the flow direction, seeded at x = {X0} ({(X0 - 0.5) * DX:.2f} mm), followed {L_CELLS * DX:.1f} mm\ncorrelation with Saliba et al., Fig. 4 (a): {score:.2f}", color="white", fontsize=11)
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


rows = []
cache = {}
for X0 in range(45, XN - L_CELLS - 5, STEP):
    segs, cols = lines(X0)
    if len(segs) < 50:
        continue
    vmax = float(np.nanpercentile(cols, 95))
    m = raster(segs, cols, vmax)
    mb = gaussian_filter(m.astype(float), 10)
    score = float(np.corrcoef(mb.ravel(), his_blur.ravel())[0, 1])
    zrow = int(round((ROW_FLOOR - ROW_TOP)))                                              # row of z = 0 inside the cropped window
    half = WPX // 2
    lr = float(m[zrow:, :half].sum() / max(1, m[zrow:, half:].sum()))
    rows.append((X0, (X0 - 0.5) * DX, score, float(m.mean() * 100), lr, divider(X0)))
    cache[X0] = (segs, cols)
    print(f"x0 {X0:4d} ({(X0 - 0.5) * DX:5.2f} mm): score {score:+.3f}  ink {m.mean() * 100:5.1f} %  groove ink left/right {lr:5.2f}  divider y {rows[-1][5]:.2f} mm", flush=True)

his_lr = float(his[int(ROW_FLOOR - ROW_TOP):, :WPX // 2].sum() / max(1, his[int(ROW_FLOOR - ROW_TOP):, WPX // 2:].sum()))
print(f"his image: ink {his.mean() * 100:.1f} %, groove ink left/right {his_lr:.2f}")
with open(os.path.join(HERE, f"xs1_fig4_scan_iter{ITER:06d}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x0_cell", "x_mm", "score", "ink_percent", "groove_ink_left_over_right", "divider_y_mm"])
    w.writerows(rows)

# ranking: only cross-sections with grooves (ink 15-45 %) and the larger groove vortex on the left (ink ratio 1.4-3.5, his 2.32);
# rank by  correlation - 0.1 |ln(ratio / his ratio)|
eligible = [r for r in rows if 15 <= r[3] <= 45 and 1.4 <= r[4] <= 3.5]
rows_sorted = sorted(eligible, key=lambda r: -(r[2] - 0.1 * abs(np.log(r[4] / his_lr))))
picked = []
for r in rows_sorted:
    if all(abs(r[0] - p[0]) >= MINSEP for p in picked):
        picked.append(r)
    if len(picked) == BEST:
        break
print("best:", [(p[0], round(p[2], 3)) for p in picked])
for k, r in enumerate(picked, 1):
    plot(*cache[r[0]], r[0], r[2], os.path.join(HERE, f"xs1_fig4_best_iter{ITER:06d}_rank{k}_x{r[0]}.png"))

fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
ax.plot([r[1] for r in rows], [r[2] for r in rows], "o-", color="#2171b5")
for k, r in enumerate(picked, 1):
    ax.annotate(str(k), (r[1], r[2]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=12, fontweight="bold")
ax.set(xlabel="start of the streamlines along the channel, x (mm)", ylabel="similarity to Saliba et al., Fig. 4 (a)\n(correlation of the blurred streamline ink)", title=f"Simulation ({label}, iteration {ITER})")
ax.grid(alpha=0.3)
fig.savefig(os.path.join(HERE, f"xs1_fig4_scan_iter{ITER:06d}.png"), dpi=110)

# side by side: his figure and the best one of the simulation, both at the same scale (channel 1.2 mm wide = the same number of pixels in both panels)
best = picked[0]
DPI, SC, M, GAP, TOPM = 100, 1.8, 20, 40, 70
PXMM = PX_MM * SC                                                                   # pixels per mm in both panels
ref = mi.imread(os.path.join(HERE, "saliba_fig4a_reference.png"))
rh, rw = ref.shape[:2]
ref_w, ref_h = rw * SC, rh * SC
sim_w, sim_h = 1.68 * PXMM, 0.84 * PXMM                                             # x range -0.06..1.62 mm, z range -0.22..0.62 mm of draw_sim
above = max(ROW_FLOOR * SC, 0.62 * PXMM)                                            # both floor lines are put at the same height
below = max((rh - ROW_FLOOR) * SC, 0.22 * PXMM + 110)                               # room for the colour bar under the simulation panel
W_PX, H_PX = M + ref_w + GAP + sim_w + M, TOPM + above + below + M
floor_y = M + below                                                                 # pixels from the bottom of the figure
fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor=BG)
ax0 = fig.add_axes([M / W_PX, (floor_y - (rh - ROW_FLOOR) * SC) / H_PX, ref_w / W_PX, ref_h / H_PX])
ax0.imshow(ref, interpolation="lanczos")
ax0.axis("off")
sim_left = M + ref_w + GAP
ax1 = fig.add_axes([sim_left / W_PX, (floor_y - 0.22 * PXMM) / H_PX, sim_w / W_PX, sim_h / H_PX])
lc = draw_sim(ax1, *cache[best[0]], PXMM, DPI, 26 * PXMM / 786)
chan_mid = sim_left + (0.6 + 0.06) * PXMM                                           # pixel of the middle of the channel (y = 0.6 mm; the axes start at y = -0.06 mm)
cax = fig.add_axes([(chan_mid - 0.25 * sim_w) / W_PX, (M + 25) / H_PX, 0.5 * sim_w / W_PX, 20 / H_PX])
cb = fig.colorbar(lc, cax=cax, orientation="horizontal")
cb.set_label("integration time along the streamline (ms)", color="white")
cb.ax.xaxis.set_tick_params(color="white", labelcolor="white")
fig.text((M + ref_w / 2) / W_PX, (H_PX - 25) / H_PX, "Saliba et al., Fig. 4 (a), 1 ml/min", color="white", ha="center", va="center", fontsize=13)
fig.text((sim_left + sim_w / 2) / W_PX, (H_PX - 28) / H_PX, f"Simulation ({label}, iteration {ITER}): best match, streamlines seeded at x = {best[1]:.2f} mm, followed {L_CELLS * DX:.1f} mm\ncorrelation with Saliba et al., Fig. 4 (a): {best[2]:.2f}",
         color="white", ha="center", va="center", fontsize=11)
fig.savefig(os.path.join(HERE, f"xs1_fig4_compare_iter{ITER:06d}.png"), dpi=DPI, facecolor=BG)
print("done")
