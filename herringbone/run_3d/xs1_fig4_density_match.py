"""How dense is Saliba's streamline plot (Fig. 4 a), and which seeding of the inlet-plane streamlines gives the same density?
Measured on his image (window = inside his white walls, rows ROW_TOP..ROW_BOT, columns COL_L..COL_R, coloured pixels = saturation > 0.45 and value > 0.45):
  coverage    fraction of the window that is covered by coloured pixels
  width       mean line width in pixels = coloured pixels / skeleton pixels
  occupancy   fraction of the blocks of B px x B px (B = 12, 24, 48 px = 0.034, 0.068, 0.135 mm) that contain any coloured pixel (how evenly the lines fill the window)
  segments    number of separate connected pieces of ink and their median skeleton length (mm)
The same quantities are computed for our streamlines (full 3-D streamlines from the inlet plane x = 1 to the outlet, projected on y-z), drawn at his pixel scale with line width = his line width,
for seed spacings 1, 2, 3, 4, 6 cells (all seeds on the grid of spacing 1; the coarser sets are subsets).
Usage (PC, herringbone\\run_3d):  python xs1_fig4_density_match.py fields3d_W14long/fields3d_iter_520000.npz
"""
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
from scipy import ndimage
from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize

os.environ["TP"] = "4.9"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

path = sys.argv[1]
d = np.load(path)
phi, u = d["phi"].astype(np.float64), d["u"].astype(np.float64)
XN, YN, ZN = phi.shape
DX = 1.2 / YN
Z_FLOOR = 15
TAU_F, NU = 0.6, 1.0e-6
DT_MS = (1 / 3) * (TAU_F - 0.5) * (DX * 1e-3) ** 2 / NU * 1e3
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
liq = (phi >= 0.5) & ~solid
BG = "#10163a"
CMAP = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.16, 0.90, 256)))
PX_MM = (450.1 - 34.7) / 1.2
ROW_FLOOR, ROW_TOP, ROW_BOT, COL_L, COL_R = 193, 62, 241, 37, 458

# ---- his image ----------------------------------------------------------------------------------------------------------------------------------------------
img = mi.imread(os.path.join(HERE, "saliba_fig4a_reference.png"))[:, :, :3].astype(float)
if img.max() > 1.5:
    img /= 255.0
hsv_ = mcol.rgb_to_hsv(img)
his = ((hsv_[:, :, 1] > 0.45) & (hsv_[:, :, 2] > 0.45) & ~(img.min(axis=2) > 0.88))[ROW_TOP:ROW_BOT + 1, COL_L:COL_R + 1]
HPX, WPX = his.shape
Y0, Y1 = (COL_L - 34.7) / PX_MM, (COL_R + 1 - 34.7) / PX_MM
Z1, Z0 = (ROW_FLOOR - (ROW_TOP - 0.5)) / PX_MM, (ROW_FLOOR - (ROW_BOT + 0.5)) / PX_MM


def metrics(m):
    sk = skeletonize(m)
    width = m.sum() / max(1, sk.sum())
    occ = []
    for B in (12, 24, 48):
        h, w = (m.shape[0] // B) * B, (m.shape[1] // B) * B
        blocks = m[:h, :w].reshape(h // B, B, w // B, B).any(axis=(1, 3))
        occ.append(blocks.mean() * 100)
    lab, nseg = ndimage.label(m, structure=np.ones((3, 3)))
    lens = np.array([skeletonize(lab == k).sum() for k in range(1, nseg + 1)]) / PX_MM if nseg < 4000 else np.array([np.nan])
    mc = ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=1)                 # joins gaps of 1-2 px (thin anti-aliased lines break up under the colour threshold)
    labc, nsegc = ndimage.label(mc, structure=np.ones((3, 3)))
    lensc = np.array([skeletonize(labc == k).sum() for k in range(1, nsegc + 1)]) / PX_MM if nsegc < 4000 else np.array([np.nan])
    return dict(cover=m.mean() * 100, width=width, occ=occ, nseg=nseg, medlen=float(np.median(lens)), skmm=sk.sum() / PX_MM / (m.size / PX_MM ** 2),
                nsegc=nsegc, medlenc=float(np.median(lensc)), p90c=float(np.percentile(lensc, 90)))


hm = metrics(his)
print(f"HIS: coverage {hm['cover']:.1f} %, line width {hm['width']:.1f} px, occupancy (12/24/48 px blocks) {hm['occ'][0]:.0f} / {hm['occ'][1]:.0f} / {hm['occ'][2]:.0f} %, "
      f"{hm['nseg']} separate pieces, median piece {hm['medlen']:.3f} mm, skeleton density {hm['skmm']:.0f} mm per mm^2")
print(f"HIS after closing 1-2 px gaps: {hm['nsegc']} pieces, median piece {hm['medlenc']:.3f} mm, 90th percentile {hm['p90c']:.3f} mm")

# ---- our streamlines from the inlet plane (all seeds of the grid with spacing 1) ----------------------------------------------------------------------------
sy, sz = np.meshgrid(np.arange(0, YN), np.arange(0, ZN), indexing="ij")
ok = liq[0][sy, sz]
seed_idx = np.c_[sy[ok], sz[ok]]
pos = np.c_[np.zeros(len(seed_idx)), seed_idx]
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
y_mm, z_mm = (traj[:, :, 1] + 0.5) * DX, (traj[:, :, 2] + 0.5 - Z_FLOOR) * DX
print(f"{n} streamlines integrated from the inlet plane")


def raster(sel, lw_px):
    segs, cols = [], []
    for k in np.where(sel)[0]:
        m = np.isfinite(y_mm[k])
        p_ = np.c_[y_mm[k][m], z_mm[k][m]]
        if len(p_) < 2:
            continue
        segs.append(np.stack([p_[:-1], p_[1:]], axis=1))
        cols.append(tt[k][m][:-1])
    segs, cols = np.concatenate(segs), np.concatenate(cols)
    f = plt.figure(figsize=(WPX / 100, HPX / 100), dpi=100, facecolor=BG)
    a = f.add_axes([0, 0, 1, 1]); a.set_facecolor(BG); a.axis("off")
    o = np.argsort(cols)
    a.add_collection(LineCollection(segs[o], array=cols[o], cmap=CMAP, linewidths=lw_px * 72 / 100, clim=(0, float(np.nanpercentile(cols, 95)))))
    a.set(xlim=(Y0, Y1), ylim=(Z0, Z1))
    f.canvas.draw()
    im = np.asarray(f.canvas.buffer_rgba())[:, :, :3] / 255.0
    plt.close(f)
    h = mcol.rgb_to_hsv(im)
    return (h[:, :, 1] > 0.45) & (h[:, :, 2] > 0.45)


LW = max(1.0, hm["width"])
print(f"ours, drawn at his scale with line width {LW:.1f} px:")
for s in (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
    gy, gz = np.meshgrid(np.round(np.arange(0, YN, s)).astype(int), np.round(np.arange(0, ZN, s)).astype(int), indexing="ij")
    have = {(a, b) for a, b in zip(gy.ravel(), gz.ravel())}
    sel = np.array([(a, b) in have for a, b in seed_idx])
    mm_ = metrics(raster(sel, LW))
    print(f"  spacing {s}: {int(sel.sum()):4d} seeds, coverage {mm_['cover']:5.1f} %, occupancy {mm_['occ'][0]:.0f} / {mm_['occ'][1]:.0f} / {mm_['occ'][2]:.0f} %, "
          f"{mm_['nsegc']} pieces after closing, median {mm_['medlenc']:.3f} mm, p90 {mm_['p90c']:.3f} mm, skeleton density {mm_['skmm']:.0f}")
