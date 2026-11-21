"""WINDOW TEST (derived from xs1_fig4_regions.py): streamlines of limited length 3 mm from several seed planes inside the groove trains against his ink coverage per band.
Original doc: Where do the simulation streamlines (inlet plane x = 1 -> outlet) differ from Saliba's Fig. 4 (a)?  Measured, per band of height above the floor line and per position across the channel.
  1  free-surface height of the simulation (top-most liquid cell of every column, averaged along x) against the upper edge of the ink in his image
  2  groove depth of the simulation geometry (lowest liquid cell below the floor line) against the depth the ink reaches in his image
  3  ink coverage (at his pixel scale, his line width) per height band: his image, our streamlines for seed spacing 4.2 and 1 cells
Usage (PC, herringbone\\run_3d):  python xs1_fig4_regions.py fields3d_W14long/fields3d_iter_520000.npz
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
from scipy.ndimage import map_coordinates

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
hb = Herringbone()
solid = herringbone_mask(XN, YN, ZN, hb)[1:XN + 1, 1:YN + 1, 1:ZN + 1]
liq = (phi >= 0.5) & ~solid
BG = "#10163a"
CMAP = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.16, 0.90, 256)))
PX_MM = (450.1 - 34.7) / 1.2
ROW_FLOOR, ROW_TOP, ROW_BOT, COL_L, COL_R = 193, 62, 241, 37, 458
zc = (np.arange(ZN) + 0.5 - Z_FLOOR) * DX                                       # height of the cell centres above the floor line (mm)

# ---- 1 and 2: geometry and film of the simulation ------------------------------------------------------------------------------------------------------------
(lambda *a, **k: None)(f"cell size {DX * 1e3:.1f} um, floor line at z = {Z_FLOOR} cells; Herringbone() parameters: {hb}")
anyliq = liq.any(axis=(0, 1))
zlow = np.where(anyliq)[0].min()
(lambda *a, **k: None)(f"lowest liquid cell: z index {zlow} -> {(zlow - Z_FLOOR) * DX:+.3f} mm relative to the floor line (the groove depth the simulation resolves: {(Z_FLOOR - zlow) * DX:.3f} mm)")
top = np.where(liq.any(axis=2), ZN - 1 - np.argmax(liq[:, :, ::-1], axis=2), -1)     # top-most liquid cell per column (x, y)
htop = np.where(top >= 0, (top + 0.5 - Z_FLOOR) * DX, np.nan)
(lambda *a, **k: None)("free-surface height above the floor line (mm), mean / max over x, at y =")
for y in (0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.15):
    j = int(y / DX)
    (lambda *a, **k: None)(f"   {y:4.2f} mm: mean {np.nanmean(htop[:, j]):.3f}, max {np.nanmax(htop[:, j]):.3f}")
(lambda *a, **k: None)(f"highest liquid cell anywhere: {np.nanmax(htop):.3f} mm above the floor line")

img = mi.imread(os.path.join(HERE, "saliba_fig4a_reference.png"))[:, :, :3].astype(float)
if img.max() > 1.5:
    img /= 255.0
hsv_ = mcol.rgb_to_hsv(img)
his = ((hsv_[:, :, 1] > 0.45) & (hsv_[:, :, 2] > 0.45) & ~(img.min(axis=2) > 0.88))[ROW_TOP:ROW_BOT + 1, COL_L:COL_R + 1]
HPX, WPX = his.shape
Y0, Y1 = (COL_L - 34.7) / PX_MM, (COL_R + 1 - 34.7) / PX_MM
Z1, Z0 = (ROW_FLOOR - (ROW_TOP - 0.5)) / PX_MM, (ROW_FLOOR - (ROW_BOT + 0.5)) / PX_MM
rows_z = Z1 - (np.arange(HPX) + 0.5) / PX_MM                                    # height (mm) of every row of his window
cols_y = Y0 + (np.arange(WPX) + 0.5) / PX_MM
(lambda *a, **k: None)(f"his window: y {Y0:.3f}..{Y1:.3f} mm, z {Z0:.3f}..{Z1:.3f} mm")
(lambda *a, **k: None)(f"his ink: highest {rows_z[np.where(his.any(axis=1))[0].min()]:.3f} mm, lowest {rows_z[np.where(his.any(axis=1))[0].max()]:.3f} mm above the floor line")
(lambda *a, **k: None)("upper edge of his ink (99th percentile row of the ink in each y band) and of the lowest ink, mm above the floor line:")
for a, b in ((0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.75), (0.75, 0.9), (0.9, 1.05), (1.05, 1.2)):
    mk = (cols_y >= a) & (cols_y < b)
    zz = rows_z[np.where(his[:, mk])[0]]
    (lambda *a, **k: None)(f"   y {a:4.2f}-{b:4.2f}: ink top {np.percentile(zz, 99.5):.3f} (max {zz.max():.3f}), ink bottom {np.percentile(zz, 0.5):.3f} (min {zz.min():.3f})")



def vel(p):
    return np.stack([map_coordinates(u[k], p.T, order=1, mode="nearest") for k in range(3)], axis=1)


def inside(p):
    i = np.clip(np.round(p).astype(int), 0, [XN - 1, YN - 1, ZN - 1])
    return liq[i[:, 0], i[:, 1], i[:, 2]] & (p[:, 1] >= 0) & (p[:, 1] <= YN - 1) & (p[:, 2] >= 0) & (p[:, 2] <= ZN - 1)


def lines(X0, L_cells, sp):
    """3-D streamlines seeded on the plane x = X0 (1-based cell), grid spacing sp cells, followed L_cells downstream; returns y, z (mm) of the path points"""
    gy, gz = np.meshgrid(np.arange(0, YN, sp), np.arange(0, ZN, sp), indexing="ij")
    iy, iz = np.round(gy).astype(int), np.round(gz).astype(int)
    ok = liq[X0 - 1][iy, iz]
    pos = np.c_[np.full(ok.sum(), X0 - 1.0), iy[ok], iz[ok]].astype(float)
    n = len(pos)
    alive = np.ones(n, bool)
    traj = [pos.copy()]
    for _ in range(int(L_cells / 0.4 * 3)):
        v1 = vel(pos)
        sp1 = np.linalg.norm(v1, axis=1)
        okk = alive & (sp1 > 1e-6)
        mid = pos + 0.5 * 0.4 * v1 / np.maximum(sp1, 1e-30)[:, None]
        v2 = vel(mid)
        sp2 = np.linalg.norm(v2, axis=1)
        new = pos + 0.4 * v2 / np.maximum(sp2, 1e-30)[:, None]
        okk &= (sp2 > 1e-6) & inside(new) & (new[:, 0] <= X0 - 1 + L_cells)
        pos = np.where(okk[:, None], new, pos)
        alive = okk
        traj.append(np.where(okk[:, None], pos, np.nan))
        if not alive.any():
            break
    traj = np.stack(traj, axis=1)
    return (traj[:, :, 1] + 0.5) * DX, (traj[:, :, 2] + 0.5 - Z_FLOOR) * DX


def raster(y_mm, z_mm, lw_px):
    segs = []
    for k in range(len(y_mm)):
        m = np.isfinite(y_mm[k])
        p_ = np.c_[y_mm[k][m], z_mm[k][m]]
        if len(p_) >= 2:
            segs.append(np.stack([p_[:-1], p_[1:]], axis=1))
    segs = np.concatenate(segs)
    f = plt.figure(figsize=(WPX / 100, HPX / 100), dpi=100, facecolor=BG)
    a = f.add_axes([0, 0, 1, 1]); a.set_facecolor(BG); a.axis("off")
    a.add_collection(LineCollection(segs, colors="#20c997", linewidths=lw_px * 72 / 100))
    a.set(xlim=(Y0, Y1), ylim=(Z0, Z1))
    f.canvas.draw()
    im = np.asarray(f.canvas.buffer_rgba())[:, :, :3] / 255.0
    plt.close(f)
    h = mcol.rgb_to_hsv(im)
    return (h[:, :, 1] > 0.45) & (h[:, :, 2] > 0.45)


BANDS = ((-0.14, -0.08), (-0.08, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.38))


def bands(m):
    return np.array([m[(rows_z >= a) & (rows_z < b)].mean() * 100 for a, b in BANDS])


hb_ = bands(his)
print("band (mm above the floor line)               " + "  ".join(f"{a:+.2f}..{b:+.2f}" for a, b in BANDS))
print("his image                                    " + "  ".join(f"{v:11.1f}" for v in hb_))
for L_mm, sp, X0s in ((3.0, 4.2, (55, 70, 85, 100, 320, 335, 350)), (3.0, 2.0, (55, 70, 85, 100, 320, 335, 350)), (3.0, 1.0, (55, 70, 85, 100, 320, 335, 350)), (13.8, 4.2, (1,)), (13.8, 1.0, (1,))):
    Lc = int(round(L_mm / DX)) if L_mm < 13 else XN - 2
    acc = np.array([bands(raster(*lines(X0, Lc, sp), 1.7)) for X0 in X0s])
    print(f"ours, L {L_mm:4.1f} mm, spacing {sp}, {len(X0s)} seed planes  " + "  ".join(f"{v:11.1f}" for v in acc.mean(axis=0)) + "   plane-to-plane std: " + " ".join(f"{v:.1f}" for v in acc.std(axis=0)))
