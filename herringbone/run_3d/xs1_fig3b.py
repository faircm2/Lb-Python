"""Fig. 3b of Saliba et al. for the herringbone simulation: the channel is split along its length into two halves.
 left half   the liquid film (phi >= 0.5 above the grooved floor), coloured by the height z above the smooth floor (the paper colours by its x = channel height)
 right half  the three-component velocity field as arrows, coloured by |u| (same colormap as the paper: turbo), moved aside in y
Window of the channel: cells X0..X1 along x (default x100-260 = 3.0 mm, inside train 1).  Flow and gravity act along +x.
Usage (PC, herringbone\\run_3d):  python xs1_fig3b.py fields3d/fields3d_iter_010200.npz [label] [X0 X1]
Writes xs1_fig3b_iterNNNNNN.png and paraview/xs1_fig3b_iterNNNNNN_{film,vectors}.vtp.
"""
import os
import sys

import numpy as np
import pyvista as pv

os.environ["TP"] = "4.9"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "W=4, job 8815366"
X0, X1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (100, 260)      # cell numbers, 1-based, inclusive
ITER = int(path[-10:-4])
d = np.load(path)
phi, u = d["phi"].astype(np.float32), d["u"].astype(np.float32)
XN, YN, ZN = phi.shape
DX = 1.2 / YN                                                  # mm
Z_FLOOR = 15                                                   # smooth floor top = z 0
TAU_F, NU = 0.6, 1.0e-6
U_UNIT = (DX * 1e-3) / ((1 / 3) * (TAU_F - 0.5) * (DX * 1e-3) ** 2 / NU) * 100             # lattice velocity -> cm/s
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
sl = slice(X0 - 1, X1)                                         # array index = cell number - 1
liq = (phi >= 0.5) & ~solid
speed = np.sqrt((u ** 2).sum(axis=0)) * U_UNIT                 # cm/s

# volume flow rate through the cross sections of the window (check against the experiment: 1 ml/min)
Q = (u[0] * np.where(liq, 1.0, 0.0)).sum(axis=(1, 2)) * U_UNIT * 1e-2 * (DX * 1e-3) ** 2 * 6e7          # m/s * m^2 -> m^3/s -> ml/min
print(f"{path}: {label}; window x{X0}-{X1} = {(X1 - X0 + 1) * DX:.2f} mm; lattice -> cm/s factor {U_UNIT:.1f}")
print(f"volume flow rate in the window: mean {Q[sl].mean():.2f} ml/min (experiment 1 ml/min); max |u| in the liquid of the window {speed[sl][liq[sl]].max():.2f} cm/s (paper scale 0.1-8)")

# ---------------- left half of the channel: the film above the smooth floor (its free surface; the groove liquid below z = 0 is left out, as in the paper) ----------
YMID = 0.5 * YN * DX                                           # mm, centre plane of the channel
field = np.pad(np.where(solid, 0.0, phi)[sl, :, Z_FLOOR:], 1)  # only z >= 0; zero padding closes the body at the window ends, the walls and the floor plane
img = pv.ImageData(dimensions=field.shape, spacing=(DX,) * 3, origin=((X0 - 1 - 0.5) * DX, (-0.5) * DX, (-0.5) * DX))
img.point_data["phi"] = field.ravel(order="F")
body = img.contour([0.5], scalars="phi")
body = body.clip_closed_surface(normal=(0, -1, 0), origin=(0, YMID, 0))                     # left half of the channel: keeps y < YMID (bounds printed below)
print("film body bounds (mm):", [round(b, 3) for b in body.bounds])
body["z (mm)"] = body.points[:, 2]
os.makedirs(os.path.join(HERE, "paraview"), exist_ok=True)
body.save(os.path.join(HERE, "paraview", f"xs1_fig3b_iter{ITER:06d}_film.vtp"))

# ---------------- right half of the channel: the velocity vectors, directly joined to the film half (y >= YMID, no gap) ----------------------------------------------------------------------
SX, SY, SZ = 4, 3, 3                                           # subsampling in cells
ii, jj, kk = np.meshgrid(np.arange(X0 - 1, X1, SX), np.arange(YN // 2, YN, SY), np.arange(ZN), indexing="ij")
m = liq[ii, jj, kk]
ii, jj, kk = ii[m], jj[m], kk[m]
GAP = 0.0                                                      # mm: the two halves are joined (set > 0 to pull them apart)
pts = np.c_[(ii + 0.5) * DX, (jj + 0.5) * DX + GAP, (kk + 0.5 - Z_FLOOR) * DX]
vec = np.c_[u[0][ii, jj, kk], u[1][ii, jj, kk], u[2][ii, jj, kk]] * U_UNIT
cloud = pv.PolyData(pts)
cloud["vec"] = vec
cloud["|u| (cm/s)"] = np.linalg.norm(vec, axis=1)
cloud["len"] = 0.3 + 0.7 * cloud["|u| (cm/s)"] / cloud["|u| (cm/s)"].max()                  # arrow length 30 % ... 100 % with the speed
arrows = cloud.glyph(orient="vec", scale="len", factor=0.08, geom=pv.Arrow(tip_length=0.45, tip_radius=0.18, shaft_radius=0.06))
arrows.save(os.path.join(HERE, "paraview", f"xs1_fig3b_iter{ITER:06d}_vectors.vtp"))
umax = float(cloud["|u| (cm/s)"].max())
print(f"vectors: {cloud.n_points} arrows, |u| {cloud['|u| (cm/s)'].min():.2f}..{umax:.2f} cm/s")

# ---------------- render ------------------------------------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(2400, 1100))
pl.set_background("white")
pl.add_mesh(body, scalars="z (mm)", cmap="turbo", smooth_shading=True,
            scalar_bar_args=dict(title="film: height z above the smooth floor (mm)", vertical=False, position_x=0.05, position_y=0.90, width=0.35, height=0.05,
                                 title_font_size=26, label_font_size=22, color="black", fmt="%.2f"))
pl.add_mesh(arrows, scalars="|u| (cm/s)", cmap="turbo", clim=(0.1, umax),
            scalar_bar_args=dict(title="velocity magnitude (cm/s)", vertical=False, position_x=0.58, position_y=0.90, width=0.35, height=0.05,
                                 title_font_size=26, label_font_size=22, color="black", fmt="%.1f"))
xc = 0.5 * (X0 + X1 - 1) * DX
# axes as in the paper's figure, in our naming: x along the channel (flow, g), z = channel height; g in red
O = np.array([X0 * DX + 0.05, -0.55, -0.1])
for vec_, length, color, lab, off in (((0, 0, 1), 0.55, "black", "z", (0, 0, 0.1)), ((1, 0, 0), 0.9, "black", "x", (0.1, 0, 0)), ((1, 0, 0), 0.9, "#d00000", "g", (0.1, 0, 0))):
    start = O + (np.array([0, -0.35, 0]) if color != "black" else 0)
    pl.add_mesh(pv.Arrow(start=tuple(start), direction=vec_, scale=length, tip_length=0.25, tip_radius=0.06, shaft_radius=0.022), color=color)
    pl.add_point_labels([tuple(start + np.array(vec_) * length + off)], [lab], font_size=34, text_color=color, shape=None, show_points=False, always_visible=True, bold=True)
C = np.array([xc, 0.6 + GAP / 2, 0.1])
EL, AZ, R = np.radians(24), np.radians(48), 7.0                    # camera elevation / azimuth: +x (flow) points to the lower right, +y to the upper right, z up (as in the paper)
pl.camera_position = [tuple(C + R * np.array([np.cos(EL) * np.sin(AZ), -np.cos(EL) * np.cos(AZ), np.sin(EL)])), tuple(C), (0, 0, 1)]
pl.camera.view_angle = 20
pl.camera.zoom(1.12)
out = os.path.join(HERE, f"xs1_fig3b_iter{ITER:06d}.png")
pl.screenshot(out)
pl.close()
print("wrote", out)
