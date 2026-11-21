"""3D view of gas enclosed in the liquid and of isolated liquid pieces, from a fields3d snapshot (job 8815366, W=4).
gas    = phi < 0.5 and not solid;  liquid = phi >= 0.5 and not solid;  connected components with 6-neighbour connectivity.
 - atmosphere   = gas component that touches the top slice
 - enclosed gas = every other gas component (bubbles, closed cavities)        -> red
 - main liquid  = largest liquid component; other liquid components (drops, slivers)  -> blue
 - colours as in the phi plots: floor grey, free surface light blue (translucent), enclosed gas near-white with black outline (Blues at phi 0),
   isolated liquid dark blue (Blues at phi 1); black sphere = failing cell x132 y28 z15. Its gas component, size and connection to the atmosphere are printed.
Usage (PC, herringbone\\run_3d):  python xs1_gas_3d.py fields3d/fields3d_iter_010200.npz
Writes xs1_gas_3d_iterNNNNNN_full.png, _zoom.png and paraview/xs1_gas_iterNNNNNN_*.vtp.
"""
import os
import sys

import numpy as np
import pyvista as pv
from scipy import ndimage

os.environ["TP"] = "4.9"                                      # must be set before the import
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

path = sys.argv[1]
ITER = int(path[-10:-4])
phi = np.load(path)["phi"].astype(np.float32)
XN, YN, ZN = phi.shape
DX = 1.2 / YN
Z_FLOOR = 15
SITE = (132, 28, 15)                                           # failing cell, array index of the fields3d arrays (= trace coordinates)
site_idx = SITE
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
print(f"{path}: phi {phi.shape}, solid cells {int(solid.sum())}")

gas = (phi < 0.5) & ~solid
liq = (phi >= 0.5) & ~solid
glab, gn = ndimage.label(gas)
llab, ln = ndimage.label(liq)
atm = np.unique(glab[:, :, -1])
atm = atm[atm > 0]
gsize = np.bincount(glab.ravel(), minlength=gn + 1)
enclosed = [i for i in range(1, gn + 1) if i not in atm]
main_liq = int(np.argmax(np.bincount(llab.ravel())[1:])) + 1
lsize = np.bincount(llab.ravel(), minlength=ln + 1)
others = [i for i in range(1, ln + 1) if i != main_liq]
print(f"gas components {gn}: atmosphere labels {list(atm)} (sizes {[int(gsize[a]) for a in atm]}), enclosed gas components {len(enclosed)}")
print(f"liquid components {ln}: main {int(lsize[main_liq])} cells, other liquid pieces {len(others)}")
g_site = int(glab[site_idx])
print(f"failing cell {SITE}: phi {phi[site_idx]:.3f}, gas component {g_site}" +
      (f" = {'atmosphere' if g_site in atm else 'ENCLOSED'}, size {int(gsize[g_site])} cells" if g_site else " (cell is liquid)"))


def describe(lab, ids, size, name, phi_ref, top=12):
    rows = []
    for i in ids:
        sl = ndimage.find_objects((lab == i).astype(np.uint8))[0]
        m = lab[sl] == i
        c = [int(round(s.start + (s.stop - s.start - 1) / 2)) + 1 for s in sl]
        rows.append((int(size[i]), c, float(phi_ref[sl][m].min()), float(phi_ref[sl][m].max())))
    rows.sort(key=lambda r: -r[0])
    print(f"\n{name}: {len(rows)} pieces, largest {top}  (cells, centre x y z, min phi, max phi)")
    for r in rows[:top]:
        print(f"  {r[0]:7d}  x{r[1][0]} y{r[1][1]} z{r[1][2]}   {r[2]:.3f} {r[3]:.3f}")


describe(glab, enclosed, gsize, "enclosed gas", phi)
describe(llab, others, lsize, "isolated liquid", phi)

# ---------------- meshes --------------------------------------------------------------------------------------------------
os.makedirs("paraview", exist_ok=True)


def grid_of(field, sub=None):
    g = pv.ImageData(dimensions=field.shape, spacing=(DX,) * 3, origin=(0.5 * DX, 0.5 * DX, (0.5 - Z_FLOOR) * DX))
    g.point_data["v"] = field.ravel(order="F").astype(np.float32)
    return g


def surf(mask, smooth=True):
    """isosurface of a boolean mask (0.5 level of its smoothed indicator)"""
    if not mask.any():
        return None
    f = ndimage.gaussian_filter(mask.astype(np.float32), 0.8) if smooth else mask.astype(np.float32)
    s = grid_of(f).contour([0.5], scalars="v")
    return s if s.n_points else None


phi_s = np.where(solid, 1.0, phi)
free_surface = grid_of(phi_s).contour([0.5], scalars="v")
floor = surf(solid, smooth=False)
enc_mask = np.isin(glab, enclosed) if enclosed else np.zeros_like(gas)
oth_mask = np.isin(llab, others) if others else np.zeros_like(liq)
site_mask = (glab == g_site) if g_site else np.zeros_like(gas)
meshes = {"free_surface": free_surface, "floor": floor, "enclosed_gas": surf(enc_mask), "isolated_liquid": surf(oth_mask), "site_gas": surf(site_mask)}
for k, m in meshes.items():
    if m is not None:
        m.save(f"paraview/xs1_gas_iter{ITER:06d}_{k}.vtp")


BLUE = "#1f4fd8"


def flow_arrow(pl, start, length, label, label_offset, size=30):
    """blue arrow along +x (flow direction) with a text label"""
    arrow = pv.Arrow(start=start, direction=(1, 0, 0), scale=length, tip_length=0.22, tip_radius=0.07, shaft_radius=0.025)
    pl.add_mesh(arrow, color=BLUE, smooth_shading=True)
    pl.add_point_labels([(start[0] + length / 2 + label_offset[0], start[1] + label_offset[1], start[2] + label_offset[2])], [label], font_size=size,
                        text_color=BLUE, shape=None, show_points=False, always_visible=True, bold=True)


def render(clip=None, size=(1800, 700)):
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    # colours of the phi plots: Blues (phi 0 = near white gas, phi 1 = dark blue liquid), Greys for the solid, black outline = phi 0.5 contour,
    # free surface light blue and translucent as in xs1_3d.py
    for name, color, op, outline in (("floor", "#8c8c8c", 1.0, False), ("free_surface", "#7fb2e5", 0.22, False),
                                     ("isolated_liquid", "#08306b", 1.0, True), ("enclosed_gas", "#f7fbff", 1.0, True)):
        m = meshes[name]
        if m is None:
            continue
        if clip is not None:
            m = m.clip_box(clip, invert=False)
            if not m.n_points:
                continue
        pl.add_mesh(m, color=color, opacity=op, smooth_shading=True)
        if outline:
            pl.add_silhouette(m, color="black", line_width=2)
    pl.add_mesh(pv.Sphere(radius=1.5 * DX, center=((SITE[0] + 0.5) * DX, (SITE[1] + 0.5) * DX, (SITE[2] + 0.5 - Z_FLOOR) * DX)), color="black")   # failing cell
    return pl


x0, x1, y0, y1, z0, z1 = 100, 180, 10, 50, 0, 40               # zoom region (cells)
clip = [(x0 - 0.5) * DX, (x1 - 0.5) * DX, (y0 - 0.5) * DX, (y1 - 0.5) * DX, (z0 - 0.5 - Z_FLOOR) * DX, (z1 - 0.5 - Z_FLOOR) * DX]

# whole channel: horizontal, flow to the right (+x), looked at from slightly above, parallel projection
pl = render(size=(2200, 460))
pl.add_mesh(pv.Box(bounds=clip), style="wireframe", color="black", line_width=3)                  # region of the zoom plot
flow_arrow(pl, (9.6, 0.6, 0.6), 2.6, "flow", (0, 0.0, 0.55), size=34)
pl.camera_position = [(6.9, -6.0, 14.0), (6.9, 0.6, 0.05), (0, 0, 1)]
pl.camera.parallel_projection = True
pl.camera.parallel_scale = 1.45
pl.screenshot(f"xs1_gas_3d_iter{ITER:06d}_full.png")
pl.close()

# zoom around the failing cell
pl = render(clip=clip, size=(1400, 1000))
flow_arrow(pl, (2.7, 0.35, 0.38), 0.5, "flow", (0, 0.0, 0.12), size=24)
cx, cy, cz = (SITE[0] + 0.5) * DX, (SITE[1] + 0.5) * DX, (SITE[2] + 0.5 - Z_FLOOR) * DX
pl.camera_position = [(cx + 1.0, cy - 1.4, cz + 1.2), (cx, cy, cz), (0, 0, 1)]
pl.screenshot(f"xs1_gas_3d_iter{ITER:06d}_zoom.png")
pl.close()

# collage: horizontal channel at the bottom, zoom plot above it on the right
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
full = plt.imread(f"xs1_gas_3d_iter{ITER:06d}_full.png")
zoom = plt.imread(f"xs1_gas_3d_iter{ITER:06d}_zoom.png")
fig = plt.figure(figsize=(22, 11), dpi=100)
ax = fig.add_axes([0.0, 0.01, 1.0, 0.418]); ax.imshow(full); ax.axis("off")
ins = fig.add_axes([0.62, 0.45, 0.36, 0.50]); ins.imshow(zoom); ins.set_xticks([]); ins.set_yticks([])
for s in ins.spines.values():
    s.set_linewidth(3)
ins.set_title("zoom at the failing cell (black sphere), box x100-180 y10-50 z0-40", fontsize=13, loc="left")
fig.text(0.02, 0.955, f"W=4, sigma 0.1 (job 8815366), iteration {ITER}: gas pockets in the grooves", fontsize=20, ha="left")
handles = [Patch(fc="#8c8c8c", ec="k", label="floor"), Patch(fc="#7fb2e5", ec="k", label="free surface (phi = 0.5)"),
           Patch(fc="#f7fbff", ec="k", label="enclosed gas"), Line2D([], [], marker="o", color="k", ls="", markersize=10, label="failing cell x132 y28 z15"),
           Line2D([], [], color=BLUE, lw=5, label="flow direction")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.88), fontsize=15, frameon=True)
fig.savefig(f"xs1_gas_3d_iter{ITER:06d}_collage.png")
print(f"wrote xs1_gas_3d_iter{ITER:06d}_full.png, _zoom.png, _collage.png and paraview/xs1_gas_iter{ITER:06d}_*.vtp")
