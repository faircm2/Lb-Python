"""3D view of the W=4 run (job 8815366): streamlines coloured by speed |u|, translucent free surface (phi = 0.5), grooved floor.
Seeds: sparse y-z grids in the liquid at x = 0.9 mm (before train 1) and x = 5.4 mm (gap before train 2), plus random seeds inside the grooves of both trains. Data: fields3d/fields3d_iter_XXXXXX.npz.
Usage: python xs1_3d.py [iteration]    ->  xs1_3d_iterXXXXXX.png and the ParaView files in paraview/ (see PARAVIEW below)."""
import os, sys
import numpy as np
import pyvista as pv

os.environ["TP"] = "4.9"                      # train period 4.9 mm = 1.0 mm gap; must be set before the import
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 10200
XN, YN, ZN = 736, 64, 64
DX = 1.2 / YN                                                   # cell size (mm)
TAU_F, NU = 0.6, 1.0e-6                                         # job 8815366: --tau_f 0.6 --nu 1e-6 (water)
TO_MM_S = (DX * 1e-3) / ((1 / 3) * (TAU_F - 0.5) * (DX * 1e-3) ** 2 / NU) * 1000   # lattice velocity -> mm/s (dx/dt, eq 25)
Z_FLOOR = 15
FILLS = {"W4": "W=4, sigma 0.1 (job 8815366)"}
SEED_XS = (48, 290)                                             # cell numbers x of the seed planes (0.9 mm and 5.4 mm)
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]    # index = cell number - 1


def build_grid(phi, u):
    """Uniform grid in mm (z = height above the floor); velocity is zero outside the liquid so streamlines stay inside it."""
    liquid = (phi >= 0.5) & ~solid
    grid = pv.ImageData(dimensions=(XN, YN, ZN), spacing=(DX,) * 3, origin=(0.5 * DX, 0.5 * DX, (0.5 - Z_FLOOR) * DX))
    flat = lambda a: a.ravel(order="F")
    f32 = lambda a: flat(a).astype(np.float32)
    grid["phi"] = f32(np.where(solid, 1.0, phi))                # no surface at the solid
    grid["solid"] = f32(solid)
    grid["u"] = np.stack([f32(np.where(liquid, u[k], 0.0)) for k in range(3)], axis=1)     # mm/s, zero outside the liquid
    grid["speed"] = np.linalg.norm(grid["u"], axis=1)
    grid["speed_yz"] = np.hypot(grid["u"][:, 1], grid["u"][:, 2])      # secondary-flow speed
    return grid, liquid


def seed_points(liquid):
    j, k = np.meshgrid(np.arange(3, YN - 2, 7), np.arange(Z_FLOOR - 2, ZN, 4), indexing="ij")
    pts = []
    for xs in SEED_XS:
        ok = liquid[xs - 1, j, k]
        pts.append(np.column_stack([np.full(ok.sum(), (xs - 0.5) * DX), (j[ok] + 0.5) * DX, (k[ok] + 0.5 - Z_FLOOR) * DX]))
    groove = np.argwhere(liquid[:, :, :Z_FLOOR - 1] & ((np.arange(XN)[:, None, None] > 60) & (np.arange(XN)[:, None, None] < 520)))
    cell = groove[np.random.default_rng(0).choice(len(groove), 150, replace=False)]
    pts.append((cell + 0.5) * DX - [0, 0, Z_FLOOR * DX] + [0, 0, 0])
    return pv.PolyData(np.vstack(pts))


data = {}
for fill in FILLS:
    d = np.load(os.path.join(HERE, "fields3d", f"fields3d_iter_{ITER:06d}.npz"))
    data[fill] = build_grid(d["phi"].astype(float), d["u"].astype(float) * TO_MM_S)
streams = {f: g.streamlines_from_source(seed_points(liq), vectors="u", max_length=16, max_steps=40000, integration_direction="forward") for f, (g, liq) in data.items()}
speed_max = max(s["speed"].max() for s in streams.values())

# PARAVIEW: per fill the volume (.vti: phi, solid, u, speed, speed_yz on the 736x64x64 grid, mm), the streamlines (.vtp, carrying the same arrays),
# the free surface (phi = 0.5) and the grooved floor (solid = 0.5) as surfaces. Lengths are in mm, z = height above the smooth floor.
OUT = os.path.join(HERE, "paraview")
os.makedirs(OUT, exist_ok=True)
for fill, (grid, _) in data.items():
    stem = os.path.join(OUT, f"xs1_fill{fill}_iter{ITER:06d}")
    grid.save(stem + "_volume.vti")
    streams[fill].save(stem + "_streamlines.vtp")
    grid.contour([0.5], scalars="phi").save(stem + "_free_surface.vtp")
    grid.contour([0.5], scalars="solid").save(stem + "_floor.vtp")
print("wrote", OUT)

pl = pv.Plotter(shape=(len(FILLS), 1), off_screen=True, window_size=(2200, 560 * len(FILLS)))
for row, (fill, name) in enumerate(FILLS.items()):
    pl.subplot(row, 0)
    grid, _ = data[fill]
    pl.add_mesh(grid.contour([0.5], scalars="solid"), color="#8c8c8c", smooth_shading=True)
    pl.add_mesh(grid.contour([0.5], scalars="phi"), color="#7fb2e5", opacity=0.22, smooth_shading=True)
    pl.add_mesh(streams[fill].tube(radius=0.009), scalars="speed", cmap="turbo", clim=(0, speed_max), show_scalar_bar=(row == 0),
                scalar_bar_args=dict(title="speed |u| (mm/s)", vertical=False, position_x=0.35, position_y=0.02, width=0.3, height=0.06))
    pl.add_text(f"{name}, iteration {ITER}", font_size=11, position="upper_left")
    pl.camera_position = [(6.9, -20.0, 22.0), (6.9, 0.6, 0.05), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 1.9                                # half the visible height (mm); the width follows from the window
pl.background_color = "white"
out = os.path.join(HERE, f"xs1_3d_iter{ITER:06d}.png")
pl.screenshot(out)
print("wrote", out)
