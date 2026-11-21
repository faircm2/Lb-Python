"""Surface-to-volume ratio of the film along the channel (Saliba et al., Fig. 3c), from a fields3d snapshot.
Per vertical slab of the channel (thickness SLAB cells in x, the direction along the channel = the paper's z):
  h(x, y)   film surface height: top-most crossing of phi = 0.5 in every column (linear interpolation between the two cells)
  z_f(x, y) floor height of the column = top face of its highest solid cell (the herringbone grooves make it irregular)
  V = sum over the columns of the slab of (h - z_f) dx dy              volume of fluid under the surface, down to the floor (gas pockets in grooves included)
  A = sum over the columns of sqrt(1 + h_x^2 + h_y^2) dx dy              area of the film surface (dry columns, h <= z_f, excluded)
  S_v = A / V
Cross-check (printed): A2 = sum |grad phi| dV over the fluid cells next to the atmosphere-connected gas (diffuse-interface area).
Usage (PC, herringbone\\run_3d):  python xs1_fig3c.py fields3d/fields3d_iter_010200.npz [label]     ->  xs1_fig3c_iterNNNNNN.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

os.environ["TP"] = "4.9"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "W=4, job 8815366"
ITER = int(path[-10:-4])
phi = np.load(path)["phi"].astype(np.float64)
XN, YN, ZN = phi.shape
DX = 1.2 / YN                                              # mm
SLAB = 10                                                  # cells per slab = 0.1875 mm
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]

# ---- floor height per column (top face of the highest solid cell, in cell units from z = 0) --------------------------
kk = np.arange(ZN)
top_solid = np.where(solid.any(axis=2), (solid * (kk + 1)).max(axis=2), 0).astype(float)        # (XN, YN)

# ---- film surface height per column ----------------------------------------------------------------------------------
fluid_liq = (phi >= 0.5) & ~solid
has = fluid_liq.any(axis=2)
k_top = np.where(has, ZN - 1 - np.argmax(fluid_liq[:, :, ::-1], axis=2), 0)                    # top-most liquid cell
ix, iy = np.meshgrid(np.arange(XN), np.arange(YN), indexing="ij")
p0 = phi[ix, iy, k_top]
p1 = phi[ix, iy, np.minimum(k_top + 1, ZN - 1)]
h = k_top + 0.5 + (p0 - 0.5) / np.where(p0 == p1, 1.0, p0 - p1)                                 # cell units, z = k + 0.5 is the centre of cell k
wet = has & (h > top_solid + 0.5)
h = np.where(wet, h, top_solid)

# ---- height-field area and volume (all lengths in cells, converted at the end) ---------------------------------------
hx, hy = np.gradient(h, axis=0), np.gradient(h, axis=1)
dA = np.where(wet, np.sqrt(1.0 + hx ** 2 + hy ** 2), 0.0)                                       # cells^2
dV = np.where(wet, h - top_solid, 0.0)                                                          # cells^3

# ---- cross-check: diffuse-interface area of the atmosphere-connected gas ---------------------------------------------
gas = (phi < 0.5) & ~solid
glab, gn = ndimage.label(gas)
atm_ids = np.unique(glab[:, :, -1]); atm_ids = atm_ids[atm_ids > 0]
atm = np.isin(glab, atm_ids)
near = ndimage.binary_dilation(atm, iterations=6) & ~solid
nearest = ndimage.distance_transform_edt(solid, return_distances=False, return_indices=True)    # index of the nearest fluid cell (itself for fluid cells)
phi_f = phi[nearest[0], nearest[1], nearest[2]]                                                  # solid cells take the phi of the nearest fluid cell
gx, gy, gz = np.gradient(phi_f)
grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
dA2 = np.where(near, grad, 0.0).sum(axis=(1, 2)) / 1.0                                          # per x layer, cells^2

n = XN // SLAB
xs = (np.arange(n) * SLAB + SLAB / 2) * DX                                                       # mm, slab centres
A = np.array([dA[i * SLAB:(i + 1) * SLAB].sum() for i in range(n)])
V = np.array([dV[i * SLAB:(i + 1) * SLAB].sum() for i in range(n)])
A2 = np.array([dA2[i * SLAB:(i + 1) * SLAB].sum() for i in range(n)])
Sv = A / V / (DX * 1e-3)                                                                         # (cells^2 / cells^3) / cell size -> 1/m
Sv2 = A2 / V / (DX * 1e-3)

REGIONS = {"train 1": (53, 261), "train 2": (315, 523), "smooth run-out": (600, 700), "3 mm window (x100-260)": (100, 260)}
print(f"{path}: {label}; slab {SLAB} cells = {SLAB * DX:.4f} mm, {n} slabs; film height mean {np.mean(h[wet] - top_solid[wet]) * DX:.3f} mm over the floor")
print(f"{'region':26s} {'S_v mean (m2/m3)':>18s} {'std':>8s} {'rel. std':>9s}   cross-check S_v (grad phi)")
for name, (a, b) in REGIONS.items():
    m = (xs / DX >= a) & (xs / DX < b)
    print(f"{name:26s} {Sv[m].mean():18.0f} {Sv[m].std():8.0f} {Sv[m].std() / Sv[m].mean() * 100:8.1f}%   {Sv2[m].mean():8.0f}")
m = np.isfinite(Sv)
print(f"{'whole channel':26s} {Sv[m].mean():18.0f} {Sv[m].std():8.0f} {Sv[m].std() / Sv[m].mean() * 100:8.1f}%   {Sv2[m].mean():8.0f}")

fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
for j, (name, (a, b)) in enumerate(list(REGIONS.items())[:2]):
    ax.axvspan(a * DX, b * DX, color="#fdd49e", alpha=0.35, lw=0, label="grooved trains" if j == 0 else None)
ax.plot(xs, Sv, "o", color="k", ms=5, label="S_v, height-field method")
ax.set(xlabel="x along the channel (mm)", ylabel="S_v  (m$^2$ m$^{-3}$)", ylim=(0, 7000), xlim=(0, XN * DX),
       title=f"Surface-to-volume ratio along the channel, {label}, iteration {ITER}\n"
             f"3 mm window x100-260: mean {Sv[(xs / DX >= 100) & (xs / DX < 260)].mean():.0f} m$^2$/m$^3$, relative std {Sv[(xs / DX >= 100) & (xs / DX < 260)].std() / Sv[(xs / DX >= 100) & (xs / DX < 260)].mean() * 100:.1f} %")
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower center", ncol=3)
out = os.path.join(HERE, f"xs1_fig3c_iter{ITER:06d}.png")
fig.savefig(out, dpi=110)
print("wrote", out)
