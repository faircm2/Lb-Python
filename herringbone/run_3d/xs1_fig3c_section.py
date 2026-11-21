"""Surface-to-volume ratio along one 3 mm section of the channel, in the style of Saliba Fig. 3c (15 points about 0.2 mm apart), for every possible position of the section.
S_v per slab (11 cells = 0.206 mm, 15 slabs = 3.09 mm): film surface area / liquid volume under the surface down to the floor (height-field method of xs1_fig3c.py).
The sections are compared with his numbers (mean 3709 m^2/m^3, relative standard deviation < 5 %); ranking: |mean - 3709| / 3709 + relative std.
Only sections that lie inside a groove train (train 1 x53-261, train 2 x315-523) contain about five chevrons like his; the scan covers all positions.
Usage (PC, herringbone\\run_3d):  python xs1_fig3c_section.py fields3d_W7/fields3d_iter_019222.npz "W=7, job 8866565"
Writes xs1_fig3c_scan_iterNNNNNN.png (mean and relative std against the start of the section), xs1_fig3c_section_iterNNNNNN_xSTART.png for the best sections.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mi
from scipy import ndimage

os.environ["TP"] = "4.9"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "W=7, job 8866565"
ITER = int(path[-10:-4])
phi = np.load(path)["phi"].astype(np.float64)
XN, YN, ZN = phi.shape
DX = 1.2 / YN
SLAB, NSLAB = 11, 15
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
kk = np.arange(ZN)
top_solid = np.where(solid.any(axis=2), (solid * (kk + 1)).max(axis=2), 0).astype(float)
fluid_liq = (phi >= 0.5) & ~solid
has = fluid_liq.any(axis=2)
k_top = np.where(has, ZN - 1 - np.argmax(fluid_liq[:, :, ::-1], axis=2), 0)
ix, iy = np.meshgrid(np.arange(XN), np.arange(YN), indexing="ij")
p0, p1 = phi[ix, iy, k_top], phi[ix, iy, np.minimum(k_top + 1, ZN - 1)]
h = k_top + 0.5 + (p0 - 0.5) / np.where(p0 == p1, 1.0, p0 - p1)
wet = has & (h > top_solid + 0.5)
h = np.where(wet, h, top_solid)
hx, hy = np.gradient(h, axis=0), np.gradient(h, axis=1)
dA = np.where(wet, np.sqrt(1.0 + hx ** 2 + hy ** 2), 0.0)
dV = np.where(wet, h - top_solid, 0.0)
A_x, V_x = dA.sum(axis=1), dV.sum(axis=1)                       # per x layer


def section(start):
    """S_v (m^2/m^3) of the NSLAB slabs of a section starting at cell number start (1-based); x positions in mm from the start"""
    i0 = start - 1
    A = np.array([A_x[i0 + k * SLAB:i0 + (k + 1) * SLAB].sum() for k in range(NSLAB)])
    V = np.array([V_x[i0 + k * SLAB:i0 + (k + 1) * SLAB].sum() for k in range(NSLAB)])
    return (np.arange(NSLAB) + 1) * SLAB * DX, A / V / (DX * 1e-3)


rows = []
for start in range(1, XN - NSLAB * SLAB, 5):
    x, sv = section(start)
    rows.append((start, (start - 1) * DX, sv.mean(), sv.std() / sv.mean() * 100))
rows = np.array(rows)
print(f"{path}: {label}; section = {NSLAB} slabs of {SLAB} cells = {NSLAB * SLAB * DX:.2f} mm; his: mean 3709 m2/m3, relative std < 5 %")
in_train = lambda s: (53 <= s and s + NSLAB * SLAB <= 262) or (315 <= s and s + NSLAB * SLAB <= 524)
cand = [r for r in rows if in_train(r[0])]
rank = sorted(cand, key=lambda r: abs(r[2] - 3709) / 3709 + r[3] / 100)
print("sections inside a groove train, best match to his numbers first:")
for r in rank[:6]:
    print(f"  start x = {int(r[0]):3d} cells ({r[1]:.2f} mm): mean {r[2]:.0f} m2/m3, relative std {r[3]:.1f} %")

fig, axs = plt.subplots(2, 1, figsize=(10, 7), layout="constrained", sharex=True)
axs[0].plot(rows[:, 1], rows[:, 2], "o-", color="#2171b5", ms=3)
axs[0].axhline(3709, color="k", ls="--", lw=1, label="Saliba 3709")
axs[1].plot(rows[:, 1], rows[:, 3], "o-", color="#2171b5", ms=3)
axs[1].axhline(5, color="k", ls="--", lw=1, label="Saliba: < 5 %")
for a in axs:
    for lo, hi in ((53, 261), (315, 523)):
        a.axvspan(lo * DX, (hi - NSLAB * SLAB) * DX, color="#fdd49e", alpha=0.4, lw=0)
    a.grid(alpha=0.3)
    a.legend(loc="upper right")
axs[0].set(ylabel="mean S_v of the 3 mm section (m$^2$/m$^3$)", title=f"{label}, iteration {ITER}: S_v of a 3 mm section against its start\n(shaded: whole section inside a groove train)")
axs[1].set(ylabel="relative std (%)", xlabel="start of the 3 mm section, x (mm)")
fig.savefig(os.path.join(HERE, f"xs1_fig3c_scan_iter{ITER:06d}.png"), dpi=110)

for k, r in enumerate(rank[:2], 1):
    x, sv = section(int(r[0]))
    fig, ax = plt.subplots(figsize=(6.4, 5), layout="constrained")
    ax.plot(x, sv / 1e3, "o", color="k", ms=9)
    ax.set(xlim=(0, 3.2), ylim=(0, 6.6), xlabel="x along the channel in the section [mm]", ylabel="$S_v$ [$10^3$ m$^2$ m$^{-3}$]",
           title=f"{label}, iteration {ITER}\nsection x = {r[1]:.2f}-{r[1] + NSLAB * SLAB * DX:.2f} mm: mean {sv.mean():.0f} m$^2$m$^{{-3}}$, rel. std {sv.std() / sv.mean() * 100:.1f} %")
    ax.set_yticks(range(0, 7))
    fig.savefig(os.path.join(HERE, f"xs1_fig3c_section_iter{ITER:06d}_rank{k}_x{int(r[0])}.png"), dpi=110)
    plt.close(fig)
print("done")


# ---- comparison with Saliba's panel (c): his image next to ours, his 15 dots digitised from the image (frame bottom = 0, mean fixed to his stated 3709) --------------
ref = mi.imread(os.path.join(HERE, "saliba_fig3c_reference.png"))[:, :, :3]
g = ref.mean(axis=2) * (255 if ref.max() <= 1.0 else 1)
dark = g < 170
fr_rows = [i for i, c in enumerate(dark.sum(axis=1)) if c > 150]
bottom = fr_rows[-1]
inner = np.zeros(dark.shape, bool)
cols_ = [i for i, c in enumerate(dark.sum(axis=0)) if c > 150]
inner[fr_rows[0] + 6:bottom - 6, cols_[0] + 6:cols_[-1] - 6] = True
lab, n = ndimage.label((g < 80) & inner)
cent = sorted((ndimage.center_of_mass(lab == i + 1)[::-1] for i in range(n) if (lab == i + 1).sum() > 20))
px = np.array([c[0] for c in cent]); py = np.array([c[1] for c in cent])
x_his = 0.2 + (px - px[0]) / (px[-1] - px[0]) * 2.8                                   # his points are at 0.2 ... 3.0 mm
per_1000 = (bottom - py.mean()) / 3.709                                              # px per 10^3 m2/m3, from his stated mean 3709
sv_his = (bottom - py) / per_1000 * 1e3
print(f"his digitised points: {len(sv_his)}, mean {sv_his.mean():.0f}, relative std {sv_his.std() / sv_his.mean() * 100:.1f} % (he states 3709, < 5 %)")
r = rank[0]
x, sv = section(int(r[0]))
# both plots get the same frame size S (pixels): Saliba's image is enlarged until its frame is as large as the frame of the simulation plot
W_PX, H_PX, S = 1650, 704, 440
FR_L, FR_B, SIM_L = 208, 170, 905                                                     # pixel position of Saliba's frame (left, bottom) and of the simulation frame (left)
sx, sy = S / (cols_[-1] - cols_[0]), S / (bottom - fr_rows[0])                         # enlargement of the image: frame of the image -> S x S pixels
fig = plt.figure(figsize=(W_PX / 110, H_PX / 110), dpi=110)
frame_top = FR_B + S
ax0 = fig.add_axes([(FR_L - cols_[0] * sx) / W_PX, (frame_top + fr_rows[0] * sy - ref.shape[0] * sy) / H_PX, ref.shape[1] * sx / W_PX, ref.shape[0] * sy / H_PX])
ax0.imshow(ref, interpolation="lanczos")
ax0.axis("off")
fig.text((FR_L + S / 2) / W_PX, (H_PX - 26) / H_PX, "Saliba et al., Fig. 3 (c), 1 ml/min", ha="center", va="center", fontsize=12)
ax1 = fig.add_axes([SIM_L / W_PX, FR_B / H_PX, S / W_PX, S / H_PX])
ax1.plot(x, sv / 1e3, "o", color="k", ms=9, label=f"Simulation ({label}, iteration {ITER}, x = {r[1]:.2f}–{r[1] + NSLAB * SLAB * DX:.2f} mm)")
ax1.plot(x_his, sv_his / 1e3, "o", mfc="none", mec="#c0392b", ms=9, mew=1.8, label="Saliba et al. digitised from Fig. 3 (c)")
ax1.set(xlim=(0, 3.2), ylim=(0, 6.5), xlabel="position along the channel [mm]\nSimulation: x-direction;  Saliba et al.: z-direction", ylabel="$S_v$ [$10^3$ m$^2$ m$^{-3}$]",
        title=f"Simulation: mean {sv.mean():.0f} m$^2$m$^{{-3}}$, rel. std {sv.std() / sv.mean() * 100:.1f} %\nSaliba et al.: mean {sv_his.mean():.0f} m$^2$m$^{{-3}}$, rel. std {sv_his.std() / sv_his.mean() * 100:.1f} %")
ax1.set_xticks(range(0, 4)); ax1.set_yticks([0, 2, 4, 6])
ax1.tick_params(labelsize=15)
ax1.xaxis.label.set_size(14); ax1.yaxis.label.set_size(15); ax1.title.set_size(13)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=11, frameon=False)
fig.savefig(os.path.join(HERE, f"xs1_fig3c_compare_iter{ITER:06d}.png"), dpi=110)
print("wrote compare plot")
