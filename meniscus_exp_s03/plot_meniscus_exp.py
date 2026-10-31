"""3D view and sections of the meniscus of the experiment-matched run EXP_s0.3 (sigma_lat 0.3, tau_f 0.55, theta 65, W 10, fill 51), iteration 24001.
Data: phi3d_iter_024001.npz (half resolution), plane_yz / plane_xz dumps, surface_height.npz.  x = flow direction (periodic), y = channel width, z = film depth (floor at z = 0)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.fancybox"] = False

DX = 1.2e-3 / 384 * 1e6          # um per cell
DT = (1 / 3) * (0.55 - 0.5) * (1.2e-3 / 384) ** 2 / 1e-6      # s per iteration
IT = 24001

def height(col):
    m = col > 0.5
    if not m.any():
        return np.nan
    k = len(col) - 1 - np.argmax(m[::-1])
    if k >= len(col) - 1:
        return float(k)
    p0, p1 = col[k], col[k + 1]
    return k + (p0 - 0.5) / (p0 - p1) if p0 != p1 else float(k)

# ---------------- 3D ----------------
p3 = np.load("phi3d_iter_024001.npz")["phi"].astype(np.float32)           # [x/2, y/2, z/2]
H = np.array([[height(p3[i, j]) for j in range(p3.shape[1])] for i in range(p3.shape[0])]) * 2 * DX   # um
xs = np.arange(p3.shape[0]) * 2 * DX / 1000.0
ys = np.arange(p3.shape[1]) * 2 * DX / 1000.0
X, Y = np.meshgrid(xs, ys, indexing="ij")
fig = plt.figure(figsize=(9, 5.6))
ax = fig.add_subplot(111, projection="3d")
sf = ax.plot_surface(X, Y, H, cmap="viridis", rstride=4, cstride=1, linewidth=0, antialiased=True)
ax.set_xlabel("x, flow direction (mm)")
ax.set_ylabel("y, channel width (mm)")
ax.set_zlabel("film height (µm)")
ax.set_zlim(0, 300)
ax.set_box_aspect((2.4, 1.2, 0.8))
ax.view_init(elev=28, azim=-62)
fig.colorbar(sf, shrink=0.55, pad=0.08, label="film height (µm)")
ax.set_title("EXP_s0.3  film surface at iteration %d  (%.1f ms)" % (IT, IT * DT * 1e3), fontsize=10)
fig.tight_layout()
fig.savefig("meniscus_3d.png", dpi=170)
plt.close(fig)

# ---------------- sections ----------------
yz = np.load("plane_yz_iter_024001.npz")["phi"]                           # [y, z]
xz = np.load("plane_xz_iter_024001.npz")["phi"]                           # [x, z]
s = np.load("surface_height.npz")["SurfaceHeight"]
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
zmax = 110
a = ax[0, 0]
im = a.imshow(yz[:, :zmax].T, origin="lower", extent=[0, 384 * DX / 1000, 0, zmax * DX], cmap="RdBu", vmin=0, vmax=1, aspect="auto")
yy = np.arange(384) * DX / 1000
hy = np.array([height(yz[j]) for j in range(384)]) * DX
a.plot(yy, hy, "k", lw=1.2)
a.set_title("cross-section across the channel (y–z plane), iteration %d" % IT, fontsize=10)
a.set_xlabel("y, channel width (mm)")
a.set_ylabel("z (µm)")
fig.colorbar(im, ax=a, label="φ (1 = liquid)")
a = ax[0, 1]
for r, c in zip((0, 5, 20, 40, 59), plt.cm.viridis(np.linspace(0.05, 0.9, 5))):
    a.plot(yy, s[r, 1:] * DX, color=c, label="%d it. (%.2f ms)" % (s[r, 0], s[r, 0] * DT * 1e3))
a.set_title("film height across the width at several times", fontsize=10)
a.set_xlabel("y, channel width (mm)")
a.set_ylabel("film height (µm)")
a.grid(alpha=0.3)
a.legend(fontsize=8, frameon=False)
a = ax[1, 0]
im = a.imshow(xz[:, :zmax].T, origin="lower", extent=[0, 768 * DX / 1000, 0, zmax * DX], cmap="RdBu", vmin=0, vmax=1, aspect="auto")
xx = np.arange(768) * DX / 1000
a.plot(xx, np.array([height(xz[i]) for i in range(768)]) * DX, "k", lw=1.0)
a.set_title("section along the flow (x–z plane), iteration %d" % IT, fontsize=10)
a.set_xlabel("x, flow direction (mm)")
a.set_ylabel("z (µm)")
fig.colorbar(im, ax=a, label="φ (1 = liquid)")
a = ax[1, 1]
t = s[:, 0] * DT * 1e3
wall = 0.5 * (s[:, 1] + s[:, -1]) * DX
centre = s[:, 190:195].mean(axis=1) * DX
a.plot(t, wall, label="film height at the wall")
a.plot(t, centre, label="film height at the centre")
a.plot(t, wall - centre, "k--", label="meniscus rise (wall − centre)")
a.set_title("development of the meniscus", fontsize=10)
a.set_xlabel("physical time (ms)")
a.set_ylabel("µm")
a.grid(alpha=0.3)
a.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("meniscus_sections.png", dpi=150)
print("ok; final wall %.0f um, centre %.0f um, rise %.0f um" % (wall[-1], centre[-1], wall[-1] - centre[-1]))
