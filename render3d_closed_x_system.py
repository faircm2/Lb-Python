"""
3D view of a closed-x run: the channel box (Xn x Yn x Zn = 768 x 384 x 384 cells, tilted by alpha so that gravity points down),
the real phi field of the x = Xn/2 plane from the last plane dump, and the liquid surface of that plane extruded along x.
Only the mid-x plane is dumped, so the x dependence of the liquid (e.g. pile-up at the closed end) is NOT shown.

usage:  python render3d_closed_x_system.py <run_folder> <alpha> <out.png> [iter]
        run_folder holds plane_yz_iter_XXXXXX.npz (e.g. study/closed_x); iter defaults to the last dump
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["legend.fancybox"] = False      # square legend frames
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

XN, YN, ZN = 768, 384, 384
D_MM = 1.0
S = D_MM / ZN                                            # mm per cell


def main(folder, alpha_deg, out, it=None):
    a = np.deg2rad(alpha_deg)
    ex = np.array([np.cos(a), -np.sin(a)])               # channel x axis in the (x_w, z_w) world plane, +x points down-slope
    ez = np.array([np.sin(a), np.cos(a)])

    def world(x, y, z):
        x, y, z = np.broadcast_arrays(np.asarray(x, float) * S, np.asarray(y, float) * S, np.asarray(z, float) * S)
        return np.stack([x * ex[0] + z * ez[0], y, x * ex[1] + z * ez[1]], axis=-1)

    files = sorted(glob.glob(os.path.join(folder, "plane_yz_iter_*.npz")))
    f = files[-1] if it is None else os.path.join(folder, "plane_yz_iter_%06d.npz" % int(it))
    n_it = int(re.search(r"iter_(\d+)", f).group(1))
    phi = np.load(f)["phi"]
    h = phi.sum(axis=1)                                  # liquid height per y [cells]

    fig = plt.figure(figsize=(20, 9))
    views = [(22, -72, "side view"), (alpha_deg + 10, 168, "looking down the channel axis (+x direction)")]
    for k, (elev, azim, vname) in enumerate(views):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        draw(ax, world, phi, h, elev, azim, vname)
    fig.suptitle("Closed-x run: channel %d x %d x %d cells, alpha = %g deg, iteration %d   |   "
                 "red = closed x walls (x = 0, x = %d), blue = phi at x = %d (real data), cyan = that profile extruded along x"
                 % (XN, YN, ZN, alpha_deg, n_it, XN, XN // 2), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


def draw(ax, world, phi, h, elev, azim, vname):
    def quad(pts, fc, ec="none", al=0.25, lw=0.5):
        ax.add_collection3d(Poly3DCollection([pts], facecolors=fc, edgecolors=ec, alpha=al, linewidths=lw))

    # walls: x = 0 / x = XN (closed), floor z = 0 (grey), y = 0 / y = YN (light), top z = ZN (light)
    quad(world([0, 0, 0, 0], [0, YN, YN, 0], [0, 0, ZN, ZN]), "tab:red", "tab:red", 0.12, 1.2)
    quad(world([XN] * 4, [0, YN, YN, 0], [0, 0, ZN, ZN]), "tab:red", "tab:red", 0.12, 1.2)
    quad(world([0, XN, XN, 0], [0, 0, YN, YN], [0, 0, 0, 0]), "0.4", "0.2", 0.35, 1.0)
    quad(world([0, XN, XN, 0], [0, 0, 0, 0], [0, 0, ZN, ZN]), "0.8", "0.5", 0.05, 0.8)
    quad(world([0, XN, XN, 0], [YN] * 4, [0, 0, ZN, ZN]), "0.8", "0.5", 0.05, 0.8)
    quad(world([0, XN, XN, 0], [0, 0, YN, YN], [ZN] * 4), "0.8", "0.5", 0.03, 0.8)

    # liquid extruded from the mid-x plane: top surface z = h(y), end faces at x = 0 and x = XN
    ys = np.arange(0, YN, 4)
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        h0, h1 = h[y0], h[y1]
        quad(world([0, XN, XN, 0], [y0, y0, y1, y1], [h0, h0, h1, h1]), "tab:cyan", "none", 0.45)
    for xe in (0, XN):
        poly = [world(xe, 0, 0)] + [world(xe, y, h[y]) for y in range(0, YN, 4)] + [world(xe, YN - 1, h[YN - 1]), world(xe, YN, 0)]
        quad(np.array(poly), "tab:blue", "tab:blue", 0.30, 0.6)

    # the real phi field at x = XN/2
    st = 4
    yy, zz = np.meshgrid(np.arange(0, YN, st), np.arange(0, ZN, st), indexing="ij")
    P = world(XN / 2, yy, zz)
    ph = phi[::st, ::st]
    rgba = np.zeros(ph.shape + (4,))
    rgba[..., 2] = 0.8
    rgba[..., 0] = 0.1 + 0.3 * (1 - ph)
    rgba[..., 3] = np.clip(ph, 0, 1) * 0.9
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], facecolors=rgba, shade=False, rstride=1, cstride=1, linewidth=0, antialiased=False)

    # gravity arrow
    c = world(XN * 1.05, YN / 2, ZN / 2)
    ax.quiver(c[0], c[1], c[2], 0, 0, -0.35, color="k", linewidth=2.5, arrow_length_ratio=0.25)
    ax.text(c[0], c[1], c[2] - 0.40, "g", fontsize=14)

    allp = np.array([world(x, y, z) for x in (0, XN) for y in (0, YN) for z in (0, ZN)])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2] + 0.1)
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(vname, fontsize=11)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
