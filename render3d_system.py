"""
3D model and sections of the liquid from a 3D phi file.

usage:  python render3d_system.py <phi3d_iter_XXXXXX.npz | phi3d_final.npz | fields_iter_XXXXXX.npz> <alpha> <out_prefix>
                                  [--closed] [--zcrop 100]

Input files (key 'phi', index order x, y, z):
    phi3d_iter_XXXXXX.npz   snapshot written by the simulation, every 2nd cell, float16
    thin_fields output      same layout, from thin_fields_system.py
    fields_iter_XXXXXX.npz  full final state (ghost layer included); thinned by 2 here
Output:
    <out_prefix>_3d.png        liquid surface (phi = 0.5) in the tilted channel, two views, coloured by height
    <out_prefix>_sections.png  YZ section at mid-x and XZ section at mid-y, with the phi = 0.5 line
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["legend.fancybox"] = False      # square legend frames
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

XN, YN, ZN = 768, 384, 384                               # full domain in cells
D_MM = 1.0
S = D_MM / ZN                                            # mm per cell


def load_phi(path):
    d = np.load(path)
    phi = d["phi"]
    if os.path.basename(path).startswith("fields_iter"):
        phi = phi[1:-1:2, 1:-1:2, 1:-1:2]                # drop ghost layer, thin by 2
    return phi.astype(np.float32)


def main(path, alpha_deg, prefix, closed=False, zcrop=100):
    phi = load_phi(path)
    step = XN // phi.shape[0]                            # cells per thinned cell (2)
    it = "".join(c for c in os.path.basename(path) if c.isdigit())
    a = np.deg2rad(alpha_deg)
    ex = np.array([np.cos(a), -np.sin(a)])               # channel x axis in the (x_w, z_w) world plane, +x points down-slope
    ez = np.array([np.sin(a), np.cos(a)])

    def world(x, y, z):
        x, y, z = np.broadcast_arrays(np.asarray(x, float) * S, np.asarray(y, float) * S, np.asarray(z, float) * S)
        return np.stack([x * ex[0] + z * ez[0], y, x * ex[1] + z * ez[1]], axis=-1)

    # ---- liquid surface -------------------------------------------------------------------------------
    c = phi[::2, ::2, ::2]                               # coarser grid for the 3D mesh (speed)
    sp = step * 2
    verts, faces, _, _ = measure.marching_cubes(np.pad(c, 1, mode="edge"), level=0.5, spacing=(sp, sp, sp))
    verts = verts - sp                                   # undo the padding offset (cells)
    tri = verts[faces]
    zc = tri[:, :, 2].mean(axis=1)
    colors = plt.cm.viridis((zc - zc.min()) / max(np.ptp(zc), 1e-9))
    colors[:, 3] = 0.95

    # camera angles derived from the tilt: the channel axis in the world is (cos a, 0, -sin a), the channel z axis (sin a, 0, cos a)
    el_axis = min(alpha_deg + 8.0, 88.0)                 # looking down the channel axis from the upstream end (88 deg = from straight above)
    az_axis = 168.0 if alpha_deg < 80 else -90.0
    el_wall = float(np.degrees(np.arcsin(np.cos(a))))    # looking onto the channel floor (z = 0 wall) that carries the liquid
    views = [(22, -72, "side view"),
             (el_axis, az_axis, "looking down the channel axis (from the upstream end)"),
             (el_wall, 0.0, "looking onto the floor wall (the liquid film is on it)")]
    fig = plt.figure(figsize=(26, 9))
    for k, (elev, azim, name) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")

        def quad(pts, fc, ec="none", al=0.25, lw=0.8):
            ax.add_collection3d(Poly3DCollection([pts], facecolors=fc, edgecolors=ec, alpha=al, linewidths=lw))

        xwall = "tab:red" if closed else "0.6"           # closed x walls red, periodic ends grey
        quad(world([0, 0, 0, 0], [0, YN, YN, 0], [0, 0, ZN, ZN]), xwall, xwall, 0.12, 1.2)
        quad(world([XN] * 4, [0, YN, YN, 0], [0, 0, ZN, ZN]), xwall, xwall, 0.12, 1.2)
        quad(world([0, XN, XN, 0], [0, 0, YN, YN], [0, 0, 0, 0]), "0.4", "0.2", 0.35, 1.0)
        quad(world([0, XN, XN, 0], [0, 0, 0, 0], [0, 0, ZN, ZN]), "0.8", "0.5", 0.05, 0.8)
        quad(world([0, XN, XN, 0], [YN] * 4, [0, 0, ZN, ZN]), "0.8", "0.5", 0.05, 0.8)
        pc = Poly3DCollection(world(tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]), facecolors=colors, edgecolors="none")
        ax.add_collection3d(pc)
        g0 = world(XN * 1.05, YN / 2, ZN / 2)
        ax.quiver(g0[0], g0[1], g0[2], 0, 0, -0.35, color="k", linewidth=2.5, arrow_length_ratio=0.25)
        ax.text(g0[0], g0[1], g0[2] - 0.40, "g", fontsize=14)
        allp = np.array([world(x, y, z) for x in (0, XN) for y in (0, YN) for z in (0, ZN)])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2] + 0.1)
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(name, fontsize=11)
    fig.suptitle("Liquid surface phi = 0.5, alpha = %g deg, %s x walls, snapshot %s   (colour = height z)"
                 % (alpha_deg, "closed" if closed else "periodic", it), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(prefix + "_3d.png", dpi=120)
    plt.close(fig)
    print("wrote", prefix + "_3d.png")

    # ---- sections -------------------------------------------------------------------------------------
    nz = min(int(zcrop / step), phi.shape[2])
    fig, ax = plt.subplots(2, 1, figsize=(14, 7.5))
    secs = [("XZ section at y = %d (mid-width)" % (YN // 2), phi[:, phi.shape[1] // 2, :nz].T, XN, "x"),
            ("YZ section at x = %d (mid-length)" % (XN // 2), phi[phi.shape[0] // 2, :, :nz].T, YN, "y")]
    for a_, (title, sec, L, lab) in zip(ax, secs):
        im = a_.imshow(sec, origin="lower", aspect="equal", cmap="RdBu_r", vmin=0, vmax=1, extent=(0, L, 0, nz * step))
        a_.contour(np.linspace(0, L, sec.shape[1]), np.linspace(0, nz * step, sec.shape[0]), sec, levels=[0.5], colors="cyan", linewidths=1.3)
        a_.set_title("%s   (phi: red = liquid, blue = gas; z shown 0-%d of %d cells)" % (title, nz * step, ZN), fontsize=10)
        a_.set_xlabel(lab + " [cells]"); a_.set_ylabel("z [cells]")
        fig.colorbar(im, ax=a_, fraction=0.025, pad=0.01, label="phi")
    fig.tight_layout()
    fig.savefig(prefix + "_sections.png", dpi=120)
    plt.close(fig)
    print("wrote", prefix + "_sections.png")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    zc = int(sys.argv[sys.argv.index("--zcrop") + 1]) if "--zcrop" in sys.argv else 100
    if "--zcrop" in sys.argv:
        args.remove(sys.argv[sys.argv.index("--zcrop") + 1])
    main(args[0], float(args[1]), args[2], closed="--closed" in sys.argv, zcrop=zc)
