"""
Cross-section, long-section and 3D plots of a Cap07 run (periodic x, meniscus at the two y walls).

usage:  python plot_cap07_meniscus.py <run folder> [iteration]       (default: last phi3d iteration)
writes: <run folder>/cross_section_yz.png        phi in the YZ plane + h(y) over the iterations (x = Xn/2)
        <run folder>/long_section_xz.png         phi in the XZ plane (y = Yn/2) + h(x) at several y (3D dump)
        <run folder>/meniscus_3d_iter<N>.png     phi = 0.5 surface from the half-resolution 3D dump
Liquid is phi = 1 (below), gas phi = 0 (above).  Dump arrays are [x, y, z] / [y, z] / [x, z].
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def height(phi, axis=-1):
    """z of phi = 0.5 along the last axis (highest crossing from liquid to gas), linear interpolation."""
    phi = np.asarray(phi, dtype=np.float64)
    lo = phi[..., :-1] >= 0.5
    hi = phi[..., 1:] < 0.5
    c = lo & hi
    k = c.shape[-1] - 1 - np.argmax(c[..., ::-1], axis=-1)
    a = np.take_along_axis(phi, k[..., None], axis=-1)[..., 0]
    b = np.take_along_axis(phi, k[..., None] + 1, axis=-1)[..., 0]
    return k + (a - 0.5) / (a - b)


def its_of(files, prefix):
    return [int(os.path.basename(f)[len(prefix):len(prefix) + 6]) for f in files]


def main():
    run = sys.argv[1]
    f3 = sorted(glob.glob(os.path.join(run, "phi3d_iter_*.npz")))
    it = int(sys.argv[2]) if len(sys.argv) > 2 else its_of(f3, "phi3d_iter_")[-1]

    # ---------------- cross-section YZ ----------------
    fyz = sorted(glob.glob(os.path.join(run, "plane_yz_iter_[0-9]*.npz")))
    iyz = its_of(fyz, "plane_yz_iter_")
    phi_yz = np.load(os.path.join(run, "plane_yz_iter_%06d.npz" % it))["phi"]
    ny, nz = phi_yz.shape
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), facecolor="white")
    im = axs[0].imshow(phi_yz.T, origin="lower", cmap="RdBu", vmin=0, vmax=1, extent=(0, ny, 0, nz), aspect="equal")
    axs[0].contour(np.arange(ny) + 0.5, np.arange(nz) + 0.5, phi_yz.T, levels=[0.5], colors="k", linewidths=0.6)
    axs[0].set_title("phi in the YZ plane (x = Xn/2), iteration %d" % it, fontsize=10)
    axs[0].set_xlabel("y-index")
    axs[0].set_ylabel("z-index")
    axs[0].set_box_aspect(1)
    fig.colorbar(im, ax=axs[0], shrink=0.7, label="phi")
    sel = np.unique(np.linspace(0, len(fyz) - 1, 6).astype(int))
    for i in sel:
        axs[1].plot(height(np.load(fyz[i])["phi"]), label="iter %d" % iyz[i])
    axs[1].set_xlabel("y-index")
    axs[1].set_ylabel("surface height h(y) [cells]")
    axs[1].set_title("surface profile across the width (x = Xn/2)", fontsize=10)
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    axs[1].set_box_aspect(1)
    fig.tight_layout()
    out = os.path.join(run, "cross_section_yz.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)

    # ---------------- 3D dump (half resolution: every 2nd cell) ----------------
    P = np.load(os.path.join(run, "phi3d_iter_%06d.npz" % it))["phi"].astype(np.float32)   # [x, y, z]
    nx3, ny3, nz3 = P.shape
    H = 2.0 * height(P)                                  # surface height in full-resolution cells, [x, y]
    xs = 2.0 * np.arange(nx3) + 1.0
    ys = 2.0 * np.arange(ny3) + 1.0

    # ---------------- long-section XZ ----------------
    phi_xz = np.load(os.path.join(run, "plane_xz_iter_%06d.npz" % it))["phi"]
    nx, nz = phi_xz.shape
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2, 1.2]}, facecolor="white")
    im = axs[0].imshow(phi_xz.T, origin="lower", cmap="RdBu", vmin=0, vmax=1, extent=(0, nx, 0, nz), aspect="equal")
    axs[0].contour(np.arange(nx) + 0.5, np.arange(nz) + 0.5, phi_xz.T, levels=[0.5], colors="k", linewidths=0.6)
    axs[0].set_title("phi in the XZ plane (y = Yn/2), iteration %d" % it, fontsize=10)
    axs[0].set_xlabel("x-index (periodic)")
    axs[0].set_ylabel("z-index")
    fig.colorbar(im, ax=axs[0], shrink=0.8, label="phi")
    for j, lab in ((ny3 // 2, "y = Yn/2"), (ny3 // 8, "y = Yn/8"), (2, "next to the wall")):
        axs[1].plot(xs, H[:, j], label=lab)
    axs[1].set_xlabel("x-index")
    axs[1].set_ylabel("surface height h(x) [cells]")
    axs[1].set_title("surface height along x (3D dump)", fontsize=10)
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(run, "long_section_xz.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)

    # ---------------- 3D meniscus ----------------
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, H, cmap="viridis", vmin=H.min(), vmax=H.max(), rcount=60, ccount=120,
                           linewidth=0, antialiased=False, shade=False)
    ax.set_xlim(0, 2 * nx3)
    ax.set_ylim(0, 2 * ny3)
    zlo, zhi = np.floor(H.min()) - 2, np.ceil(H.max()) + 2
    ax.set_zlim(zlo, zhi)
    ax.set_box_aspect((2 * nx3, 2 * ny3, zhi - zlo))                     # true scale: 1 cell is the same length on all axes
    ax.set_zticks([30, 45, 60])
    ax.view_init(elev=18, azim=-28)                                      # look nearly along x: the rise at the y walls shows end-on
    ax.set_xlabel("x-index (periodic)")
    ax.set_ylabel("y-index")
    ax.set_zlabel("z-index of phi = 0.5", labelpad=12)
    ax.set_title("Meniscus, iteration %d (true scale, 1:1:1)" % it)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.22, label="interface height z")
    out = os.path.join(run, "meniscus_3d_iter%06d.png" % it)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)

    # ---------------- 3D liquid block: faces coloured by phi, gas transparent, interface on top ----------------
    cmap = plt.get_cmap("RdBu")
    zlo_v, zhi_v = 0.0, 2.0 * nz3                        # the whole domain height: true proportions of liquid and gas
    kmax = nz3
    zz = 2.0 * np.arange(kmax) + 1.0
    f0 = np.load(os.path.join(run, "phi3d_iter_%06d.npz" % 0))["phi"].astype(np.float32).mean()   # initial liquid volume fraction

    def face_colors(F):                                  # F[a, z] -> RGBA, gas (phi < 0.05) fully transparent
        c = cmap(np.clip(F.astype(np.float64), 0.0, 1.0))
        gas = F < 0.2
        c[gas] = cmap(0.10)                              # gas: a clear red (the darkest end of RdBu goes muddy when transparent)
        c[..., 3] = np.where(gas, 0.38, 1.0)             # liquid and interface layer opaque, gas red and transparent
        return c

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    xmax, ymax = 2.0 * nx3, 2.0 * ny3
    YY, ZZ = np.meshgrid(ys, zz, indexing="ij")
    XX, ZX = np.meshgrid(xs, zz, indexing="ij")
    for ix, xc in ((0, 0.0), (nx3 - 1, xmax)):          # end faces x = 0 and x = Xn
        ax.plot_surface(np.full_like(YY, xc), YY, ZZ, facecolors=face_colors(P[ix, :, :kmax]), shade=False,
                        rstride=2, cstride=2, linewidth=0, edgecolor="none", antialiased=True)
    for iy, yc in ((0, 0.0), (ny3 - 1, ymax)):          # side faces y = 0 and y = Yn
        ax.plot_surface(XX, np.full_like(XX, yc), ZX, facecolors=face_colors(P[:, iy, :kmax]), shade=False,
                        rstride=2, cstride=2, linewidth=0, edgecolor="none", antialiased=True)
    gx, gy = np.gradient(H, xs, ys)                      # own light shading, so the phi = 0.5 surface stays light
    nrm = np.stack([-gx, -gy, np.ones_like(H)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    light = np.array([-0.35, -0.55, 0.76])
    light /= np.linalg.norm(light)
    shade = 0.78 + 0.22 * np.clip(nrm @ light, 0.0, 1.0)
    top = np.ones(H.shape + (4,))
    top[..., :3] = np.array(cmap(0.5)[:3]) * shade[..., None]
    ax.plot_surface(X, Y, H, facecolors=top, shade=False, rcount=60, ccount=120, linewidth=0, antialiased=False)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_zlim(zlo_v, zhi_v)
    ax.set_box_aspect((xmax, ymax, zhi_v - zlo_v))       # true scale
    ax.set_zticks([0, 100, 200, 300])
    ax.view_init(elev=18, azim=-28)
    ax.set_xlabel("x-index (periodic)")
    ax.set_ylabel("y-index")
    ax.set_zlabel("z-index", labelpad=12)
    m = re.search(r"tau_f_([0-9.]+)__tau_g_([0-9.]+)__Kf_([0-9.]+)__Theta_([0-9.]+)_alpha([0-9.]+)_sigma([0-9.]+)",
                  os.path.basename(os.path.normpath(run)))
    tf, tg, kf, th, al, sg = m.groups()
    fig.suptitle(
        "Multiphase flow in longitudinal channel\n"
        "Filename: free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.py\n"
        "Liquid/Gas proportion: %.1f/%.1f (initial)\n"
        "Solution: D3Q15, Zhang et al. (2020), x periodic\n"
        "Parameters: $\\theta$ = %s$^\\circ$, $\\alpha$ = %s$^\\circ$, $\\sigma$ = %s, $\\tau_f$ = %s, $\\tau_g$ = %s, $K_f$ = %s, W = 4, "
        "$\\rho_L/\\rho_G$ = 10/0.01, grid %dx%dx%d\n"
        "Iter: %d\n"
        "Scale: true scale, 1:1:1"
        % (f0, 1.0 - f0, th.rstrip("0").rstrip(".") if "." in th else th, al, sg, tf, tg, kf, 2 * nx3, 2 * ny3, 2 * nz3, it),
        x=0.02, y=0.98, ha="left", va="top", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.22, label="order parameter phi")
    out = os.path.join(run, "meniscus_3d_liquid_iter%06d.png" % it)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)

    # numbers
    hmid = H[:, ny3 // 2]
    print("h at y wall / y mid (x mid): %.1f / %.1f" % (H[nx3 // 2, 0], H[nx3 // 2, ny3 // 2]))
    print("h along x at y = Yn/2: min %.2f max %.2f (spread %.2f cells)" % (hmid.min(), hmid.max(), np.ptp(hmid)))
    print("h along x next to the wall (y index 0): spread %.2f cells" % np.ptp(H[:, 0]))


if __name__ == "__main__":
    main()
