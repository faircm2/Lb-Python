"""
3D meniscus surface (phi = 0.5) and a 1 x 3 canvas of phi cross-sections through the cube,
from a fields_iter_<N>.npz dump of the D3Q15 Zhang run.

usage:  python plot_D3Q15_meniscus.py <run folder> [iteration]      (default iteration 12000)
writes: <run folder>/meniscus_3d_iter<N>.png
        <run folder>/cross_sections_phi_iter<N>.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def interface_height(P):
    """z of phi = 0.5 for every (x, y) column (liquid below, gas above), linear interpolation."""
    lo = P[:, :, :-1] >= 0.5
    hi = P[:, :, 1:] < 0.5
    cross = lo & hi
    k = cross.shape[2] - 1 - np.argmax(cross[:, :, ::-1], axis=2)      # highest crossing in each column
    ix, iy = np.indices(k.shape)
    a, b = P[ix, iy, k], P[ix, iy, k + 1]
    return k + (a - 0.5) / (a - b)


def main():
    run = sys.argv[1]
    it = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    d = np.load(os.path.join(run, "fields_iter_%05d.npz" % it))
    P = d["phi"][1:-1, 1:-1, 1:-1]                                      # drop the ghost layer, index [x, y, z]
    n = P.shape[0]
    h = interface_height(P)

    # ---- 3D meniscus -------------------------------------------------------------------------------------
    X, Y = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, h, cmap="viridis", vmin=h.min(), vmax=h.max(), rcount=100, ccount=100,
                           linewidth=0, antialiased=True)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(np.floor(h.min()) - 2, np.ceil(h.max()) + 2)
    ax.set_box_aspect((1, 1, 0.5))                                      # z exaggerated: the rise is ~10 of 200 cells
    ax.view_init(elev=28, azim=-55)
    ax.set_xlabel("x-index")
    ax.set_ylabel("y-index")
    ax.set_zlabel("z-index of $\\varphi$ = 0.5", labelpad=12)
    ax.set_title("Meniscus, iteration %d (z axis exaggerated)" % it)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.16, label="interface height z")
    out3d = os.path.join(run, "meniscus_3d_iter%05d.png" % it)
    fig.savefig(out3d, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out3d)

    # ---- 1 x 3 cross-sections ---------------------------------------------------------------------------
    m = n // 2
    zc = int(round(0.5 * (h[m, m] + h[0, m])))                          # plane halfway between apex and wall height
    panels = [
        ("y = %d (XZ)" % m, P[:, m, :].T, "x-index", "z-index"),
        ("x = %d (YZ)" % m, P[m, :, :].T, "y-index", "z-index"),
        ("z = %d (XY)" % zc, P[:, :, zc].T, "x-index", "y-index"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="white")
    for ax, (title, A, xl, yl) in zip(axs, panels):
        im = ax.imshow(A, origin="lower", cmap="RdBu", vmin=0.0, vmax=1.0, extent=(0, n, 0, n), aspect="equal")
        ax.contour(np.arange(n) + 0.5, np.arange(n) + 0.5, A, levels=[0.5], colors="k", linewidths=0.6)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_box_aspect(1)
    fig.suptitle("$\\varphi$ cross-sections, iteration %d" % it)
    fig.tight_layout()
    fig.colorbar(im, ax=axs, shrink=0.7, pad=0.02, label="order parameter $\\varphi$")
    out2d = os.path.join(run, "cross_sections_phi_iter%05d.png" % it)
    fig.savefig(out2d, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out2d)


if __name__ == "__main__":
    main()
