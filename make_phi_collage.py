"""
Collage of the order-parameter snapshots of one run (3 x 2 panels like the reference collage).
The run folder only holds the snapshot PNGs written by Plotter2D.save_phi_snapshot (no raw arrays),
so the axes area of each PNG is cropped and re-tiled with a shared colourbar.

usage:  python make_phi_collage.py <run folder> [iter1 iter2 ... iter6]
output: <run folder>/collage_phi.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np

PHI_G, PHI_L = 0.0, 1.0          # colour range of the snapshots of this script (vmin/vmax of save_phi_snapshot)
CROP = (48, 734, 100, 787)       # rows, cols strictly inside the axes frame of the 814 x 930 snapshot PNG (frame at row 44/738, col 96/790)


def main():
    run = sys.argv[1]
    its = [int(a) for a in sys.argv[2:]] or [0, 500, 1285, 4052, 8400, 12000]
    fig, axs = plt.subplots(3, 2, figsize=(9, 11), sharex=True, sharey=True, facecolor="white")
    for ax, it in zip(axs.flat, its):
        f = os.path.join(run, "phi_snapshot_iter_%05d.png" % it)
        a = mimg.imread(f)[CROP[0]:CROP[1], CROP[2]:CROP[3], :3]
        ax.imshow(a, extent=(0, 302, 0, 302), aspect="equal")
        ax.set_title("iteration %d" % it, fontsize=10)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("black")
    for ax in axs[-1]:
        ax.set_xlabel("x-index")
    for ax in axs[:, 0]:
        ax.set_ylabel("y-index")
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(PHI_G, PHI_L))
    fig.subplots_adjust(right=0.86, hspace=0.18, wspace=0.08)
    cax = fig.add_axes([0.89, 0.25, 0.025, 0.5])
    fig.colorbar(sm, cax=cax, label="order parameter $\\varphi$")
    out = os.path.join(run, "collage_phi.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
