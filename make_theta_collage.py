"""
1 x 3 collage of phi_snapshot_iter_<IT>.png from three Cap02 runs that differ only in the contact angle
(default: theta = 60, 90, 120 deg, iteration 12000).  Same crop / frame handling as make_phi_collage.py.

usage:  python make_theta_collage.py [iteration]
output: FreesurfaceImages/collage_theta_60_90_120_it<IT>.png
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mimg
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "FreesurfaceImages")
STUB = "free_surface_SchanChen_nondim_D2Q9_08_Zhang_Cap02__proof_capillary__nodes_300__tau_f_1.25__tau_g_0.65__Kf_0.002__Theta_%s"
THETAS = ("60.0", "90.0", "120.0")
CROP = (48, 734, 100, 787)       # strictly inside the axes frame of the 814 x 930 snapshot PNG


def main():
    it = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.8), sharex=True, sharey=True, facecolor="white")
    for ax, th in zip(axs, THETAS):
        f = os.path.join(IMAGES, STUB % th, "phi_snapshot_iter_%05d.png" % it)
        a = mimg.imread(f)[CROP[0]:CROP[1], CROP[2]:CROP[3], :3]
        ax.imshow(a, extent=(0, 302, 0, 302), aspect="equal")
        ax.set_title("$\\theta$ = %d$^\\circ$" % round(float(th)), fontsize=11)
        ax.set_xlabel("x-index")
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("black")
    axs[0].set_ylabel("y-index")
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(0.0, 1.0))
    fig.subplots_adjust(right=0.90, wspace=0.08)
    cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
    fig.colorbar(sm, cax=cax, label="order parameter $\\varphi$")
    out = os.path.join(IMAGES, "collage_theta_60_90_120_it%05d.png" % it)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
