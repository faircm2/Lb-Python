"""
Velocity and force plots over the whole domain (liquid and gas) from the field dumps written by Zhang_Cap02_fielddump.py
(see make_field_dump_copy.py).  Layout as in the reference figure: top row = magnitude map with colourbar,
bottom row = vectors over phi (grey = liquid, white = gas).

usage:  python plot_vel_force_fields.py <fields folder> [<it1> <it2> ...] [--stride 8] [--ywin 100 200]   (no iterations given: 4 spread over the run)
from a run script:  from plot_vel_force_fields import generate_plots;  generate_plots(images_dir)
writes: <fields folder>/velocity_fields.png     (|u|   and velocity vectors)
        <fields folder>/force_fields.png        (|Fs|  capillary force and force vectors)
        <fields folder>/total_force_fields.png  (|Fs + Fb| capillary + body force)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


CLIP_PERCENTILE = 99.5


def load(folder, it):
    d = np.load(os.path.join(folder, "fields_it%05d.npz" % it))
    # drop the ghost layer, index [x, y] -> plot with .T and origin='lower'
    return {k: d[k][..., 1:-1, 1:-1] for k in ("phi", "u", "Fs", "Fb", "rho")}


def panel(fig, axs, d, vec, name, unit_label, stride, ywin, cmap="magma"):
    mag = np.hypot(vec[0], vec[1])
    phi = d["phi"]
    nx, ny = phi.shape
    y0, y1 = ywin if ywin else (0, ny)
    ext = (0, nx, 0, ny)
    ax = axs[0]
    # colour range and arrow length from a high percentile, so a few corner / contact-line cells do not flatten everything else
    ref = np.percentile(mag, CLIP_PERCENTILE)
    im = ax.imshow(mag.T, origin="lower", cmap=cmap, extent=ext, aspect="equal", vmin=0, vmax=ref)
    ax.set_ylim(y0, y1)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08), extend="max")
    ax.set_ylabel("y-index")
    ax.set_title("%s, max %.2e (scale to %.1f%% = %.2e)" % (name, mag.max(), CLIP_PERCENTILE, ref), fontsize=8)
    ax2 = axs[1]
    ax2.imshow(phi.T, origin="lower", cmap="Greys", vmin=0, vmax=1, extent=ext, aspect="equal", alpha=0.55)
    xs = np.arange(0, nx, stride) + 0.5
    ys = np.arange(0, ny, stride) + 0.5
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    U = vec[0][::stride, ::stride]
    V = vec[1][::stride, ::stride]
    M = mag[::stride, ::stride]
    qv = ax2.quiver(X, Y, U, V, np.minimum(M, ref), cmap="autumn_r", clim=(0, ref), angles="xy", scale_units="xy",
                    scale=max(ref, 1e-30) / (1.5 * stride), width=0.0025)
    # same colourbar geometry as the top row, so both rows have identically sized axes (aligned panels)
    fig.colorbar(qv, cax=make_axes_locatable(ax2).append_axes("right", size="4%", pad=0.08), extend="max")
    ax2.set_ylim(y0, y1)
    ax2.set_xlabel("x-index")
    ax2.set_ylabel("y-index")
    ax2.set_title("%s vectors over phi (grey = liquid)" % name, fontsize=8)


def interface_window(ds, key, extra, margin=15, frac=0.02):
    """y-range that contains the force band of all selected iterations (+ margin)."""
    lo, hi = None, None
    for d in ds:
        vec = d[key] if extra is None else d[key] + d[extra]
        mag = np.hypot(vec[0], vec[1])
        rows = np.where((mag > frac * mag.max()).any(axis=0))[0]
        lo = rows.min() if lo is None else min(lo, rows.min())
        hi = rows.max() if hi is None else max(hi, rows.max())
    return (max(int(lo) - margin, 0), int(hi) + margin)


def figure(folder, its, key, name, out, stride, ywin, extra=None, auto_window=False):
    ds = [load(folder, it) for it in its]
    if auto_window and ywin is None:
        ywin = interface_window(ds, key, extra)
    height = 3.4 if auto_window else 7.5          # the zoomed force band is wide and thin
    fig, axs = plt.subplots(2, len(its), figsize=(5.2 * len(its), height), squeeze=False)
    for j, (it, d) in enumerate(zip(its, ds)):
        vec = d[key] if extra is None else d[key] + d[extra]
        panel(fig, axs[:, j], d, vec, "iteration %d: |%s|" % (it, name), "", stride, ywin)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def default_iterations(folder, n=2):
    """n dumped iterations: the dump nearest to 10% of the run (early flow) up to the last one."""
    its = sorted(int(f[9:14]) for f in os.listdir(folder) if f.startswith("fields_it") and f.endswith(".npz"))
    targets = np.linspace(0.1 * its[-1], its[-1], n)
    return sorted(set(min(its, key=lambda i: abs(i - t)) for t in targets))


def generate_plots(folder, its=None, stride=8, ywin=None):
    """All three figures for the dumps in `folder`; called at the end of a run."""
    its = its or default_iterations(folder)
    figure(folder, its, "u", "u", os.path.join(folder, "velocity_fields.png"), stride, ywin)
    # the capillary force lives in a thin band at the interface: zoom on it and sample finer (stride 8 would miss the band)
    figure(folder, its, "Fs", "Fs", os.path.join(folder, "force_fields.png"), min(stride, 3), ywin, auto_window=True)
    figure(folder, its, "Fs", "Fs+Fb", os.path.join(folder, "total_force_fields.png"), min(stride, 3), ywin, extra="Fb", auto_window=True)


def main():
    a = sys.argv[1:]
    folder = a[0]
    stride = 8
    ywin = None
    its = []
    i = 1
    while i < len(a):
        if a[i] == "--stride":
            stride = int(a[i + 1]); i += 2
        elif a[i] == "--ywin":
            ywin = (int(a[i + 1]), int(a[i + 2])); i += 3
        else:
            its.append(int(a[i])); i += 1
    generate_plots(folder, its or None, stride, ywin)


if __name__ == "__main__":
    main()
