"""Wall-time analysis of studies 1 and 2.

usage: python analyse_timing.py <logs folder> [<out folder>]
  logs folder holds the lbm job outputs  t1_D<D>_r<rep>_<jobid>.out  and  t2_n<N>_d<dims>_r<rep>_<jobid>.out
Run-time is the loop time the script prints ("Run-time: X s"), set-up excluded. MAXITER is read from the log
(STUDY_MAXITER=...), ranks and node count from the "Running with N ranks on M node(s)" line.
"""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(f):
    t = open(f, errors="replace").read()
    m_rt = re.search(r"Run-time:\s*([0-9.]+) s", t)
    m_rk = re.search(r"Running with (\d+) ranks on (\d+) node", t)
    m_it = re.search(r"STUDY_MAXITER=(\d+)", t)
    m_d = re.search(r"STUDY_D_ND=(\d+)", t)
    m_dims = re.search(r"STUDY_DIMS=([0-9,]+)", t)
    if not (m_rt and m_rk and m_it and m_d and m_dims):
        return None
    return dict(runtime=float(m_rt.group(1)), ranks=int(m_rk.group(1)), nodes=int(m_rk.group(2)),
                iters=int(m_it.group(1)), D=int(m_d.group(1)), dims=m_dims.group(1))


def main():
    logs = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else logs
    os.makedirs(out, exist_ok=True)
    rows = []
    failed = []
    for f in sorted(glob.glob(os.path.join(logs, "t[12]_*.out"))):
        tag = re.match(r"(t[12]_.+?)_\d+\.out$", os.path.basename(f)).group(1)
        r = parse(f)
        if r is None:
            failed.append(tag)
            continue
        r["tag"] = tag
        r["study"] = tag[:2]
        r["rep"] = int(tag.rsplit("_r", 1)[1])
        r["cells"] = 2 * r["D"] ** 3
        r["s_per_iter"] = r["runtime"] / r["iters"]
        rows.append(r)

    with open(os.path.join(out, "timing_summary.csv"), "w") as fh:
        fh.write("tag,D_ND,dims,ranks,nodes,iters,runtime_s,s_per_iter\n")
        for r in rows:
            fh.write("%s,%d,%s,%d,%d,%d,%.2f,%.4f\n" % (r["tag"], r["D"], r["dims"].replace(",", "x"), r["ranks"], r["nodes"],
                     r["iters"], r["runtime"], r["s_per_iter"]))
    if failed:
        print("no Run-time line (job failed / timed out / still running):", ", ".join(failed))

    print("\nstudy 1: lattice size, 1024 ranks, dims 16,8,8")
    print("%6s %12s %6s %12s %10s" % ("D_ND", "cells", "nodes", "runtime[s]", "s/iter"))
    by = defaultdict(list)
    for r in rows:
        if r["study"] == "t1":
            by[r["D"]].append(r)
    s1 = []
    for D in sorted(by):
        rs = by[D]
        m = np.mean([x["s_per_iter"] for x in rs])
        s1.append((D, m, [x["s_per_iter"] for x in rs]))
        print("%6d %12d %6s %12.1f %10.4f" % (D, rs[0]["cells"], "/".join(str(x["nodes"]) for x in rs),
              np.mean([x["runtime"] for x in rs]), m))

    print("\nstudy 2: MPI decomposition, D_ND = 384")
    print("%6s %10s %6s %10s %s" % ("ranks", "dims", "nodes", "s/iter", "repeats s/iter"))
    by = defaultdict(list)
    for r in rows:
        if r["study"] == "t2":
            by[(r["ranks"], r["dims"])].append(r)
    s2 = []
    for key in sorted(by):
        rs = by[key]
        m = np.mean([x["s_per_iter"] for x in rs])
        s2.append((key, m, [x["s_per_iter"] for x in rs]))
        print("%6d %10s %6s %10.4f %s" % (key[0], key[1].replace(",", "x"), "/".join(str(x["nodes"]) for x in rs),
              m, " ".join("%.4f" % x["s_per_iter"] for x in rs)))

    ranks = sorted({r["ranks"] for r in rows})
    colors = ["purple", "red", "orange", "lime", "aqua"]

    def fit_line(ax, x, y, label_sym):
        """straight-line least-squares fit in log-log, drawn over the x range"""
        p = np.polyfit(np.log10(x), np.log10(y), 1)
        xs = np.logspace(np.log10(min(x)), np.log10(max(x)), 50)
        ax.plot(xs, 10 ** np.polyval(p, np.log10(xs)), "k--", lw=1.5,
                label="best fit: t ~ %s^%.2f" % (label_sym, p[0]))

    # plot 1: lattice size, 1024 ranks only
    fig, ax = plt.subplots(figsize=(8, 8))
    rs = [r for r in rows if r["ranks"] == 1024]
    x = [r["D"] for r in rs]
    y = [r["s_per_iter"] for r in rs]
    ax.scatter(x, y, s=45, color="lime", edgecolor="black", linewidth=0.5, label="1024 ranks")
    fit_line(ax, x, y, "D")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("DEFAULT_D_ND (cells across the channel; lattice 2 D x D x D)")
    ax.set_ylabel("wall time / iter (s)")
    ax.set_title("Variation wall time/iter for various lattice sizes, 1024 ranks")
    ax.legend(); ax.grid(alpha=0.25); ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timing_vs_lattice.png"), dpi=130)

    # plot 2: number of ranks, lattice 768 x 384 x 384
    fig, ax = plt.subplots(figsize=(8, 8))
    x, y = [], []
    for k, nr in enumerate(ranks):
        rs = [r for r in rows if r["ranks"] == nr and r["D"] == 384]
        ax.scatter([nr] * len(rs), [r["s_per_iter"] for r in rs], s=45, color=colors[k % len(colors)],
                   edgecolor="black", linewidth=0.5)
        x += [nr] * len(rs)
        y += [r["s_per_iter"] for r in rs]
    fit_line(ax, x, y, "ranks")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ranks); ax.set_xticklabels([str(n) for n in ranks])
    ax.minorticks_off()
    ax.set_xlabel("number of MPI ranks")
    ax.set_ylabel("wall time / iter (s)")
    ax.set_title("Wall time/iter vs number of MPI ranks (lattice 768 x 384 x 384)")
    ax.legend(); ax.grid(alpha=0.25); ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timing_vs_ranks.png"), dpi=130)
    print("\nwrote timing_summary.csv, timing_vs_lattice.png and timing_vs_ranks.png in", out)


if __name__ == "__main__":
    main()
