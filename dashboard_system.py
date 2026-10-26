"""
Post-run dashboard for the free-surface LBM runs (inclined channel, Zhang et al. 2020).

Reads only what the simulation already writes into its output folder:
    plane_yz_iter_XXXXXX.npz   phi (Y,Z) and u (3,Y,Z) at x = Xn//2   (folder or folder/planes)
    diagnostic_lists.npz       optional, used for the mass-conservation panel
    parameters.txt             optional, otherwise the run parameters are parsed from the folder name

usage:  python dashboard_system.py <run_folder> [out.png] [--sigma S] [--tau_f T] [--tau_g G] [--theta TH] [--alpha A]
        (the options are only needed when the folder name does not carry the parameters, e.g. a local copy)
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["legend.fancybox"] = False      # square legend frames
import numpy as np

CS = np.sqrt(1 / 3)
CS2 = 1 / 3
D_PHYS, N_CELLS, NU_PHYS = 1e-3, 384, 2.9e-6          # same constants as the simulation
DX = D_PHYS / N_CELLS
ZCROP = 100                                            # height of the y-z maps shown (--zcrop overrides)
C_LIQ, C_GAS, C_L, C_R = "tab:red", "tab:blue", "black", "darkgreen"   # same as the phi map: red = liquid (phi 1), blue = gas (phi 0); left/right use neutral colours


def run_parameters(folder):
    """tau_f, tau_g, sigma, theta, alpha, rho_L, rho_G, multiples - from the folder name (+ defaults)."""
    name = os.path.basename(os.path.normpath(folder))
    g = lambda pat, default: float(re.search(pat, name).group(1)) if re.search(pat, name) else default
    p = dict(tau_f=g(r"tau_f_([\d.]+?)__", 1.0), tau_g=g(r"tau_g_([\d.]+?)__", 0.65),
             theta=g(r"Theta_([\d.]+?)_", 60.0), alpha=g(r"_alpha([\d.]+?)_", 45.0),
             sigma=g(r"_sigma([\d.]+?)(?:_|$)", 0.0), rho_G=g(r"_rhoG([\d.]+?)(?:_|$)", 0.01),
             mult=g(r"_x(\d+)$", 1), rho_L=g(r"_rhoL([\d.]+?)(?:_|$)", 10.0))
    return p


def load_planes(folder):
    files = sorted(glob.glob(os.path.join(folder, "plane_yz_iter_*.npz")) +
                   glob.glob(os.path.join(folder, "planes", "plane_yz_iter_*.npz")))
    its, phis, us = [], [], []
    for f in files:
        d = np.load(f)
        its.append(int(re.search(r"iter_(\d+)", f).group(1)))
        phis.append(d["phi"])
        us.append(d["u"])
    order = np.argsort(its)
    return np.array(its)[order], [phis[i] for i in order], [us[i] for i in order]


def main(folder, out, overrides=None, zcrop=None):
    global ZCROP
    if zcrop:
        ZCROP = int(zcrop)
    p = run_parameters(folder)
    p.update({k: v for k, v in (overrides or {}).items() if v is not None})
    nu = CS2 * (p["tau_f"] - 0.5)
    mobility = CS2 * (p["tau_g"] - 0.5)
    dt = nu * DX**2 / NU_PHYS                               # eq.(25) solved for dt
    its, phis, us = load_planes(folder)
    t = its * dt * 1e3                                      # ms
    Y, Z = phis[0].shape

    # ---- time series from the plane dumps -------------------------------------------------
    wall, well, centre, mach_liq, mach_gas, corner_l, corner_r, speed_liq, speed_all = ([] for _ in range(9))
    for phi, u in zip(phis, us):
        h = phi.sum(axis=1)                                  # liquid height per y (cells)
        wall.append(0.5 * (h[0] + h[-1]))
        well.append(0.5 * (h[:Y // 2].min() + h[Y // 2:].min()))
        centre.append(0.5 * (h[Y // 2 - 1] + h[Y // 2]))
        speed = np.sqrt((u ** 2).sum(axis=0))
        speed_liq.append(speed[phi > 0.5].max() if (phi > 0.5).any() else 0.0)
        speed_all.append(speed[phi < 0.5].max() if (phi < 0.5).any() else 0.0)   # gas
        mach_liq.append(speed[phi > 0.5].max() / CS if (phi > 0.5).any() else 0.0)
        mach_gas.append(speed[phi < 0.5].max() / CS if (phi < 0.5).any() else 0.0)
        corner_l.append(int((phi[:40, :30] < 0.5).sum()))        # gas cells in the left floor corner box
        corner_r.append(int((phi[Y - 40:, :30] < 0.5).sum()))

    phi, u = phis[-1], us[-1]
    speed = np.sqrt((u ** 2).sum(axis=0))
    title = (f"alpha={p['alpha']:g}; vf_sigma={p['sigma']:g}; vf_theta={p['theta']:g}; tau_f={p['tau_f']:g}; "
             f"rho_G:rho_L=1:{p['rho_L'] / p['rho_G']:.0f};  iter={its[-1]}  (t={t[-1]:.2f} ms)")

    fig, ax = plt.subplots(3, 3, figsize=(19, 11.5))
    fig.suptitle("Dashboard: " + title, fontsize=12)

    def yz_map(a, f, name, cbar, cmap="viridis", sym=False):
        v = np.abs(f).max()                                  # colour scale = true range, no clipping
        im = a.imshow(f[:, :ZCROP].T, origin="lower", aspect="equal", cmap=cmap,
                      vmin=-v if sym else 0, vmax=v, extent=(0, Y, 0, ZCROP))
        a.contour(np.arange(Y) + 0.5, np.arange(ZCROP) + 0.5, phi[:, :ZCROP].T, levels=[0.5],
                  colors="cyan", linewidths=1.4)
        a.set_title(f"{name}  (max {np.abs(f).max():.2e})\nz shown: 0-{ZCROP} of {Z} cells", fontsize=10)
        a.set_xlabel("y"); a.set_ylabel("z")
        fig.colorbar(im, ax=a, label=cbar, fraction=0.04, pad=0.02)

    # row 1: y-z maps of the last dump ---------------------------------------------------------
    yz_map(ax[0, 1], phi, "phi (cyan: phi=0.5)", "phi", cmap="RdBu_r")
    ax[0, 1].images[0].set_clim(0, 1)
    yz_map(ax[0, 2], u[0], "u_x", "u_x", cmap="RdBu_r", sym=True)
    yz_map(ax[1, 0], speed / CS, "Mach number", "Ma = |u| / c_s")

    # row 2: time series ---------------------------------------------------------------------
    a = ax[1, 1]
    a.plot(t, wall, color="tab:green", lw=2, label="wall")
    a.plot(t, well, color="tab:orange", lw=2, label="well")
    a.plot(t, centre, color="tab:purple", lw=2, label="centre")
    a.set_title("Meniscus heights vs. time", fontsize=10); a.set_xlabel("t [ms]"); a.set_ylabel("liquid height [cells]")
    a.legend(); a.grid(alpha=.3)

    a = ax[1, 2]
    a.plot(t, mach_liq, color=C_LIQ, lw=2, label="liquid")
    a.plot(t, mach_gas, color=C_GAS, lw=2, label="gas")
    a.set_title("Max Mach number vs. time", fontsize=10); a.set_xlabel("t [ms]"); a.set_ylabel("Ma")
    a.legend(); a.grid(alpha=.3)

    a = ax[2, 0]
    a.plot(t, corner_l, color=C_L, lw=2, label="left corner")
    a.plot(t, corner_r, color=C_R, lw=2, ls="--", label="right corner")
    a.set_xlabel("t [ms]"); a.set_ylabel("gas cells (phi < 0.5)")
    a.set_ylim(bottom=0, top=max(1, max(corner_l + corner_r) * 1.1))
    a.set_title("Gas in the floor corners\n(boxes of 40 x 30 cells at y=0 and y=383, z=0)", fontsize=10)
    a.legend(fontsize=9, loc="upper left"); a.grid(alpha=.3)

    # row 3: mass, viscosity profile, parameters --------------------------------------------------
    a = ax[2, 1]
    dl = os.path.join(folder, "diagnostic_lists.npz")
    if os.path.exists(dl):
        mass = np.load(dl, allow_pickle=True)["PhiCollector"]
        a.plot(mass[:, 0] * dt * 1e3, mass[:, 1] / mass[0, 1] - 1, color="tab:blue", lw=2)
        a.set_ylabel("relative drift of total phi")
    else:
        a.text(0.5, 0.5, "diagnostic_lists.npz not in folder", ha="center", va="center", transform=a.transAxes)
    a.set_title("Mass conservation", fontsize=10); a.set_xlabel("t [ms]"); a.grid(alpha=.3)

    a = ax[2, 2]
    col = Y // 2
    rho = phi[col] * p["rho_L"] + (1 - phi[col]) * p["rho_G"]
    eta = rho * nu                                           # eq.(25), lattice units
    zz = np.arange(Z)
    a.semilogy(zz, eta, color="tab:blue", lw=2, label="eta = rho*nu  (eq.25)")
    a.semilogy(zz, rho, color="tab:red", lw=2, label="rho")
    a.set_xlim(0, ZCROP)
    a.set_title("Interface: density and viscosity", fontsize=10)
    a.set_xlabel("z"); a.set_ylabel("lattice units"); a.legend(); a.grid(alpha=.3, which="both")

    a = ax[0, 0]; a.axis("off")
    eta_L, eta_G = p["rho_L"] * nu, p["rho_G"] * nu
    g_lat = 9.81 * dt**2 / DX                                   # lattice gravity
    g_z = g_lat * np.cos(np.radians(p["alpha"]))                # component normal to the surface
    d_rho = p["rho_L"] - p["rho_G"]
    no_gz = abs(g_z) < 1e-9 * g_lat                             # alpha = 90: no gravity normal to the surface
    l_c = np.sqrt(p["sigma"] / (d_rho * g_z)) if (p["sigma"] > 0 and not no_gz) else float("nan")
    bond = d_rho * g_z * Y**2 / p["sigma"] if (p["sigma"] > 0 and not no_gz) else float("nan")
    L0 = 37.5                                                   # initial liquid depth (10 % fill)
    L_gas = Z - L0                                              # gas depth above the initial liquid level
    re_liq, re_gas = max(speed_liq) * L0 / nu, max(speed_all) * L_gas / nu          # max over all dumps
    re_gas_last = speed_all[-1] * L_gas / nu
    txt = (f"tau_f = {p['tau_f']:g}   ->  nu = c_s^2 (tau_f - 1/2) = {nu:.4f}\n"
           f"tau_g = {p['tau_g']:g}   ->  mobility M_phi = {mobility:.4f}\n"
           f"eta_L = {eta_L:.4g},  eta_G = {eta_G:.4g}   (ratio {eta_L / eta_G:.0f})\n"
           f"dt = {dt:.4e} s,   dx = {DX:.4e} m\n"
           f"sigma = {p['sigma']:g}\n"
           f"theta = {p['theta']:g} deg\n"
           f"alpha = {p['alpha']:g} deg\n"
           f"rho_L = {p['rho_L']:g},  rho_G = {p['rho_G']:g}\n"
           f"W = {Y} cells ({Y * DX * 1e3:.2f} mm)\n"
           + ("Bond number Bo = 0  (g_z = 0 at alpha 90)\n"
              "capillary length l_c = infinite  (g_z = 0)\n"
              "W/l_c = 0\n" if no_gz else
              f"Bond number Bo = drho g_z W^2 / sigma = {bond:.3g}\n"
              f"capillary length l_c = {l_c:.0f} cells = {l_c * DX * 1e3:.2f} mm\n"
              f"W/l_c = {Y / l_c:.2f}\n") +
           f"Re (fluid) max = {re_liq:.3g}   (|u|max in liquid x L_liq {L0} / nu)\n"
           f"Re (gas)   max = {re_gas:.3g}, last dump {re_gas_last:.3g}   (L_gas {L_gas:g})\n"
           f"dumps: {len(its)}\n"
           f"final iter: {its[-1]}\n"
           f"simulated time: {t[-1]:.2f} ms")
    a.text(0.0, 0.98, txt, va="top", family="monospace", fontsize=8.5, transform=a.transAxes)
    a.set_title("Parameters", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--zcrop", type=int, default=None)
    for k in ("sigma", "tau_f", "tau_g", "theta", "alpha", "rho_L", "rho_G"):
        ap.add_argument("--" + k, type=float, default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.folder, "dashboard_system.png")
    main(a.folder, out, {k: getattr(a, k) for k in ("sigma", "tau_f", "tau_g", "theta", "alpha", "rho_L", "rho_G")}, a.zcrop)
