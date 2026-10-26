"""
Isolated 2D test of Zhang et al. (2020) phase-field LBM with the wetting boundary condition (eqs. 31-32), for the YZ cross-section of the channel:
a flat liquid layer (initial depth 38 cells) on the floor, two side walls, top wall, no gravity (alpha = 90), same lattice parameters as the 3D runs.
Question: does the floor ripple / bubble formation of the 3D runs (theta 60: ripple from iteration 100, bubbles at 36,000; theta 90: none) appear in this
reduced model too?  If yes, the cause sits in the wall treatment + phase-field model itself and fixes can be tested here in minutes.

Mirrors the 3D script (free_surface_..._Cap06): eqs (4)-(6), (12)-(13), (17)-(20), (22)-(23), (26)-(28), (31)-(32), phi clipped to [0, 1] after the moment,
global mass rescaling after the wetting step, ghost layers mirrored then overwritten by phi_s, n = grad(phi)/|grad(phi)| (n = grad if |grad| < 1e-12).
Simplifications: D2Q9; half-way bounce-back for BOTH g and f (the 3D script uses Zhang's interpolation scheme, eq. 30, for f); no body force.

usage:  python zhang2d_wetting_test_system.py --theta 60 --sigma 0.05 --iters 40000 --out run_t60.txt [--noclip] [--nomass] [--ny 384 --nz 110]
"""
import argparse
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--theta", type=float, default=60.0)
ap.add_argument("--sigma", type=float, default=0.05)
ap.add_argument("--tau_f", type=float, default=0.65)
ap.add_argument("--tau_g", type=float, default=0.65)
ap.add_argument("--W", type=float, default=4.0)
ap.add_argument("--ny", type=int, default=384)
ap.add_argument("--nz", type=int, default=110)
ap.add_argument("--iters", type=int, default=40000)
ap.add_argument("--every", type=int, default=500)
ap.add_argument("--out", default="run2d.txt")
ap.add_argument("--fbc", default="bb", choices=["bb", "interp", "paper", "readC"], help="f boundary treatment: bb = half-way bounce-back, interp = Zhang eq. (30) with q = 1/2 as in the 3D script (TEMPORARY TEST OPTION)")
ap.add_argument("--tauvar", action="store_true", help="TEMPORARY TEST: tau_f depends on phi through the mixture rule eta = phi*eta_L + (1-phi)*eta_G (0.5 and 0.007, Zhang case A) and eta = rho c_s^2 (tau_f - 1/2)")
ap.add_argument("--noclip", action="store_true", help="do not clip phi to [0,1]")
ap.add_argument("--nomass", action="store_true", help="do not rescale phi to the previous total mass")
ap.add_argument("--depth", type=float, default=38.5, help="initial interface position (cells above the floor)")
a = ap.parse_args()

Ny, Nz = a.ny, a.nz
NY, NZ = Ny + 2, Nz + 2                       # arrays carry one ghost layer on every side (indices 0 and N+1)
cs2 = 1.0 / 3.0
cs4 = cs2 ** 2
c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]
rho_L, rho_G = 10.0, 0.01
beta = 12.0 * a.sigma / a.W
kappa = 3.0 * a.sigma * a.W / 2.0
Theta = -np.sqrt(2.0 * beta / kappa) * np.cos(np.deg2rad(a.theta))


def zero_ghosts(f):
    f[0, :] = 0.0; f[-1, :] = 0.0; f[:, 0] = 0.0; f[:, -1] = 0.0
    return f


def shift(f, i):
    """value at x + e_i"""
    return np.roll(np.roll(f, -c[i, 0], axis=0), -c[i, 1], axis=1)


def grad(f):
    gy = np.zeros_like(f); gz = np.zeros_like(f)
    for i in range(9):
        s = shift(f, i)
        gy += w[i] * c[i, 0] * s
        gz += w[i] * c[i, 1] * s
    return zero_ghosts(gy / cs2), zero_ghosts(gz / cs2)          # eq (22)


def lap(f):
    L = np.zeros_like(f)
    for i in range(1, 9):
        L += w[i] * (shift(f, i) - f)
    return zero_ghosts(L * (2.0 / cs2))                          # eq (23)


def phi_s(pp, nd=1.0):
    h = s = nd / 2.0
    if a.theta == 90.0:
        return pp
    disc = 1.0 + h * Theta - np.sqrt((1.0 + h * Theta) ** 2 - 4.0 * h * Theta * pp)
    return ((s + h) / (2.0 * h ** 2 * Theta)) * disc - (s / h) * pp   # eq (32)


def set_solid(phi):
    p = phi.copy()
    p[0, :] = p[1, :]; p[-1, :] = p[-2, :]; p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]      # mirror the ghosts first
    bottom = phi_s(p[:, 1]); top = phi_s(p[:, Nz]); left = phi_s(p[1, :]); right = phi_s(p[Ny, :])
    p[:, 0] = bottom; p[:, Nz + 1] = top; p[0, :] = left; p[Ny + 1, :] = right          # same order as the 3D script: z walls, then side walls
    return p


def mu_phi(phi):
    return 4.0 * beta * phi * (phi - 1.0) * (phi - 0.5) - kappa * lap(phi)               # eq (6)


def normal(phi):
    gy, gz = grad(phi)
    mag = np.sqrt(gy ** 2 + gz ** 2)
    mag = np.where(mag < 1e-12, 1.0, mag)
    return gy / mag, gz / mag


# ---- initial state -----------------------------------------------------------------------------------------------------------------------------
zz = np.arange(NZ)[None, :] * np.ones((NY, 1))
phi = 0.5 * (1.0 + np.tanh(2.0 * (a.depth - zz) / a.W))
phi = set_solid(phi)
uy = np.zeros((NY, NZ)); uz = np.zeros((NY, NZ))
rho = phi * rho_L + (1 - phi) * rho_G
p = np.zeros((NY, NZ))
g = np.array([w[i] * phi for i in range(9)])
f = np.zeros((9, NY, NZ))
phi_old = phi.copy(); uy_old = uy.copy(); uz_old = uz.copy()
fluid = (slice(1, Ny + 1), slice(1, Nz + 1))


def bounce(arr_pre, arr_post):
    """half-way bounce-back at the four walls: incoming populations of the wall-adjacent row = reflected post-collision populations of the same node"""
    for i in range(9):
        j = opp[i]
        if c[i, 1] > 0:      # comes from the floor side: unknown at z = 1
            arr_post[i, :, 1] = arr_pre[j, :, 1]
        if c[i, 1] < 0:      # top wall: unknown at z = Nz
            arr_post[i, :, Nz] = arr_pre[j, :, Nz]
        if c[i, 0] > 0:      # left wall (y = 1)
            arr_post[i, 1, :] = arr_pre[j, 1, :]
        if c[i, 0] < 0:      # right wall (y = Ny)
            arr_post[i, Ny, :] = arr_pre[j, Ny, :]
    return arr_post


def stream(arr):
    return np.array([np.roll(np.roll(arr[i], c[i, 0], axis=0), c[i, 1], axis=1) for i in range(9)])


def bubbles(ph):
    try:
        from scipy import ndimage as ndi
    except Exception:
        return None
    lab, n = ndi.label(ph[1:Ny + 1, 1:Nz + 1] < 0.5)
    top = set(np.unique(lab[:, -1])) - {0}
    return [int((lab == k).sum()) for k in range(1, n + 1) if k not in top]


def ripple(ph):
    y = np.arange(2, 42); out = []
    for z in (1, 2):                              # rows z = 0 and 1 of the 3D analysis (first fluid rows)
        d = 1.0 - ph[2 + 0:42 + 0, z] if False else 1.0 - ph[3:43, z]
        d = d - d.mean(); cc = np.cos(2 * np.pi * y / 4.0); ss = np.sin(2 * np.pi * y / 4.0)
        out.append(2.0 / len(y) * np.hypot((d * cc).sum(), (d * ss).sum()))
    return float(np.mean(out))


log = open(a.out, "w")
def say(s):
    print(s, flush=True); log.write(s + "\n"); log.flush()

say("# fbc %s  tauvar %s" % (a.fbc, a.tauvar))
say("# theta %g sigma %g tau_f %g tau_g %g W %g  grid %dx%d  clip %s  mass-rescale %s  Theta %.4f beta %.4f kappa %.4f" %
    (a.theta, a.sigma, a.tau_f, a.tau_g, a.W, Ny, Nz, not a.noclip, not a.nomass, Theta, beta, kappa))
say("#    iter  floor_min_phi  at_y  cells<0.999  ripple4   bubbles(count,cells)  max|u|   mass")
t0 = time.time()
mass0 = phi[fluid].sum()
for it in range(a.iters + 1):
    if it % a.every == 0 or it < 500 and it % 100 == 0:
        row = phi[3:190, 1]
        b = bubbles(phi)
        say("%9d  %12.5f %5d %9d  %9.2e   %s   %.2e  %+.1e   (%.0fs)" % (
            it, row.min(), int(row.argmin()) + 3, int((row < 0.999).sum()), ripple(phi),
            "n/a" if b is None else "%d,%d" % (len(b), sum(b)), np.sqrt(uy ** 2 + uz ** 2).max(), phi[fluid].sum() / mass0 - 1, time.time() - t0))
        if np.isnan(phi).any():
            say("NaN - stop"); break
    # equilibria (17), (18)
    cu = np.array([c[i, 0] * uy + c[i, 1] * uz for i in range(9)])
    geq = np.array([w[i] * phi * (1 + 3.0 * cu[i]) for i in range(9)])
    u2 = uy ** 2 + uz ** 2
    feq = np.array([w[i] * (p + rho * (cu[i] + cu[i] ** 2 / (2 * cs2) - 0.5 * u2)) for i in range(9)])
    # eq (19)
    ny_, nz_ = normal(phi)
    lam = 4.0 * phi * (1.0 - phi) / a.W
    vy = (phi * uy - phi_old * uy_old) + cs2 * lam * ny_
    vz = (phi * uz - phi_old * uz_old) + cs2 * lam * nz_
    Gi = (1.0 - 1.0 / (2.0 * a.tau_g)) * np.array([w[i] * (c[i, 0] * vy + c[i, 1] * vz) / cs2 for i in range(9)])
    gpost = g - (g - geq) / a.tau_g + Gi                                   # eq (12)
    # force and eq (20)
    mu = mu_phi(phi); gy, gz = grad(phi)
    Fy, Fz = mu * gy, mu * gz                                              # eq (5)
    gry, grz = grad(rho * cs2)
    si = [w[i] * (1 + cu[i] / cs2 + cu[i] ** 2 / (2 * cs4) - u2 / (2 * cs2)) for i in range(9)]
    Fi = np.array([(1.0 - 1.0 / (2.0 * a.tau_f)) * (
        (c[i, 0] - uy) * (Fy * si[i] + (si[i] - w[i]) * gry) + (c[i, 1] - uz) * (Fz * si[i] + (si[i] - w[i]) * grz)) for i in range(9)])
    if a.tauvar:                                                           # TEMPORARY: tau_f(phi) = 1/2 + eta(phi) / (rho(phi) c_s^2)
        tauF = 0.5 + (phi * 0.5 + (1 - phi) * 0.007) / (rho * cs2)
        Fi = Fi / (1.0 - 1.0 / (2.0 * a.tau_f)) * (1.0 - 1.0 / (2.0 * tauF))
        fpost = f - (f - feq) / tauF + Fi
    else:
        fpost = f - (f - feq) / a.tau_f + Fi                               # eq (13)
    g = bounce(gpost, stream(gpost))
    if a.fbc == "bb":
        f = bounce(fpost, stream(fpost))
    elif a.fbc == "readC":                                                 # TEMPORARY: eq. (30) with ALL terms read as post-streaming values at time t, q = 1/2: f_i(x_f) = 1/3 [ f_i(x_ff) + f_i*(x_f) + f_i*(x_s) ]
        f = stream(fpost)
        for i in range(9):
            j = opp[i]
            fj_ff = np.roll(np.roll(fpost[j], -c[i, 0], axis=0), -c[i, 1], axis=1)         # f~_i* at x_ff = post-stream f_i*(x_f)
            if c[i, 1] > 0:
                f[i, 1:Ny + 1, 1] = (fpost[i, 1:Ny + 1, 1] + fj_ff[1:Ny + 1, 1] + fpost[j, 1:Ny + 1, 1]) / 3.0
                if c[i, 0] > 0: f[i, Ny, 1] = fpost[j, Ny, 1]
                if c[i, 0] < 0: f[i, 1, 1] = fpost[j, 1, 1]
            if c[i, 1] < 0:
                f[i, 1:Ny + 1, Nz] = (fpost[i, 1:Ny + 1, Nz] + fj_ff[1:Ny + 1, Nz] + fpost[j, 1:Ny + 1, Nz]) / 3.0
                if c[i, 0] > 0: f[i, Ny, Nz] = fpost[j, Ny, Nz]
                if c[i, 0] < 0: f[i, 1, Nz] = fpost[j, 1, Nz]
        for i in range(9):
            j = opp[i]
            fj_ff = np.roll(np.roll(fpost[j], -c[i, 0], axis=0), -c[i, 1], axis=1)
            if c[i, 0] > 0:
                f[i, 1, 1:Nz + 1] = (fpost[i, 1, 1:Nz + 1] + fj_ff[1, 1:Nz + 1] + fpost[j, 1, 1:Nz + 1]) / 3.0
                if c[i, 1] > 0: f[i, 1, Nz] = fpost[j, 1, Nz]
                if c[i, 1] < 0: f[i, 1, 1] = fpost[j, 1, 1]
            if c[i, 0] < 0:
                f[i, Ny, 1:Nz + 1] = (fpost[i, Ny, 1:Nz + 1] + fj_ff[Ny, 1:Nz + 1] + fpost[j, Ny, 1:Nz + 1]) / 3.0
                if c[i, 1] > 0: f[i, Ny, Nz] = fpost[j, Ny, Nz]
                if c[i, 1] < 0: f[i, Ny, 1] = fpost[j, Ny, 1]
    elif a.fbc == "paper":                                                 # TEMPORARY: eq. (30) as printed, q = 1/2, x_s value = reflected value:  f_i(x_f) = 1/3 f~_i(x_ff) + 2/3 f~_i*(x_f)
        f = stream(fpost)
        for i in range(9):
            j = opp[i]
            sy = np.roll(fpost[i], -c[i, 0], axis=0)                       # f~_i at (y + c_y, z)
            sz = np.roll(fpost[i], -c[i, 1], axis=1)                       # f~_i at (y, z + c_z)
            syz = np.roll(np.roll(fpost[i], -c[i, 0], axis=0), -c[i, 1], axis=1)   # f~_i at x_ff = x_f + e_i
            if c[i, 1] > 0:
                f[i, 1:Ny + 1, 1] = (1 / 3) * syz[1:Ny + 1, 2] + (2 / 3) * fpost[j, 1:Ny + 1, 1]
                if c[i, 0] > 0: f[i, Ny, 1] = fpost[j, Ny, 1]              # x_ff would be a ghost node: plain bounce-back
                if c[i, 0] < 0: f[i, 1, 1] = fpost[j, 1, 1]
            if c[i, 1] < 0:
                f[i, 1:Ny + 1, Nz] = (1 / 3) * syz[1:Ny + 1, Nz - 1] + (2 / 3) * fpost[j, 1:Ny + 1, Nz]
                if c[i, 0] > 0: f[i, Ny, Nz] = fpost[j, Ny, Nz]
                if c[i, 0] < 0: f[i, 1, Nz] = fpost[j, 1, Nz]
        for i in range(9):
            j = opp[i]
            syz = np.roll(np.roll(fpost[i], -c[i, 0], axis=0), -c[i, 1], axis=1)
            if c[i, 0] > 0:
                f[i, 1, 1:Nz + 1] = (1 / 3) * syz[2, 1:Nz + 1] + (2 / 3) * fpost[j, 1, 1:Nz + 1]
                if c[i, 1] > 0: f[i, 1, Nz] = fpost[j, 1, Nz]
                if c[i, 1] < 0: f[i, 1, 1] = fpost[j, 1, 1]
            if c[i, 0] < 0:
                f[i, Ny, 1:Nz + 1] = (1 / 3) * syz[Ny - 1, 1:Nz + 1] + (2 / 3) * fpost[j, Ny, 1:Nz + 1]
                if c[i, 1] > 0: f[i, Ny, Nz] = fpost[j, Ny, Nz]
                if c[i, 1] < 0: f[i, Ny, 1] = fpost[j, Ny, 1]
    else:                                                                  # TEMPORARY: eq. (30), q = 1/2:  f_i(x_f) = 2/3 f~_i(x_f) + 1/3 f~_i*(x_f)  (f~ = post-collision at the wall node)
        f = stream(fpost)
        for i in range(9):
            j = opp[i]
            if c[i, 1] > 0: f[i, :, 1] = (2 / 3) * fpost[i, :, 1] + (1 / 3) * fpost[j, :, 1]
            if c[i, 1] < 0: f[i, :, Nz] = (2 / 3) * fpost[i, :, Nz] + (1 / 3) * fpost[j, :, Nz]
        for i in range(9):
            j = opp[i]
            if c[i, 0] > 0: f[i, 1, :] = (2 / 3) * fpost[i, 1, :] + (1 / 3) * fpost[j, 1, :]
            if c[i, 0] < 0: f[i, Ny, :] = (2 / 3) * fpost[i, Ny, :] + (1 / 3) * fpost[j, Ny, :]
    phi_old = phi.copy(); uy_old = uy.copy(); uz_old = uz.copy()
    tot_old = phi[fluid].sum()
    phi = g.sum(axis=0)                                                    # eq (26)
    if not a.noclip:
        phi = np.clip(phi, 0.0, 1.0)
    tot_new = phi[fluid].sum()
    if not a.nomass and abs(tot_new - tot_old) > 1e-8:
        phi[fluid] *= tot_old / tot_new
    phi = set_solid(phi)
    rho = phi * rho_L + (1 - phi) * rho_G                                  # eq (4)
    mu = mu_phi(phi); gy, gz = grad(phi)
    Fy, Fz = mu * gy, mu * gz
    uy = zero_ghosts(sum(c[i, 0] * f[i] for i in range(9)) / (cs2 * rho) + Fy / (2 * rho))      # eq (28)
    uz = zero_ghosts(sum(c[i, 1] * f[i] for i in range(9)) / (cs2 * rho) + Fz / (2 * rho))
    gry, grz = grad(rho * cs2)
    p = f.sum(axis=0) + 0.5 * (uy * gry + uz * grz)                        # eq (27)
say("# done in %.0f s" % (time.time() - t0))
