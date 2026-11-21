"""The oscillation analysis of the W=14 amplitude run (job 8885554) in four separate figures (the composite xs1_amp_analysis.png split into its parts):
  xs1_osc_1_raw_signal.png           the raw signal max|u|: whole run and a zoom on six oscillations
  xs1_osc_2_frequency_analysis.png   Fourier transform of all samples after the start-up: the constituent frequencies
  xs1_osc_3_amplitude_analysis.png   one batch: straight line removed (detrending), Hilbert envelope, batch amplitude; then the amplitude of every batch
  xs1_osc_4_composite_on_raw.png     the composite of the constituent oscillations (mean + fundamental + second harmonic, fitted by least squares) superimposed on the raw signal
Run on the PC in herringbone\\run_3d\\u-oscillations:  python xs1_osc_parts.py [amplitude_history.npz]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import detrend, hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "amplitude_history.npz")
d = np.load(path)
S, B = d["samples"], d["batches"]
it, v = S[:, 0], S[:, 1]
STRIDE, BATCH, START = 10, 20000, 40000
MS = 1828.266 / 156012                                                    # physical time per iteration in ms
BLUE, RED, ORANGE = "#2171b5", "#d62728", "#ff7f0e"


def top_axis(ax):
    sec = ax.secondary_xaxis("top", functions=(lambda x: x * MS * 1e-3, lambda s: s / (MS * 1e-3)))
    sec.set_xlabel("physical time (s)")


# ---- 1: raw signal -----------------------------------------------------------------------------------------------------------------------------------------
fig, (ax, bx) = plt.subplots(2, 1, figsize=(15, 9), layout="constrained", height_ratios=[1.3, 1])
ax.plot(it, v, color=BLUE, lw=0.5)
ax.set(xlim=(0, it.max()), xlabel="iteration", ylabel="max|u| (lattice units)", title=f"Raw signal: W=14, max |u(i)| over all cells, sampling rate f=0.1Hz, {len(v)} samples")
top_axis(ax)
ax.grid(alpha=0.3)
a, b = 100000, 102050
mm = (it >= a) & (it < b)
bx.plot(it[mm], v[mm], "o-", color=BLUE, ms=3, lw=1)
bx.vlines(a + 341.5 * np.arange(0, 7), v[mm].min(), v[mm].max(), color="0.6", lw=0.8, ls="--")
bx.set(xlabel="iteration", ylabel="max|u|", title="Zoom: six oscillations, one every 341.5 iterations (4.0 ms); every dot is one sample")
bx.grid(alpha=0.3)
fig.savefig(os.path.join(HERE, "xs1_osc_1_raw_signal.png"), dpi=100)
plt.close(fig)


# ---- 2: frequency analysis ---------------------------------------------------------------------------------------------------------------------------------
def spectrum(xx):
    xx = detrend(xx)
    n = len(xx)
    return np.fft.rfftfreq(n, STRIDE), np.abs(np.fft.rfft(xx * np.hanning(n))) * 4 / n


mb = (it >= 100000) & (it < 120000)
ml = it >= START
fig, a2 = plt.subplots(figsize=(10, 5.8), layout="constrained")
f2, F2 = spectrum(v[ml])
ok = f2 > 1 / 700
a2.plot(1 / f2[ok], F2[ok], color=BLUE, lw=1.2)
for k, name in ((341.5, "fundamental"), (170.75, "second harmonic")):
    i = np.argmin(np.abs(1 / f2[ok] - k))
    j = i - 3 + np.argmax(F2[ok][i - 3:i + 4])
    per = 1 / f2[ok][j]
    a2.annotate(f"{per:.1f} iterations = {per * MS:.2f} ms ({name})\namplitude {F2[ok][j]:.5f}", (per, F2[ok][j]), textcoords="offset points", xytext=(10, -6), fontsize=10)
a2.set(xlim=(150, 700), xlabel="period (iterations)", ylabel="amplitude (lattice units)",
       title=f"Fourier transform of all {ml.sum()} samples after iteration {START}\none peak at 341.5 and its second harmonic at 170.7")
a2.grid(alpha=0.3)
fig.savefig(os.path.join(HERE, "xs1_osc_2_frequency_analysis.png"), dpi=100)
plt.close(fig)

# ---- 3: amplitude analysis ---------------------------------------------------------------------------------------------------------------------------------
tb, xb = it[mb], v[mb]
line = np.polyval(np.polyfit(tb, xb, 1), tb)
xd = xb - line
env = np.abs(hilbert(xd))
cut = max(1, len(xd) // 10)
amp = env[cut:-cut].max()
fig, axs = plt.subplots(3, 1, figsize=(15, 13), layout="constrained")
axs[0].plot(tb, xb, color=BLUE, lw=0.5, label="batch 6: max|u|")
axs[0].plot(tb, line, color=RED, lw=2, label="best-fit straight line (subtracted)")
axs[0].set(xlim=(tb[0], tb[-1]), ylabel="max|u|", title="Step 1: batch 6 (iterations 100000-119990) and its best-fit straight line")
axs[0].legend(loc="upper right")
axs[1].plot(tb, xd, color=BLUE, lw=0.5, label="detrended signal")
axs[1].plot(tb, env, color=ORANGE, lw=1.5, label="Hilbert envelope")
axs[1].axvspan(tb[0], tb[cut], color="0.85"); axs[1].axvspan(tb[-cut], tb[-1], color="0.85")
axs[1].axhline(amp, color=RED, lw=1.6, label=f"batch amplitude = maximum of the envelope in the white part = {amp:.5f}")
axs[1].set(xlim=(tb[0], tb[-1]), ylabel="max|u| minus the line", title="Step 2: detrended signal, its Hilbert envelope, and the batch amplitude (the first and last 10 % are left out)")
axs[1].legend(loc="lower left", fontsize=9)
ba = B[:, 4]
x0 = np.arange(1, len(ba) + 1)
axs[2].bar(x0, ba, color=["0.75" if B[k, 0] < START else BLUE for k in range(len(ba))])
for k in range(2, len(ba)):
    axs[2].text(x0[k], ba[k], f"{ba[k]:.5f}", ha="center", va="bottom", fontsize=9)
axs[2].text(1.5, ba[2:].max() * 1.15, f"batches 1 and 2: amplitude {ba[0]:.3f} and {ba[1]:.3f} (start-up, off the scale)", ha="center", fontsize=9)
last = ba[-4:]
axs[2].axhspan(last.mean() * 0.975, last.mean() * 1.025, color=RED, alpha=0.18, label="±2.5 % around the mean of the last 4 batches (the 5 % spread the stop rule allows)")
axs[2].set(xlabel="batch number (grey = before iteration 40000, ignored)", ylabel="batch amplitude", ylim=(0, ba[2:].max() * 1.25),
           title=f"Step 3: amplitude of every batch; the last 4 batches (6-9) agree within {(last.max() - last.min()) / last.mean() * 100:.1f} % -> stop")
axs[2].set_xticks(x0)
axs[2].legend(loc="upper right", fontsize=9)
axs[2].grid(alpha=0.3, axis="y")
fig.savefig(os.path.join(HERE, "xs1_osc_3_amplitude_analysis.png"), dpi=100)
plt.close(fig)

# ---- 4: composite of the constituent oscillations on the raw signal ------------------------------------------------------------------------------------------
fit = it >= 100000                                                        # the slow ringing has died out here: constant amplitudes
t, y = it[fit], v[fit]
t0 = t[0]


def model(p, t):
    f1 = p[0]
    return p[1] + p[2] * np.cos(2 * np.pi * f1 * (t - t0)) + p[3] * np.sin(2 * np.pi * f1 * (t - t0)) + p[4] * np.cos(4 * np.pi * f1 * (t - t0)) + p[5] * np.sin(4 * np.pi * f1 * (t - t0))


p0 = [1 / 341.5, y.mean(), 0.001, 0.001, 0.0005, 0.0005]
sol = least_squares(lambda p: model(p, t) - y, p0, x_scale=[1e-6, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
p = sol.x
comp = model(p, t)
resid = y - comp
A1, A2 = np.hypot(p[2], p[3]), np.hypot(p[4], p[5])
r2 = 1 - resid.var() / y.var()
print(f"fitted fundamental period {1 / p[0]:.2f} iterations, amplitude {A1:.5f}; second harmonic amplitude {A2:.5f}; mean {p[1]:.5f}; share of the variance explained {r2 * 100:.1f} %")
fig, axs = plt.subplots(3, 1, figsize=(15, 12), layout="constrained", height_ratios=[1.1, 1.1, 0.8])
axs[0].plot(t, y, color=BLUE, lw=0.5, label="raw signal")
axs[0].plot(t, comp, color=RED, lw=0.5, label="composite: mean + fundamental + second harmonic")
axs[0].set(xlim=(t[0], t[-1]), ylabel="max|u|", title=f"Composite of the constituent oscillations superimposed on the raw signal, iterations 100000-180000 (W=14)")
axs[0].legend(loc="upper right")
axs[0].grid(alpha=0.3)
zz = (t >= 100000) & (t < 102050)
axs[1].plot(t[zz], y[zz], "o", color=BLUE, ms=3.5, label="raw signal (one dot per sample)")
axs[1].plot(t[zz], comp[zz], color=RED, lw=1.8, label=f"composite (explains {r2 * 100:.0f} % of the variance)")
axs[1].plot(t[zz], p[1] + p[2] * np.cos(2 * np.pi * p[0] * (t[zz] - t0)) + p[3] * np.sin(2 * np.pi * p[0] * (t[zz] - t0)), color="#2ca02c", lw=1.2, ls="--",
            label=f"mean + fundamental alone (period {1 / p[0]:.1f} iterations, amplitude {A1:.5f})")
axs[1].set(xlabel="iteration", ylabel="max|u|", title="Zoom on six oscillations")
axs[1].legend(loc="upper right", fontsize=9)
axs[1].grid(alpha=0.3)
axs[2].plot(t, resid, color="0.3", lw=0.5)
axs[2].set(xlabel="iteration", ylabel="raw minus composite", xlim=(t[0], t[-1]), title=f"What the composite does not contain (rms {resid.std():.5f}, the oscillation itself has rms {y.std():.5f})")
axs[2].grid(alpha=0.3)
fig.savefig(os.path.join(HERE, "xs1_osc_4_composite_on_raw.png"), dpi=100)
print("wrote xs1_osc_1_raw_signal.png, xs1_osc_2_frequency_analysis.png, xs1_osc_3_amplitude_analysis.png, xs1_osc_4_composite_on_raw.png")
