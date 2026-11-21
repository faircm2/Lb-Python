"""How the oscillation of the flow is tracked and analysed (W=14 amplitude run, job 8885554, stopped at iteration 180000 by the amplitude monitor).
Signal: global max|u| (largest single velocity component over all cells and ranks), sampled every 10 iterations, saved by steady_amplitude.py in amplitude_history.npz.
Run on the PC in herringbone\\run_3d\\u-oscillations:  python xs1_amp_analysis.py [amplitude_history.npz] [label]    ->  xs1_amp_analysis.png
Panels
  A  the whole signal, the slow wave (low-pass, periods > 3000 iterations) in red, the batches (20000 iterations) as grey bands, the ignored start-up phase hatched
  B  a few oscillations in detail (the samples, one every 10 iterations)
  C  one batch: detrended signal, Hilbert envelope, batch amplitude = maximum of the envelope without the first and last 10 %
  D  spectra (Fourier transform of the detrended signal): one batch (2000 samples) against all samples after iteration 40000 (14000 samples)
  E  amplitude of every batch against the 5 % band the stop rule uses
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, find_peaks, hilbert, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "amplitude_history.npz")
label = sys.argv[2] if len(sys.argv) > 2 else "W=14, job 8885554"
d = np.load(path)
S, B = d["samples"], d["batches"]
it, v = S[:, 0], S[:, 1]
STRIDE, BATCH, START = 10, 20000, 40000
MS = 1828.266 / 156012                                                   # physical time per iteration in ms (tau_f 0.6, D 1.2 mm / 64 cells, nu 1e-6)

fig = plt.figure(figsize=(13, 21), layout="constrained")
gs = fig.add_gridspec(5, 2, height_ratios=[1.15, 0.9, 1.0, 1.15, 0.8])

# ---- A: whole signal and the slow wave ---------------------------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
for b in range(int(it.max() // BATCH) + 1):
    if b % 2 == 0:
        ax.axvspan(b * BATCH, (b + 1) * BATCH, color="0.93", lw=0)
ax.axvspan(0, START, facecolor="none", edgecolor="0.6", hatch="//", lw=0)
ax.plot(it, v, color="#2171b5", lw=0.5, label="max|u| (one sample every 10 iterations)")
sos = butter(4, (1 / 3000) / (0.5 / STRIDE), output="sos")
slow = np.full(len(v), np.nan)
m = it >= STRIDE * 10
slow[m] = sosfiltfilt(sos, v[m])
ax.plot(it, slow, color="#d62728", lw=2.4, label="slow wave: low-pass of the signal, periods longer than 3000 iterations (35 ms)")
ax.set(xlim=(0, it.max()), ylim=(0.20, 0.285), xlabel="iteration", ylabel="max|u| (lattice units)",
       title=f"A  Signal over the whole run ({label}): grey bands = batches of 20000 iterations, hatched = ignored start-up (before {START})")
sec = ax.secondary_xaxis("top", functions=(lambda x: x * MS * 1e-3, lambda s: s / (MS * 1e-3)))
sec.set_xlabel("physical time (s)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

# ---- B: zoom on a few oscillations -------------------------------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
a, b = 100000, 102050
mm = (it >= a) & (it < b)
ax.plot(it[mm], v[mm], "o-", color="#2171b5", ms=3, lw=1)
x = detrend(v[(it >= 100000) & (it < 120000)])
F = np.abs(np.fft.rfft(x * np.hanning(len(x))))
fr = np.fft.rfftfreq(len(x), STRIDE)
pk = a + 341.5 * np.arange(0, 7)
ax.vlines(pk, v[mm].min(), v[mm].max(), color="0.6", lw=0.8, ls="--")
ax.set(xlabel="iteration", ylabel="max|u|", title="B  Six oscillations: one every 341.5 iterations (4.0 ms)")
ax.grid(alpha=0.3)

# ---- C: one batch: detrend, envelope, amplitude ------------------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
mb = (it >= 100000) & (it < 120000)
tb, xb = it[mb], detrend(v[mb])
env = np.abs(hilbert(xb))
cut = max(1, len(xb) // 10)
amp = env[cut:-cut].max()
ax.plot(tb, xb, color="#2171b5", lw=0.5, label="detrended signal (straight line removed)")
ax.plot(tb, env, color="#ff7f0e", lw=1.4, label="Hilbert envelope")
ax.axvspan(tb[0], tb[cut], color="0.85"); ax.axvspan(tb[-cut], tb[-1], color="0.85")
ax.axhline(amp, color="#d62728", lw=1.6, label=f"batch amplitude = max of the envelope inside the white part = {amp:.5f}")
ax.set(xlabel="iteration", ylabel="max|u| minus its trend", xlim=(tb[0], tb[-1]), title="C  Batch 6 (100000-119990): how its amplitude is computed")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.3)

# ---- D: spectra --------------------------------------------------------------------------------------------------------------------------------------------
def spectrum(xx):
    xx = detrend(xx)
    n = len(xx)
    return np.fft.rfftfreq(n, STRIDE), np.abs(np.fft.rfft(xx * np.hanning(n))) * 4 / n

ax = fig.add_subplot(gs[2, 0])
f1, F1 = spectrum(v[mb])
ok = f1 > 1 / 700
ax.stem(1 / f1[ok][::-1], F1[ok][::-1], basefmt=" ", linefmt="C0-", markerfmt="C0o")
ax.set(xlim=(150, 700), xlabel="period (iterations)", ylabel="amplitude (lattice units)",
       title="D1  Spectrum of ONE batch (2000 samples, bin width 5e-5 per iteration)\nthe peak sits between two bins and shows up in both (58: 344.8, 59: 339.0)")
ax.grid(alpha=0.3)
ax = fig.add_subplot(gs[2, 1])
ml = it >= START
f2, F2 = spectrum(v[ml])
ok = f2 > 1 / 700
ax.plot(1 / f2[ok], F2[ok], color="#2171b5", lw=1.2)
pk2, _ = find_peaks(F2, height=F2.max() * 0.15)
for i in pk2:
    ax.annotate(f"{1 / f2[i]:.1f}", (1 / f2[i], F2[i]), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set(xlim=(150, 700), xlabel="period (iterations)", ylabel="amplitude (lattice units)",
       title=f"D2  Spectrum of all {ml.sum()} samples after {START} (7x finer)\none peak at 341.5 and its second harmonic at 170.7")
ax.grid(alpha=0.3)
print("long-window peaks (period its, amplitude):", [(round(1 / f2[i], 1), round(float(F2[i]), 5)) for i in pk2])

# ---- E: batch amplitudes -----------------------------------------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[3, :])
ba = B[:, 4]
x0 = np.arange(1, len(ba) + 1)
ax.bar(x0, ba, color=["0.75" if B[k, 0] < START else "#2171b5" for k in range(len(ba))])
for k in range(2, len(ba)):
    ax.text(x0[k], ba[k], f"{ba[k]:.5f}", ha="center", va="bottom", fontsize=8.5)
ax.text(1.5, ba[2:].max() * 1.15, f"batches 1 and 2: amplitude {ba[0]:.3f} and {ba[1]:.3f} (start-up, off the scale)", ha="center", fontsize=8.5)
last = ba[-4:]
mean = last.mean()
ax.axhspan(mean * 0.975, mean * 1.025, color="#d62728", alpha=0.18, label=f"±2.5 % around the mean of the last 4 batches (= the 5 % spread the stop rule allows)")
ax.set(xlabel="batch number (grey = before iteration 40000, ignored)", ylabel="batch amplitude", ylim=(0, ba[2:].max() * 1.25),
       title=f"E  Batch amplitudes: the last 4 batches (6-9) agree within {(last.max() - last.min()) / mean * 100:.1f} % -> stop")
ax.set_xticks(x0)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3, axis="y")

# ---- F: what the slow wave looks like after the start-up ---------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[4, :])
mk = it >= START
pks, _ = find_peaks(slow[mk], distance=400, prominence=0.0003)
tp, yp = it[mk][pks], slow[mk][pks]
sel = tp < 100000
tp, yp = tp[sel], yp[sel]
sp = np.diff(tp)
PER = float(np.median(sp[sp > 6000]))                                       # the spacing of the big peaks (the small ones in between are the second harmonic, half the period)
ax.plot(tp, yp, "kv", ms=7, label=f"peaks of the slow wave, spacing about {PER:.0f} iterations ({PER * MS:.0f} ms)")
ax.plot(it[mk], v[mk], color="#2171b5", lw=0.4, alpha=0.6, label="max|u|")
ax.plot(it[mk], slow[mk], color="#d62728", lw=2.2, label="slow wave (periods > 3000 iterations)")
ax.set(xlabel="iteration", ylabel="max|u|", ylim=(v[mk].min() - 0.001, v[mk].max() + 0.001),
       title=f"F  The slow wave after the start-up: a damped oscillation of period about {PER:.0f} iterations that has died out by about iteration 100000\n(from 80000 on it stays within {np.nanmax(slow[it >= 80000]) - np.nanmin(slow[it >= 80000]):.4f}, much smaller than the fast oscillation)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
print(f"slow wave after 80000: min {np.nanmin(slow[it >= 80000]):.5f} max {np.nanmax(slow[it >= 80000]):.5f}")

out = os.path.join(HERE, "xs1_amp_analysis.png")
fig.savefig(out, dpi=100)
print("wrote", out)
