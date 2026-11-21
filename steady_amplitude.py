"""Amplitude monitor for the herringbone runs: stops a run when the oscillation of the flow has reached a statistically steady state.

Signal: the global maximum of |u| (the same number as in the 'iter N ... max|u|=' log lines), sampled every AMP_STRIDE iterations.
Batch:  a block of AMP_BATCH iterations (many oscillation periods, and at least one grow-and-shrink cycle of the envelope).
        Amplitude of a batch  A_j = maximum of the Hilbert envelope (scipy.signal.hilbert) of the linearly detrended signal in the batch.
Stop:   as soon as the last AMP_NBATCH batches all start at or after iteration AMP_START and
        (max(A) - min(A)) / mean(A)  <  AMP_CUTOFF            over these batches.
A snapshot of the fields (fields3d_iter_NNNNNN.npz) is requested for the iteration after the decision, the run ends one iteration later.
The signal is not finite -> the run stops at once (reason 'non-finite').

Environment variables (all optional; without AMP_CUTOFF the monitor is switched off and only costs one comparison per iteration):
  AMP_CUTOFF  relative spread of the batch amplitudes below which the run stops, e.g. 0.02          (no default: off)
  AMP_STRIDE  sampling interval in iterations                                                       (default 10)
  AMP_BATCH   length of a batch in iterations                                                       (default 20000)
  AMP_NBATCH  number of consecutive batches that must agree                                         (default 4)
  AMP_START   first iteration whose batches count (spin-up is ignored)                              (default 40000)

Written to <images_dir>/amplitude_history.npz after every batch and at the stop:
  samples  (n, 3)  iteration, largest single velocity component max(|Ux|,|Uy|,|Uz|) (the 'max|u|' of the log; the stop rule uses this one),
                   largest Euclidean magnitude sqrt(Ux^2+Uy^2+Uz^2) over all cells;   batches  (m, 6)  first iteration, last iteration, min, max, amplitude, mean of each batch.
"""
from __future__ import annotations

import os

import numpy as np
from mpi4py import MPI
from scipy.signal import detrend, hilbert


class AmplitudeMonitor:
    def __init__(self, out_dir, cutoff=None, stride=10, batch=20000, n_batch=4, start=40000):
        self.out_dir, self.cutoff, self.stride, self.batch, self.n_batch, self.start = out_dir, cutoff, stride, batch, n_batch, start
        self.samples = []                 # (iteration, global max|u|)
        self.batches = []                 # (first it, last it, min, max, amplitude, mean)
        self._batch_index = 0
        self._batch_values = []
        self.snapshot_it = -1             # iteration at which the final fields are dumped (set when the criterion is met)
        self.stop = False
        self.reason = ""

    @classmethod
    def from_env(cls, out_dir):
        c = os.environ.get("AMP_CUTOFF")
        return cls(out_dir, cutoff=float(c) if c else None, stride=int(os.environ.get("AMP_STRIDE", 10)), batch=int(os.environ.get("AMP_BATCH", 20000)),
                   n_batch=int(os.environ.get("AMP_NBATCH", 4)), start=int(os.environ.get("AMP_START", 40000)))

    def record(self, iteration, u_ckl, comm):
        """Call at the end of every iteration (after iteration += 1) with the velocity array u_ckl[3, nx+2, ny+2, nz+2] of this rank. Collective: every rank must call it."""
        if self.cutoff is None:
            return
        if self.snapshot_it >= 0:
            if iteration > self.snapshot_it:          # the iteration with the final snapshot has run
                self.stop = True
                self._save(comm)
            return
        if iteration % self.stride:
            return
        ui = u_ckl[:, 1:-1, 1:-1, 1:-1]
        s = comm.allreduce(float(np.abs(ui).max()), op=MPI.MAX)                            # largest single component; the same value on every rank, so every rank takes the same decision
        s_euclid = comm.allreduce(float(np.sqrt((ui ** 2).sum(axis=0)).max()), op=MPI.MAX)   # largest Euclidean magnitude
        self.samples.append((iteration, s, s_euclid))                                      # the stop rule works on s, as before
        if not np.isfinite(s):
            self.reason = f"non-finite max|u| at iteration {iteration}"
            self.stop = True
            if comm.Get_rank() == 0:
                print(f"AMP STOP: {self.reason}", flush=True)
            self._save(comm)
            return
        b = iteration // self.batch
        if b != self._batch_index and self._batch_values:
            self._close_batch(comm, iteration)
        self._batch_index = b
        self._batch_values.append((iteration, s))

    def _close_batch(self, comm, iteration):
        its = [i for i, _ in self._batch_values]
        v = np.array([x for _, x in self._batch_values])
        env = np.abs(hilbert(detrend(v)))                  # envelope of the oscillation around its slow drift
        cut = max(1, len(v) // 10)                         # the first and last 10 % of the envelope are distorted by the transform
        amp = float(env[cut:-cut].max())
        self.batches.append((its[0], its[-1], float(v.min()), float(v.max()), amp, float(v.mean())))
        self._batch_values = []
        last = [bt for bt in self.batches if bt[0] >= self.start][-self.n_batch:]
        spread = None
        if len(last) == self.n_batch:
            a = np.array([bt[4] for bt in last])
            spread = float((a.max() - a.min()) / a.mean())
        rank = comm.Get_rank()
        if rank == 0:
            tail = f"spread of the last {self.n_batch} amplitudes {spread:.4f} (cutoff {self.cutoff})" if spread is not None else "not enough batches after AMP_START yet"
            print(f"AMP batch {len(self.batches):3d}: iterations {its[0]}-{its[-1]}  max|u| {v.min():.5f} .. {v.max():.5f}  amplitude {amp:.5f}  mean {v.mean():.5f}  {tail}", flush=True)
        self._save(comm)
        if spread is not None and spread < self.cutoff:
            self.reason = f"amplitude of the last {self.n_batch} batches constant within {spread:.4f} < {self.cutoff}"
            self.snapshot_it = iteration                       # = index of the next pass of the main loop, which then dumps the fields
            if comm.Get_rank() == 0:
                print(f"AMP STOP: {self.reason}; final snapshot at iteration {self.snapshot_it}, then the run ends", flush=True)

    def _save(self, comm):
        if comm.Get_rank() == 0:
            np.savez(os.path.join(self.out_dir, "amplitude_history.npz"), samples=np.array(self.samples), batches=np.array(self.batches))
