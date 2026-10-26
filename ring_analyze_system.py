"""
Read the rolling crash buffer (ring/ring_y<y>_z<z>.npy, one file per rank on the mid-x plane) of a run
and print, for the frames just before the end, where the velocity is largest and what phi does.

Layout of each file: shape (RING_N, 5, local_Yn, local_Zn), float32
    channel 0 = phi, 1..3 = u_x, u_y, u_z, channel 4 [0,0] = iteration number stored in that slot (-1 = empty)

usage:   python3 ring_analyze_claude.py <ring_dir> [out.npz] [last_n_frames]
Only numpy is needed. The compact out.npz keeps ~25 frames (phi and u) for plotting on your PC.
"""
import glob
import os
import re
import sys

import numpy as np

ring_dir = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "ring_summary.npz"
last_n = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

files = {}
for f in glob.glob(os.path.join(ring_dir, "ring_y*_z*.npy")):
    m = re.search(r"ring_y(\d+)_z(\d+)\.npy$", f)
    files[(int(m.group(1)), int(m.group(2)))] = np.load(f, mmap_mode="r")
keys = sorted(files)
lY, lZ = files[keys[0]].shape[2], files[keys[0]].shape[3]
Y = max(k[0] for k in keys) + lY
Z = max(k[1] for k in keys) + lZ
print("ring files: %d, local block %dx%d, plane %dx%d" % (len(keys), lY, lZ, Y, Z))

ring_n = files[keys[0]].shape[0]
slot_iter = np.array([int(files[keys[0]][s, 4, 0, 0]) for s in range(ring_n)])
valid = np.where(slot_iter >= 0)[0]
order = valid[np.argsort(slot_iter[valid])]          # slots sorted by iteration
order = order[-last_n:]
iters = slot_iter[order]
print("frames available: %d, iterations %d .. %d" % (len(iters), iters[0], iters[-1]))


def frame(slot):
    phi = np.zeros((Y, Z), np.float32)
    u = np.zeros((3, Y, Z), np.float32)
    for (y0, z0), arr in files.items():
        blk = np.asarray(arr[slot])
        phi[y0:y0 + lY, z0:z0 + lZ] = blk[0]
        u[:, y0:y0 + lY, z0:z0 + lZ] = blk[1:4]
    return phi, u


def stats(phi, u):
    fin_u = np.isfinite(u).all(axis=0)
    fin = np.isfinite(phi) & fin_u
    nbad = int((~fin).sum())
    sp = np.sqrt((np.where(fin, u, 0.0) ** 2).sum(axis=0))
    iy, iz = np.unravel_index(sp.argmax(), sp.shape)
    pz = np.where(np.isfinite(phi), phi, 1.0)
    c = [(slice(0, 40), slice(0, 30)), (slice(Y - 40, Y), slice(0, 30))]
    cphi = [float(pz[a, b].min()) for a, b in c]
    cu = max(float(sp[a, b].max()) for a, b in c)
    bad_loc = ""
    if nbad:
        by, bz = np.where(~fin)
        bad_loc = "first bad cell (y=%d,z=%d)" % (by[0], bz[0])
    return sp.max(), iy, iz, float(pz[iy, iz]), cphi, cu, nbad, bad_loc


print("\n%7s | %10s %4s %4s %6s | corner phi L / R  | corner max|u| | non-finite" % ("iter", "max|u|", "y", "z", "phi"))
step = max(1, len(order) // 40)
rows = list(range(0, len(order), step)) + list(range(max(0, len(order) - 12), len(order)))
seen = set()
keep = {}
for k in rows:
    if k in seen:
        continue
    seen.add(k)
    phi, u = frame(order[k])
    s = stats(phi, u)
    print("%7d | %10.3e %4d %4d %6.3f | %7.4f / %7.4f | %13.3e | %d %s" % (iters[k], s[0], s[1], s[2], s[3], s[4][0], s[4][1], s[5], s[6], s[7]))
    if k % max(1, len(order) // 20) == 0 or k >= len(order) - 5:
        keep["it%07d" % iters[k]] = np.concatenate([phi[None], u]).astype(np.float32)
np.savez_compressed(out, **keep)
print("\nsaved %d frames to %s (%.1f MB)" % (len(keep), out, os.path.getsize(out) / 1e6))


# ---------------------------------------------------------------------------------------------
# where does the first non-finite value appear?  (scan all frames, then show the frames around it)
# ---------------------------------------------------------------------------------------------
def count_bad(slot):
    nb = 0
    for arr in files.values():
        nb += int((~np.isfinite(np.asarray(arr[slot, :4]))).sum())
    return nb


first_bad = None
for k in range(len(order)):
    if count_bad(order[k]) > 0:
        first_bad = k
        break
print("\n=== first frame with non-finite values: %s" % ("none" if first_bad is None else "iteration %d" % iters[first_bad]))
if first_bad is not None:
    phi, u = frame(order[first_bad])
    fin = np.isfinite(phi) & np.isfinite(u).all(axis=0)
    by, bz = np.where(~fin)
    print("bad cells: %d, bounding box y %d..%d, z %d..%d" % (len(by), by.min(), by.max(), bz.min(), bz.max()))
    print("first bad cells (y,z):", list(zip(by[:10].tolist(), bz[:10].tolist())))
    for k in range(max(0, first_bad - 12), first_bad + 1):
        phi, u = frame(order[k])
        fin = np.isfinite(phi) & np.isfinite(u).all(axis=0)
        sp = np.sqrt((np.where(fin, u, 0.0) ** 2).sum(axis=0))
        iy, iz = np.unravel_index(sp.argmax(), sp.shape)
        pz = np.where(np.isfinite(phi), phi, 0.5)
        pmin = np.unravel_index(pz.argmin(), pz.shape)
        pmax = np.unravel_index(pz.argmax(), pz.shape)
        print("it %d: max|u| %.3e at (%d,%d) phi %.3f | phi min %.4f at (%d,%d), max %.4f at (%d,%d) | phi<0: %d, phi>1: %d | bad: %d"
              % (iters[k], sp.max(), iy, iz, pz[iy, iz], pz.min(), pmin[0], pmin[1], pz.max(), pmax[0], pmax[1],
                 int((pz < 0).sum()), int((pz > 1).sum()), int((~fin).sum())))
    for k in range(max(0, first_bad - 6), first_bad + 1):
        phi, u = frame(order[k])
        keep["it%07d" % iters[k]] = np.concatenate([phi[None], u]).astype(np.float32)
    np.savez_compressed(out, **keep)
    print("re-saved %d frames to %s (%.1f MB)" % (len(keep), out, os.path.getsize(out) / 1e6))
