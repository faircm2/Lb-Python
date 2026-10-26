"""
Read the corner crash buffer (ring/corner_y<y>.npy, one file per rank touching the high-x wall / front wall y=0 / floor z=0)
and print, for the last frames, where |u| is largest (global x, y, z) and what phi is there; find the first non-finite frame.

Layout of each file: shape (CORNER_N, 5, local_Xn, local_Yn, local_Zn), float32
    channel 0 = phi, 1..3 = u_x, u_y, u_z, channel 4 [0,0,0] = iteration number stored in that slot (-1 = empty)
The blocks are the last x block (x = Xn - local_Xn .. Xn-1), z = 0 .. local_Zn-1, y = y_offset .. y_offset+local_Yn-1.

usage:   python3 corner_analyze_system.py <ring_dir> [out.npz] [Xn]
Only numpy is needed. out.npz keeps the last frames of the assembled corner block (phi, u) for plotting on your PC.
"""
import glob
import os
import re
import sys

import numpy as np

ring_dir = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "corner_summary.npz"
Xn = int(sys.argv[3]) if len(sys.argv) > 3 else 768

files = {}
for f in glob.glob(os.path.join(ring_dir, "corner_y*.npy")):
    files[int(re.search(r"corner_y(\d+)\.npy$", f).group(1))] = np.load(f, mmap_mode="r")
ys = sorted(files)
a0 = files[ys[0]]
n_slot, _, lX, lY, lZ = a0.shape
x0 = Xn - lX
Y = ys[-1] + lY
print("corner files: %d (y offsets %s), local block %dx%dx%d, assembled block x %d..%d, y 0..%d, z 0..%d" % (len(ys), ys, lX, lY, lZ, x0, Xn - 1, Y - 1, lZ - 1))

slot_iter = np.array([int(a0[s, 4, 0, 0, 0]) for s in range(n_slot)])
valid = np.where(slot_iter >= 0)[0]
order = valid[np.argsort(slot_iter[valid])]
iters = slot_iter[order]
print("frames available: %d, iterations %d .. %d" % (len(iters), iters[0], iters[-1]))


def frame(slot):
    phi = np.zeros((lX, Y, lZ), np.float32)
    u = np.zeros((3, lX, Y, lZ), np.float32)
    for y0, arr in files.items():
        blk = np.asarray(arr[slot])
        phi[:, y0:y0 + lY] = blk[0]
        u[:, :, y0:y0 + lY] = blk[1:4]
    return phi, u


def stats(phi, u):
    fin = np.isfinite(phi) & np.isfinite(u).all(axis=0)
    sp = np.sqrt((np.where(fin, u, 0.0) ** 2).sum(axis=0))
    ix, iy, iz = np.unravel_index(sp.argmax(), sp.shape)
    pz = np.where(np.isfinite(phi), phi, 1.0)
    return sp.max(), x0 + ix, iy, iz, float(pz[ix, iy, iz]), float(pz.min()), int((~fin).sum())


print("\n%7s | %10s %5s %4s %4s %6s | %9s | non-finite" % ("iter", "max|u|", "x", "y", "z", "phi", "phi min"))
step = max(1, len(order) // 40)
rows = sorted(set(list(range(0, len(order), step)) + list(range(max(0, len(order) - 12), len(order)))))
keep = {}
for k in rows:
    phi, u = frame(order[k])
    s = stats(phi, u)
    print("%7d | %10.3e %5d %4d %4d %6.3f | %9.4f | %d" % (iters[k], s[0], s[1], s[2], s[3], s[4], s[5], s[6]))
    if k >= len(order) - 5 or k % max(1, len(order) // 10) == 0:
        keep["it%07d" % iters[k]] = np.concatenate([phi[None], u]).astype(np.float32)
np.savez_compressed(out, x0=x0, **keep)
print("\nsaved %d frames to %s (%.1f MB)" % (len(keep), out, os.path.getsize(out) / 1e6))

first_bad = None
for k in range(len(order)):
    if sum(int((~np.isfinite(np.asarray(arr[order[k], :4]))).sum()) for arr in files.values()) > 0:
        first_bad = k
        break
print("\n=== first frame with non-finite values: %s" % ("none" if first_bad is None else "iteration %d" % iters[first_bad]))
if first_bad is not None:
    for k in range(max(0, first_bad - 12), first_bad + 1):
        phi, u = frame(order[k])
        fin = np.isfinite(phi) & np.isfinite(u).all(axis=0)
        bx, by, bz = np.where(~fin)
        s = stats(phi, u)
        print("it %d: max|u| %.3e at (x=%d,y=%d,z=%d) phi %.3f | phi min %.4f | bad cells: %d %s"
              % (iters[k], s[0], s[1], s[2], s[3], s[4], s[5], s[6],
                 "first (x=%d,y=%d,z=%d)" % (x0 + bx[0], by[0], bz[0]) if len(bx) else ""))
    phi, u = frame(order[first_bad])
    keep["bad_it%07d" % iters[first_bad]] = np.concatenate([phi[None], u]).astype(np.float32)
    np.savez_compressed(out, x0=x0, **keep)
    print("re-saved to %s" % out)
