"""
Run on the cluster login node: cut the huge final-state file (fields_iter_XXXXXX.npz, phi and u_ckl for the whole domain)
down to a small file you can copy home.  Only numpy is needed.

    python3 thin_fields_system.py <fields_iter_XXXXXX.npz> <out.npz> [--u]

Output: phi (every 2nd cell in x, y and z, float16, ghost layer removed) and, with --u, u (3 components, same thinning).
The result has the same layout as the phi3d_iter_XXXXXX.npz snapshots that the simulation writes during the run,
so render3d_system.py reads both.
"""
import os
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
with_u = "--u" in sys.argv

d = np.load(src)
phi = d["phi"][1:-1:2, 1:-1:2, 1:-1:2].astype(np.float16)        # drop ghost layer, take every 2nd cell
keep = {"phi": phi}
if with_u:
    u = d["u_ckl"]
    keep["u"] = np.stack([u[k][1:-1:2, 1:-1:2, 1:-1:2] for k in range(3)]).astype(np.float16)
np.savez_compressed(out, **keep)
print("phi thinned shape", phi.shape, "->", out, "(%.1f MB)" % (os.path.getsize(out) / 1e6))
