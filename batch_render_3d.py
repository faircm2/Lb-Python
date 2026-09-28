"""
Batch-render 3D views from the fields_iter_*.npz dumps of a completed run,
reusing the render functions in render_3d_fields.py.

Per selected snapshot it writes four PNGs into <out_dir>:
    <tag>_phi_isosurface.png   - phase boundary (phi=0.5), coloured by height
    <tag>_phi_volume.png       - liquid body solid / gas see-through
    <tag>_chempot_volume.png   - chemical_potential volume render
    <tag>_velocity_glyphs.png  - subsampled u_ckl arrow field

Default selection is the first dump plus the last four (iter 0, 11040, 11280,
11520, 12000 for the NEMO2 Cap02 run). Override with --iters or --all.

Usage:
    python batch_render_3d.py RESULTS_DIR [--out_dir DIR] [--all | --iters 0 12000 ...]
"""
import argparse
import os
import time
import glob
import re

from render_3d_fields import (
    load_snapshot,
    render_phi_isosurface,
    render_phi_volume,
    render_scalar_volume,
    render_vector_glyphs,
)


def iter_of(path):
    m = re.search(r"fields_iter_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def select(npz_paths, iters, take_all):
    npz_paths = sorted(npz_paths, key=iter_of)
    if take_all:
        return npz_paths
    if iters:
        wanted = set(iters)
        return [p for p in npz_paths if iter_of(p) in wanted]
    # default: first + last four
    return npz_paths[:1] + npz_paths[-4:]


def render_one(npz_path, out_dir):
    fields = load_snapshot(npz_path)
    tag = os.path.splitext(os.path.basename(npz_path))[0]

    jobs = [
        ("phi_isosurface", lambda p: render_phi_isosurface(fields["phi"], p)),
        ("phi_volume",      lambda p: render_phi_volume(fields["phi"], p)),
        ("chempot_volume",  lambda p: render_scalar_volume(
            fields["chemical_potential"], "chemical_potential", p)),
        ("velocity_glyphs", lambda p: render_vector_glyphs(
            fields["u_ckl"], "u_ckl", p)),
    ]

    for name, fn in jobs:
        out_path = os.path.join(out_dir, f"{tag}_{name}.png")
        t0 = time.time()
        try:
            fn(out_path)
            print(f"    {name:16s} {time.time() - t0:6.1f}s  -> {os.path.basename(out_path)}")
        except Exception as e:
            print(f"    {name:16s} FAILED: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="folder containing fields_iter_*.npz")
    ap.add_argument("--out_dir", default=None,
                    help="output folder (default: <results_dir>/renders_3d)")
    ap.add_argument("--all", action="store_true", help="render every snapshot")
    ap.add_argument("--iters", type=int, nargs="+", default=None,
                    help="explicit iteration numbers to render")
    args = ap.parse_args()

    npz_paths = glob.glob(os.path.join(args.results_dir, "fields_iter_*.npz"))
    if not npz_paths:
        raise SystemExit(f"no fields_iter_*.npz found in {args.results_dir}")

    chosen = select(npz_paths, args.iters, args.all)
    out_dir = args.out_dir or os.path.join(args.results_dir, "renders_3d")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{len(chosen)} snapshot(s) -> {out_dir}")
    t_start = time.time()
    for i, npz_path in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {os.path.basename(npz_path)}")
        render_one(npz_path, out_dir)
    print(f"done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
