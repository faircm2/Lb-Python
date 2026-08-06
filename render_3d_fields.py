"""
Render 3D visualizations from the fields_iter_*.npz snapshots dumped by
Cap02.py at each iterationsOfInterest checkpoint.

Requires: pip install pyvista

Usage:
    python render_3d_fields.py path/to/fields_iter_00500.npz --out_dir renders/
"""
import argparse
import os

import numpy as np
import pyvista as pv


def load_snapshot(npz_path):
    data = np.load(npz_path)
    return {
        "phi": data["phi"],
        "u_ckl": data["u_ckl"],
        "chemical_potential": data["chemical_potential"],
        "zhang_surface_tension_force": data["zhang_surface_tension_force"],
    }


def make_grid(shape):
    grid = pv.ImageData()
    grid.dimensions = shape
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    return grid


def render_phi_isosurface(phi, out_path, iso_value=0.5):
    """Interface shape in 3D - the phase boundary where phi crosses 0.5."""
    grid = make_grid(phi.shape)
    grid.point_data["phi"] = phi.flatten(order="F")

    contour = grid.contour(isosurfaces=[iso_value], scalars="phi")

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(contour, color="steelblue", smooth_shading=True)
    pl.add_axes()
    pl.camera_position = "iso"
    pl.screenshot(out_path)
    pl.close()


def render_scalar_volume(field, field_name, out_path, cmap="viridis"):
    grid = make_grid(field.shape)
    grid.point_data[field_name] = field.flatten(order="F")

    pl = pv.Plotter(off_screen=True)
    pl.add_volume(grid, scalars=field_name, cmap=cmap, opacity="sigmoid")
    pl.add_axes()
    pl.camera_position = "iso"
    pl.screenshot(out_path)
    pl.close()


def render_vector_glyphs(vec_field, field_name, out_path, stride=6, factor=2.0):
    """vec_field shape: (3, X, Y, Z). Subsampled so glyphs aren't overwhelming."""
    Xn, Yn, Zn = vec_field.shape[1:]
    grid = make_grid((Xn, Yn, Zn))

    vectors = np.stack([vec_field[0], vec_field[1], vec_field[2]], axis=-1)
    vectors = vectors.reshape(-1, 3, order="F")
    grid.point_data[field_name] = vectors
    grid.set_active_vectors(field_name)

    sampled = grid.extract_points(
        np.arange(0, grid.n_points, stride), adjacent_cells=False
    )
    glyphs = sampled.glyph(orient=field_name, scale=field_name, factor=factor)

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(glyphs, cmap="viridis")
    pl.add_axes()
    pl.camera_position = "iso"
    pl.screenshot(out_path)
    pl.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="path to a single fields_iter_NNNNN.npz file")
    parser.add_argument("--out_dir", default=".", help="where to save the rendered PNGs")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fields = load_snapshot(args.npz_path)
    tag = os.path.splitext(os.path.basename(args.npz_path))[0]

    render_phi_isosurface(fields["phi"], os.path.join(args.out_dir, f"{tag}_phi_isosurface.png"))
    render_scalar_volume(fields["chemical_potential"], "chemical_potential",
                          os.path.join(args.out_dir, f"{tag}_chem_pot_volume.png"))
    render_vector_glyphs(fields["u_ckl"], "u_ckl",
                          os.path.join(args.out_dir, f"{tag}_u_ckl_glyphs.png"))
    render_vector_glyphs(fields["zhang_surface_tension_force"], "surface_tension_force",
                          os.path.join(args.out_dir, f"{tag}_surface_tension_glyphs.png"))

    print(f"Saved renders for {tag} to {args.out_dir}")


if __name__ == "__main__":
    main()
