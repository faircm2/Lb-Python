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
    """
    Interface shape in 3D - the phase boundary where phi crosses 0.5.
    Colored by height (z) with the same RdBu palette used for the 2D phi
    plots, and semi-transparent, so meniscus curvature near the walls is
    visible through the surface instead of only as a silhouette.
    """
    grid = make_grid(phi.shape)
    grid.point_data["phi"] = phi.flatten(order="F")

    contour = grid.contour(isosurfaces=[iso_value], scalars="phi")
    contour["height"] = contour.points[:, 2]

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(contour, scalars="height", cmap="RdBu", opacity=0.6,
                smooth_shading=True, show_scalar_bar=True)
    pl.add_axes()
    pl.camera_position = "iso"
    pl.screenshot(out_path)
    pl.close()


def render_phi_volume(phi, out_path, iso_value=0.5, gas_opacity=0.15):
    """
    Full liquid body shape, not just the interface surface: liquid region
    as a solid opaque mesh, gas region as a genuinely see-through mesh.

    Not a volume render - volume rendering's per-voxel opacity compositing
    (add_volume with either a custom opacity array or the built-in presets)
    proved unpredictable here: custom arrays came out fully blank across a
    wide tested range, and the "sigmoid" preset came out fully opaque even
    for the gas region, on the same data, for reasons that didn't match the
    expected compositing math. Thresholding phi into two separate meshes
    and setting plain per-mesh opacity is simple, predictable, and what
    actually produces visible transparency.
    """
    grid = make_grid(phi.shape)
    grid.point_data["phi"] = phi.flatten(order="F")

    # threshold() keeps whole voxel cells, giving a blocky/staircase boundary;
    # smooth_taubin() (volume-preserving, unlike plain Laplacian smoothing)
    # smooths that down to the interface without shrinking the shape
    liquid = grid.threshold(iso_value, scalars="phi").extract_surface(algorithm="dataset_surface")
    liquid = liquid.smooth_taubin(n_iter=200, pass_band=0.05)
    gas = grid.threshold(iso_value, scalars="phi", invert=True).extract_surface(algorithm="dataset_surface")
    gas = gas.smooth_taubin(n_iter=200, pass_band=0.05)

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(liquid, color="darkred", opacity=1.0, smooth_shading=True)
    pl.add_mesh(gas, color="lightblue", opacity=gas_opacity, smooth_shading=True)
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
    """
    vec_field shape: (3, X, Y, Z). Subsampled by stride so glyphs aren't
    overwhelming. Built as a plain point cloud (not a subsampled ImageData)
    because extract_points() on a structured grid silently returns zero
    points with an index array - not worth fighting that API.
    """
    Xn, Yn, Zn = vec_field.shape[1:]
    xs, ys, zs = np.meshgrid(
        np.arange(0, Xn, stride), np.arange(0, Yn, stride), np.arange(0, Zn, stride),
        indexing="ij"
    )
    points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=-1).astype(float)

    vec_sub = vec_field[:, ::stride, ::stride, ::stride]
    vectors = np.stack([vec_sub[0], vec_sub[1], vec_sub[2]], axis=-1).reshape(-1, 3)

    cloud = pv.PolyData(points)
    cloud[field_name] = vectors
    cloud.set_active_vectors(field_name)
    glyphs = cloud.glyph(orient=field_name, scale=True, factor=factor)

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
    render_phi_volume(fields["phi"], os.path.join(args.out_dir, f"{tag}_phi_volume.png"))

    print(f"Saved renders for {tag} to {args.out_dir}")


if __name__ == "__main__":
    main()
