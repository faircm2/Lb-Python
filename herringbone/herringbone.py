import numpy as np
import trimesh


def stl_to_voxelmask(stl_path="herringbone.stl", resolution=5e-6, padding=10):
    mesh = trimesh.load(stl_path)
    print(f"Original faces: {len(mesh.faces)}")

    # Light decimation for speed (optional but recommended)
    if len(mesh.faces) > 10000:
        mesh = mesh.simplified_quadric_decimation(8000)
        print(f"Decimated to {len(mesh.faces)} faces")

    # THE LOW-MEMORY METHOD: ray casting – no subdivision explosion
    v = mesh.voxelized(pitch=resolution, method="ray")  # ← this is the key
    mask = v.fill().matrix.astype(bool)  # True = solid (grooves)

    # Padding
    if padding > 0:
        mask = np.pad(mask, pad_width=padding, mode='constant', constant_values=False)

    origin = mesh.bounds[0] - padding * resolution
    return mask, origin, resolution

# Run it
mask, origin, dx = stl_to_voxelmask("herringbone.stl", resolution=5e-6, padding=10)

print("\nACTUALLY DONE")
print(f"Grid: {mask.shape}")
print(f"Memory: {mask.nbytes / 1e9:.2f} GB")
print(f"Domain (µm): {np.round(np.array(mask.shape) * dx * 1e6, 1)}")