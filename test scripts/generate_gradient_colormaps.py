import os
import numpy as np
import matplotlib.pyplot as plt

# --- SETTINGS ---
GRADIENTS = "gradients"                        # Folder name for gradients
script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
gradients_subdir = os.path.join(script_dir, GRADIENTS)  # Full path

use_common_scale = True                        # Use common vmin/vmax per component

# --- Gather all .txt files ---
txt_files = [f for f in os.listdir(gradients_subdir) if f.endswith(".txt")]
if not txt_files:
    print("No .txt files found in folder.")
    exit()

# --- Determine common color scales per component (optional) ---
if use_common_scale:
    vmins = {}
    vmaxs = {}
    components = set(f.split("_f")[0] for f in txt_files)  # extract component name before '_f'
    
    for comp in components:
        all_values = []
        for f in txt_files:
            if f.startswith(comp):
                arr = np.loadtxt(os.path.join(gradients_subdir, f))
                all_values.append(arr.flatten())
        if all_values:
            stacked = np.concatenate(all_values)
            vmins[comp] = np.min(stacked)
            vmaxs[comp] = np.max(stacked)
        else:
            vmins[comp] = 0
            vmaxs[comp] = 1

# --- Generate colormaps for all files ---
for f in txt_files:
    file_path = os.path.join(gradients_subdir, f)
    data = np.loadtxt(file_path)
    
    # Extract component name for color scaling
    comp = f.split("_f")[0]
    
    plt.figure(figsize=(6, 5))
    if use_common_scale and comp in vmins:
        plt.imshow(data, origin="lower", cmap="viridis", vmin=vmins[comp], vmax=vmaxs[comp])
    else:
        plt.imshow(data, origin="lower", cmap="viridis")
    
    plt.colorbar(label=comp)
    plt.title(f"{comp} from {f}")
    
    out_file = os.path.join(gradients_subdir, f.replace(".txt", ".png"))
    plt.savefig(out_file, dpi=150)
    plt.close()
    
    print(f"Saved colormap: {out_file}")