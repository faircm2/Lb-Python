import os

folder = r"C:\Dirk\Neethling\iems\master\PythonLBCourse\frames2D"
# Filter files matching the pattern frame_XXXX.png
files = [f for f in os.listdir(folder) if f.startswith("frame_") and f.endswith(".png")]
# Sort by the numeric part
files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
for i, file in enumerate(files, 1):
    old_path = os.path.join(folder, file)
    new_path = os.path.join(folder, f"frame{i:04d}.png")
    os.rename(old_path, new_path)
    print(f"Renamed {file} to frame{i:04d}.png")