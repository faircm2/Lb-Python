# prereqs_check.py
import importlib

packages = {
    "numpy": "2.3.4",
    "trimesh": "4.9.0",
    "shapely": "2.1.2",
    "stl (numpy-stl)": "3.2.0"
}

print("Checking Python prerequisites...\n")

for pkg_name, expected_version in packages.items():
    # 'stl' is imported as 'stl', not 'numpy-stl'
    import_name = "stl" if pkg_name.startswith("stl") else pkg_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{pkg_name:20s} -> Found, version: {version}")
    except ImportError:
        print(f"{pkg_name:20s} -> NOT FOUND!")

print("\nAll packages above must be found and versions reasonably match your workflow.")
