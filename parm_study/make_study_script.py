"""Writes parm_study/<script>_study.py: a copy of the channel script with environment-variable hooks.
The original script is not touched. Every patch must match exactly once, otherwise this stops.

Hooks (all optional; without them the copy behaves like the original):
  STUDY_D_ND         lattice cells across the channel (replaces DEFAULT_D_ND = 384)
  STUDY_DIMS         MPI decomposition "16,8,8" (replaces the 512/1024/2048 table)
  STUDY_MAXITER      iterations (replaces CORE_TOTAL_ITERATIONS = 12001, MULTIPLES stays 1)
  STUDY_TIMING=1     no dumps, no diagnostics in the loop (pure compute timing)
  STUDY_DUMP_EVERY   dump the diagnostics every N iterations instead of the script's own schedule
  STUDY_LEAN=1       at a dump only surface_height + diagnostic_lists (no plane_yz, plane_xz, phi3d)
  STUDY_OUT          output folder (replaces <script folder>/FreesurfaceImages)
  STUDY_NORING=1     no rolling crash buffers (ring/*.npy, ~4 GB per 1024-rank run)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster_channel.py")
DST = os.path.join(HERE, "free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster_channel_study.py")

with open(SRC, "r", encoding="utf-8", newline="") as f:
    text = f.read()
text = text.replace("\r\n", "\n")


def patch(old, new):
    global text
    n = text.count(old)
    if n != 1:
        sys.exit("patch anchor found %d times (need 1):\n%s" % (n, old))
    text = text.replace(old, new)


patch("DEFAULT_D_ND = 384\n",
      "DEFAULT_D_ND = int(os.environ.get(\"STUDY_D_ND\", 384))\n")

patch("    CORE_TOTAL_ITERATIONS: int = 12001\n",
      "    CORE_TOTAL_ITERATIONS: int = int(os.environ.get(\"STUDY_MAXITER\", 12001))\n")

patch("ADD_METRICS = True\n",
      "ADD_METRICS = os.environ.get(\"STUDY_TIMING\", \"0\") != \"1\"\n")

patch("images_subdir = os.path.join(script_dir, IMAGES_SUBDIR)\n",
      "images_subdir = os.environ.get(\"STUDY_OUT\", os.path.join(script_dir, IMAGES_SUBDIR))\n")

patch("assert comm.Get_size() in (512, 1024, 2048), f\"need 512, 1024 or 2048 ranks, got {comm.Get_size()}\"\n"
      "dims = {512: [8, 8, 8], 1024: [16, 8, 8], 2048: [16, 16, 8]}[comm.Get_size()]\n",
      "if \"STUDY_DIMS\" in os.environ:\n"
      "    dims = [int(v) for v in os.environ[\"STUDY_DIMS\"].split(\",\")]\n"
      "    assert dims[0] * dims[1] * dims[2] == comm.Get_size(), f\"dims {dims} need {dims[0]*dims[1]*dims[2]} ranks, got {comm.Get_size()}\"\n"
      "    assert Xn % dims[0] == 0 and Yn % dims[1] == 0 and Zn % dims[2] == 0, f\"dims {dims} do not divide the lattice {Xn}x{Yn}x{Zn}\"\n"
      "else:\n"
      "    assert comm.Get_size() in (512, 1024, 2048), f\"need 512, 1024 or 2048 ranks, got {comm.Get_size()}\"\n"
      "    dims = {512: [8, 8, 8], 1024: [16, 8, 8], 2048: [16, 16, 8]}[comm.Get_size()]\n")

patch("iterationsOfInterest = get_iterations_of_interest(fc.TOTAL_ITERATIONS, no_slices=51*fc.MULTIPLES, exp_factor=4.0)\n",
      "if \"STUDY_DUMP_EVERY\" in os.environ:\n"
      "    iterationsOfInterest = set(range(0, fc.TOTAL_ITERATIONS, int(os.environ[\"STUDY_DUMP_EVERY\"]))) | {fc.TOTAL_ITERATIONS - 1}\n"
      "else:\n"
      "    iterationsOfInterest = get_iterations_of_interest(fc.TOTAL_ITERATIONS, no_slices=51*fc.MULTIPLES, exp_factor=4.0)\n")

patch("RING_N = 1500\nif plane_x_is_local:\n",
      "RING_N = 1500\nRING_ON = os.environ.get(\"STUDY_NORING\") != \"1\"\nif plane_x_is_local and RING_ON:\n")
patch("    if plane_x_is_local:\n        _slot = iteration % RING_N\n",
      "    if plane_x_is_local and RING_ON:\n        _slot = iteration % RING_N\n")

# lean dump: indent everything from the chemical potential up to the phi3d snapshot under a guard
patch("        if iteration in snapshot_iterations:\n",
      "        if iteration in snapshot_iterations and os.environ.get(\"STUDY_LEAN\") != \"1\":\n")

lines = text.split("\n")
start = [i for i, l in enumerate(lines) if l == "        _mu_phi = chemical_potential(fc, _phi)"]
stop = [i for i, l in enumerate(lines) if l.startswith("        if iteration in snapshot_iterations and")]
if len(start) != 1 or len(stop) != 1 or start[0] >= stop[0]:
    sys.exit("lean block anchors not found once")
block = ["    " + l if l.strip() else l for l in lines[start[0]:stop[0]]]
lines[start[0]:stop[0]] = ["        if os.environ.get(\"STUDY_LEAN\") != \"1\":"] + block
text = "\n".join(lines)

with open(DST, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

compile(text, DST, "exec")
print("wrote", DST)
