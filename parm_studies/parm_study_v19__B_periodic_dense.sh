#!/bin/bash
# B_periodic (sigma_lat 0.01, theta 60, tau_f = tau_g = 0.65, alpha 90, periodic x, 120,010 iterations) packed densely so it needs fewer nodes.
# Ranks per node: RPN (default 128 -> 16 nodes).  Slurm can hand out only about 28.2 GB of memory on a busy genoa node (750 GB - 5 GB system reserve - other jobs),
# so the memory request per rank is 28000/RPN MB: RPN=128 -> 218 MB, RPN=96 -> 291 MB, RPN=64 -> 437 MB.
# The estimated use after the clean-up is about 230 MB per rank: 128 per node is therefore likely too tight (29 GB); RPN=96 (22 nodes) fits.
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v19__B_periodic_dense.sh     or     RPN=96 bash parm_study_v19__B_periodic_dense.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.py
RPN=${RPN:-128}
MEM=$((28000 / RPN))

grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Python file on the server does not have the snapshot edits - upload it first"; exit 1; }
grep -q "np.meshgrid(np.arange(Xn" "$PYFILE" && { echo "STOP: the Python file on the server still builds full-domain arrays - upload the lean version first"; exit 1; }
grep -q "^import matplotlib" "$PYFILE" && { echo "STOP: the Python file on the server still imports matplotlib - upload the newest lean version first"; exit 1; }

echo "ranks per node $RPN, memory per rank ${MEM} MB, nodes $(( (2048 + RPN - 1) / RPN ))"
sbatch --job-name=B_periodic --time=16:00:00 --ntasks=2048 --ntasks-per-node=$RPN --mem-per-cpu=$MEM \
    --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=60,TAUF=0.65,MULT=10,SIGMA=0.01 \
    "$SCRIPT"
echo "submitted - check with: squeue -u $USER"
