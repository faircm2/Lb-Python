#!/bin/bash
# Baseline A: tau_f = tau_g = 0.65, sigma_lat 0.05 (= 6.5 mN/m), alpha 90, theta 60, rho_L 10, rho_G 0.01, nu_phys 2.9e-6.
# Two runs: periodic x and closed x walls.  Run ONCE from /work/classic/fr_jn194-lbm/scaling :
#     bash parm_study_v8__baseline_A_periodic_and_closed.sh
# MULT=10 -> 120,010 iterations = 14.0 ms physical, about 11.6 h at 0.35 s per iteration (time limit 16 h).
# New outputs: plane_xz dumps, 10 phi3d snapshots, physical-value block in parameters.txt.

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.py

grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Python file on the server does not have the snapshot edits - upload it first"; exit 1; }
grep -q "PHYSICAL VALUES" "$PYFILE" || { echo "STOP: the Python file on the server does not have the physical-values block - upload it first"; exit 1; }

submit() {   # name  periodic_x
    sbatch --job-name="$1" --time=16:00:00 --ntasks=2048 --ntasks-per-node=64 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X="$2",THETA=60,TAUF=0.65,MULT=10,SIGMA=0.05 \
        "$SCRIPT"
}

submit A_periodic 1
submit A_closed   0
echo "submitted - check with: squeue -u $USER"
