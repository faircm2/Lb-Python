#!/bin/bash
# Baseline B: tau_f = tau_g = 0.65, sigma_lat 0.01 (= 1.29 mN/m, capillary speed 0.02), alpha 90, theta 60, rho_L 10, rho_G 0.01, nu_phys 2.9e-6.
# Two runs: periodic x and closed x walls, on genoa OR milan (whichever has room first).  Run ONCE from /work/classic/fr_jn194-lbm/scaling :
#     bash parm_study_v11__B_sigma0.01_genoa_or_milan.sh
# MULT=10 -> 120,010 iterations = 14.0 ms physical.  genoa: about 10-12 h; milan is slower (the time limit 40 h allows for that).

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.py

grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Python file on the server does not have the snapshot edits - upload it first"; exit 1; }

submit() {   # name  periodic_x
    sbatch --job-name="$1" --partition=genoa,milan --time=40:00:00 --ntasks=2048 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X="$2",THETA=60,TAUF=0.65,MULT=10,SIGMA=0.01 \
        "$SCRIPT"
}

submit B_periodic 1
submit B_closed   0
echo "submitted - check with: squeue -u $USER"
