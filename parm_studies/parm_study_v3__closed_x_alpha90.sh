#!/bin/bash
# Closed x walls (PERIODIC_X=0), vertical channel (alpha 90), twins of the periodic runs baseline and sig_0.1.
# Run from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v3__closed_x_alpha90.sh
# Fixed: theta 60, tau_f 1.0, rho_G = 0.01, rho_L = 10, 2048 ranks (about 0.46 s per iteration).
# Iterations = 12001 * 3 = 36,003 (about 4.7 h).

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh

submit() {   # name  sigma
    sbatch --job-name="$1" --time=8:00:00 --ntasks=2048 --ntasks-per-node=64 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=0,THETA=60,TAUF=1.0,MULT=3,SIGMA="$2" \
        "$SCRIPT"
}

submit closed90_0.1   0.1
submit closed90_0.01  0.01
echo "submitted - check with: squeue -u $USER"
