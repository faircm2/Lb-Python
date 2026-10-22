#!/bin/bash
# Short stability test A: nu_phys 2.9e-6, tau_f 0.65, tau_g 0.65 (default), sigma_lat 0.05 (= 6.5 mN/m), alpha 90, theta 60, periodic x.
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v7__test_A_tauf0.65_sigma0.05.sh
# MULT=1 -> 12,001 iterations (about 25 min at 0.48 s per iteration, 1.4 ms physical).

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh

sbatch --job-name=test_A --time=2:00:00 --ntasks=2048 --ntasks-per-node=64 \
    --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=60,TAUF=0.65,MULT=1,SIGMA=0.05 \
    "$SCRIPT"
echo "submitted - check with: squeue -u $USER"
