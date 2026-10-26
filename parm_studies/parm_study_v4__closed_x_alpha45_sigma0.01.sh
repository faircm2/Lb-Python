#!/bin/bash
# Closed x walls (PERIODIC_X=0), alpha 45, sigma 0.01, theta 60, tau_f 1.0 - with the corner crash buffer.
# Run from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v4__closed_x_alpha45_sigma0.01.sh
# MULT=9 -> 108,009 iterations (about 14 h); MULT 9 also gives a folder name that no earlier run used (..._sigma0.01_x9).

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh

sbatch --job-name=closed45_0.01 --time=18:00:00 --ntasks=2048 --ntasks-per-node=64 \
    --export=ALL,ALPHA=45,RHOG=0.01,RHOL=10,PERIODIC_X=0,THETA=60,TAUF=1.0,MULT=9,SIGMA=0.01 \
    "$SCRIPT"
echo "submitted - check with: squeue -u $USER"
