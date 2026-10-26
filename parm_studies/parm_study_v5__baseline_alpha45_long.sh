#!/bin/bash
# Long baseline: sigma 0.01, theta 60, tau_f 1.0, alpha 45, periodic x - run until the wall meniscus height is flat.
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v5__baseline_alpha45_long.sh
# MULT=18 -> 216,018 iterations = 84 ms (about 27.5 h at 0.46 s per iteration, about 10 time constants of the fit).
# Plane dumps appear in the run folder as it goes, so a partial copy can be pulled at any time.

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh

sbatch --job-name=base45_long --time=32:00:00 --ntasks=2048 --ntasks-per-node=64 \
    --export=ALL,ALPHA=45,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=60,TAUF=1.0,MULT=18,SIGMA=0.01 \
    "$SCRIPT"
echo "submitted - check with: squeue -u $USER"
