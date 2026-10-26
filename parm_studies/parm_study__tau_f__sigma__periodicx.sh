#!/bin/bash
# Parameter study around Zhang / Inamuro values - submit everything, in priority order.
# Run from /work/classic/fr_jn194-lbm/scaling :   bash parm_study__tau_f__sigma__periodicx.sh
# Fixed for all runs: theta = 60, rho_G = 0.01, rho_L = 10, 2048 ranks (about 0.56 s per iteration).
# Iterations = 12001 * MULT   (MULT=8: 96,008 ; MULT=5: 60,005 ; MULT=2: 24,002)

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh

submit() {   # name  hours  sigma  tau_f  alpha  mult  periodic_x
    sbatch --job-name="$1" --time="$2":00:00 --ntasks=2048 --ntasks-per-node=64 \
        --export=ALL,THETA=60,RHOG=0.01,RHOL=10,SIGMA="$3",TAUF="$4",ALPHA="$5",MULT="$6",PERIODIC_X="$7" \
        "$SCRIPT"
}

#      name        h   sigma  tau_f alpha mult periodic
submit closed_x    20  0.1    1.0   45    8    0     # 1: closed x walls, twin of the failing sigma 0.1 run
submit baseline    20  0.01   1.0   90    8    1     # 2: baseline
submit sig_0.1     14  0.1    1.0   90    5    1     # 3: sigma study
submit sig_0.001   14  0.001  1.0   90    5    1     # 4: sigma study
submit tau_1.25    14  0.01   1.25  90    5    1     # 5: tau_f study
submit tau_0.8     14  0.01   0.8   90    5    1     # 6: tau_f study
submit tau_0.6     14  0.01   0.6   90    5    1     # 7: tau_f study
submit sig_1       8   1      1.0   90    2    1     # 8: sigma = 1, stability test (probably fails)
echo "submitted - check with: squeue -u $USER"
