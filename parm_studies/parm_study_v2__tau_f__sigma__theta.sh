#!/bin/bash
# Parameter study v2: sigma, tau_f and vf_theta, all vertical (alpha 90), periodic x.
# Run from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v2__tau_f__sigma__theta.sh
# The closed-x run (8414238) and the validation run (8414074) are already running and are NOT part of this file.
# Fixed: rho_G = 0.01, rho_L = 10, 2048 ranks (about 0.56 s per iteration).
# Baseline: sigma 0.01, tau_f 1.0, theta 60.
# Iterations = 12001 * MULT   (MULT=3: 36,003 = about 5.6 h ; MULT=2: 24,002 = about 3.7 h)

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh

submit() {   # name  hours  sigma  tau_f  theta  mult
    sbatch --job-name="$1" --time="$2":00:00 --ntasks=2048 --ntasks-per-node=64 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,SIGMA="$3",TAUF="$4",THETA="$5",MULT="$6" \
        "$SCRIPT"
}

#      name        h  sigma  tau_f theta mult
submit baseline    8  0.01   1.0   60    3     # 1: baseline, theta 60
submit theta_90    8  0.01   1.0   90    3     # 2: theta study (neutral, flat meniscus)
submit theta_120   8  0.01   1.0   120   3     # 3: theta study (hydrophobic)
submit sig_0.1     8  0.1    1.0   60    3     # 4: sigma study
submit sig_0.001   8  0.001  1.0   60    3     # 5: sigma study
submit theta_75    8  0.01   1.0   75    3     # 6: theta study
submit theta_105   8  0.01   1.0   105   3     # 7: theta study
submit tau_1.25    5  0.01   1.25  60    2     # 8: tau_f study (14.0 ms)
submit tau_0.6     8  0.01   0.6   60    3     # 9: tau_f study (only 2.8 ms: dt is 5x smaller)
submit tau_0.8    12  0.01   0.8   60    5     # 10: tau_f study (14.0 ms)
submit sig_1       5  1      1.0   60    2     # 11: sigma = 1, stability test (probably fails)
echo "submitted - check with: squeue -u $USER"
