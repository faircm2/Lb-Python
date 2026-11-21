#!/bin/bash
# Study 8: surface tension sigma (vf_sigma). Baseline (sigma 0.1) is s0_base from submit_sensitivity.sh.
# 4 new runs, same set-up as the baseline (tau_f 1.0, tau_g 0.65, vf_W 4, theta 60, fill 38), 48001 iterations,
# 2048 ranks (dims 16x16x8), ~4.3 h each.
# usage on NEMO2:  bash parm_study/submit_sigma.sh      (DRY=1 only prints)
source "$(dirname "${BASH_SOURCE[0]}")/submit_lib.sh"
MAXITER=${MAXITER:-48001}
TIME=${TIME:-12:00:00}
DUMP=${DUMP:-250}
N=${N:-2048}
for s in 0.01 0.03 0.2 0.3; do
    sub "s8_sigma_${s}" $N $TIME STUDY_LEAN=1 STUDY_NORING=1 SIGMA=$s STUDY_MAXITER=$MAXITER STUDY_DUMP_EVERY=$DUMP
done
