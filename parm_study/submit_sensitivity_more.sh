#!/bin/bash
# Studies 3, 4, 6, 7, second half: two more values each, so every parameter has 4 varied values (+ the baseline s0_base):
#   tau_f 0.8, 1.5 (done) + 0.7, 2.0     tau_g 0.55, 1.0 (done) + 0.8, 1.5
#   vf_W  6, 10 (done)    + 3, 8         theta 45, 75 (done)    + 30, 90
# 8 runs, same set-up as submit_sensitivity.sh (lattice 768 x 384 x 384, 2048 ranks, sigma 0.1).
# usage on NEMO2:  bash parm_study/submit_sensitivity_more.sh      (DRY=1 only prints)
# tau_f runs get the same PHYSICAL time (iterations scale with 0.5/(tau_f-0.5)); tau_f 0.7 is 120002 iterations.
source "$(dirname "${BASH_SOURCE[0]}")/submit_lib.sh"
MAXITER=${MAXITER:-48001}
TIME=${TIME:-12:00:00}
DUMP=${DUMP:-250}
N=${N:-2048}
COMMON="STUDY_LEAN=1 STUDY_NORING=1 SIGMA=${SIGMA:-0.1}"

for tf in 0.7 2.0; do
    m=$(awk -v m=$MAXITER -v t=$tf 'BEGIN{printf "%d", m*0.5/(t-0.5)}')
    d=$(awk -v m=$DUMP -v t=$tf 'BEGIN{printf "%d", m*0.5/(t-0.5)}')
    tl=$TIME; [ "$tf" = "0.7" ] && tl=${TIME_TAUF07:-24:00:00}
    sub "s3_tauf_${tf}" $N $tl $COMMON STUDY_MAXITER=$m STUDY_DUMP_EVERY=$d TAUF=$tf
done
for tg in 0.8 1.5; do
    sub "s4_taug_${tg}" $N $TIME $COMMON STUDY_MAXITER=$MAXITER STUDY_DUMP_EVERY=$DUMP TAUG=$tg
done
for w in 3 8; do
    sub "s6_W_${w}" $N $TIME $COMMON STUDY_MAXITER=$MAXITER STUDY_DUMP_EVERY=$DUMP VFW=$w
done
for th in 30 90; do
    sub "s7_theta_${th}" $N $TIME $COMMON STUDY_MAXITER=$MAXITER STUDY_DUMP_EVERY=$DUMP THETA=$th
done
