#!/bin/bash
# Studies 3, 4, 6, 7: one parameter at a time around the baseline, 9 runs:
#   baseline (tau_f 1.0, tau_g 0.65, vf_W 4, theta 60, sigma 0.1, fill 38), plus two values for each of
#   tau_f (0.8, 1.5), tau_g (0.55, 1.0), vf_W (6, 10), theta (45, 75).
# Lattice 768 x 384 x 384, 2048 ranks (dims 16x16x8). Study 5 (Kf) is not here: Kf is never used in the channel script.
# usage on NEMO2:  bash parm_study/submit_sensitivity.sh      (DRY=1 only prints)
# The tau_f runs get the same PHYSICAL time (iterations scale with 0.5/(tau_f-0.5)), because dt follows tau_f.
source "$(dirname "${BASH_SOURCE[0]}")/submit_lib.sh"
MAXITER=${MAXITER:-48001}
TIME=${TIME:-12:00:00}
DUMP=${DUMP:-250}            # diagnostics every DUMP base-iterations
N=${N:-2048}
COMMON="STUDY_LEAN=1 STUDY_NORING=1 SIGMA=${SIGMA:-0.1}"

sub "s0_base" $N $TIME $COMMON STUDY_MAXITER=$MAXITER STUDY_DUMP_EVERY=$DUMP

for tf in 0.8 1.5; do
    m=$(awk -v m=$MAXITER -v t=$tf 'BEGIN{printf "%d", m*0.5/(t-0.5)}')
    d=$(awk -v m=$DUMP -v t=$tf 'BEGIN{printf "%d", m*0.5/(t-0.5)}')
    tl=$TIME; [ "$tf" = "0.8" ] && tl=${TIME_TAUF08:-16:00:00}
    sub "s3_tauf_${tf}" $N $tl $COMMON STUDY_MAXITER=$m STUDY_DUMP_EVERY=$d TAUF=$tf
done
for tg in 0.55 1.0; do
    sub "s4_taug_${tg}" $N $TIME $COMMON STUDY_MAXITER=$MAXITER STUDY_DUMP_EVERY=$DUMP TAUG=$tg
done
for w in 6 10; do
    sub "s6_W_${w}" $N $TIME $COMMON STUDY_MAXITER=$MAXITER STUDY_DUMP_EVERY=$DUMP VFW=$w
done
for th in 45 75; do
    sub "s7_theta_${th}" $N $TIME $COMMON STUDY_MAXITER=$MAXITER STUDY_DUMP_EVERY=$DUMP THETA=$th
done
