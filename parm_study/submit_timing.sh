#!/bin/bash
# Study 1 (wall time vs lattice size) and study 2 (wall time vs MPI decomposition), 2 repeats each.
# On NEMO2:  bash parm_study/submit_timing.sh        (DRY=1 bash ... only prints)
source "$(dirname "${BASH_SOURCE[0]}")/submit_lib.sh"
MAXITER=${MAXITER:-300}

for rep in 1 2; do
    # study 1: 1024 ranks, dims 16x8x8, lattice D_ND x 2 D_ND x D_ND ... (Y = Z = D_ND, X = 2 D_ND)
    for D in 64 96 128 192 256 320 384 448 512; do
        sub "t1_D${D}_r${rep}" 1024 00:40:00 STUDY_TIMING=1 STUDY_D_ND=$D STUDY_DIMS=16,8,8 STUDY_MAXITER=$MAXITER
    done
    # study 2: D_ND = 384 (768 x 384 x 384), different rank counts and shapes
    for cfg in "128 8,4,4" "256 8,8,4" "256 16,4,4" "512 8,8,8" "512 16,8,4" "512 32,4,4" \
               "1024 16,8,8" "1024 8,8,16" "1024 32,8,4" "1024 16,16,4" "1024 64,4,4" \
               "2048 16,16,8" "2048 32,8,8"; do
        set -- $cfg
        n=$1; dims=$2
        tl=01:00:00; [ "$n" -le 256 ] && tl=02:30:00
        sub "t2_n${n}_d${dims//,/x}_r${rep}" "$n" "$tl" STUDY_TIMING=1 STUDY_D_ND=384 STUDY_DIMS=$dims STUDY_MAXITER=$MAXITER
    done
done
