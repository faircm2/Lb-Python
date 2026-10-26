#!/bin/bash
# Rerun of the sigma, tau_f, theta and alpha studies with the corrected f wall boundary (Cap07 script).
# The earlier runs (Cap05, Cap06 `study/`, Cap04) used the old f boundary form.  Same parameters and iteration counts as parm_study_v2 where it existed.
# Fixed: tau_g 0.65, rho_L 10, rho_G 0.01, W 4, nu 2.9e-6, D 1.0 mm, periodic x, vertical channel alpha 90 unless stated, 2048 ranks.
# Baseline: sigma 0.01, tau_f 1.0, theta 60, alpha 90.   Iterations = 12001 * MULT.
#
#   name          sigma   tau_f  theta  alpha  MULT   (physical time as in the old study)
#   R_base        0.01    1.0    60     90     3      baseline
#   R_sig_0.001   0.001   1.0    60     90     3      sigma study
#   R_sig_0.1     0.1     1.0    60     90     3      sigma study
#   R_sig_1       1.0     1.0    60     90     2      sigma study (stability test)
#   R_tau_0.6     0.01    0.6    60     90     3      tau_f study (only 2.8 ms: dt is 5x smaller)
#   R_tau_0.8     0.01    0.8    60     90     5      tau_f study
#   R_tau_1.25    0.01    1.25   60     90     2      tau_f study
#   R_theta_75    0.01    1.0    75     90     3      theta study
#   R_theta_90    0.01    1.0    90     90     3      theta study
#   R_theta_105   0.01    1.0    105    90     3      theta study
#   R_theta_120   0.01    1.0    120    90     3      theta study
#   R_alpha_45    0.01    1.0    60     45     3      alpha study (alpha 90 is the baseline)
#   R_alpha_65    0.01    1.0    60     65     3      alpha study
#   R_alpha_75    0.01    1.0    60     75     3      alpha study
#   R_alpha_85    0.01    1.0    60     85     3      alpha study
# About 0.32 s per iteration with Cap07: MULT 2 about 2.1 h, MULT 3 about 3.2 h, MULT 5 about 5.3 h.  Three jobs run at a time.
# Output folders: ..._Cap07_..._tau_f_<tau_f>__tau_g_0.65__..._Theta_<theta>_alpha<alpha>_sigma<sigma>_periodicx_x<MULT>
#
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v30__study_rerun_Cap07.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.py

[ -f "$SCRIPT" ] && [ -f "$PYFILE" ] || { echo "STOP: the Cap07 .py or .sh is not in this folder - upload them first"; exit 1; }
grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Cap07 Python file has no snapshot_iterations"; exit 1; }
grep -q "(2/3)\*__fi" "$PYFILE" && { echo "STOP: the Cap07 Python file still has the old (2/3) f boundary lines"; exit 1; }
[ "$(grep -c '(1/3)\*__fi\[[0-9]*, *[^]]*\] + (1/3)\*__fi' "$PYFILE")" -ge 30 ] || { echo "STOP: fewer than 30 corrected f boundary lines in the Cap07 Python file"; exit 1; }
grep -q "def get_iterations_of_interest" "$PYFILE" || { echo "STOP: get_iterations_of_interest is missing in the Cap07 Python file"; exit 1; }
grep -q "def gather_full_volume" "$PYFILE" || { echo "STOP: gather_full_volume is missing in the Cap07 Python file"; exit 1; }
grep -q "Cap07_MPI_bwUniCluster.py" "$SCRIPT" || { echo "STOP: the Cap07 job script does not call the Cap07 Python file"; exit 1; }
grep -q $'\r' "$SCRIPT" && { echo "STOP: the Cap07 job script has CR bytes"; exit 1; }

# name  hours  sigma  tau_f  theta  alpha  mult
JOBLIST="
R_base       6   0.01   1.0   60    90   3
R_sig_0.001  6   0.001  1.0   60    90   3
R_sig_0.1    6   0.1    1.0   60    90   3
R_sig_1      4   1.0    1.0   60    90   2
R_tau_0.6    6   0.01   0.6   60    90   3
R_tau_0.8    8   0.01   0.8   60    90   5
R_tau_1.25   4   0.01   1.25  60    90   2
R_theta_75   6   0.01   1.0   75    90   3
R_theta_90   6   0.01   1.0   90    90   3
R_theta_105  6   0.01   1.0   105   90   3
R_theta_120  6   0.01   1.0   120   90   3
R_alpha_45   6   0.01   1.0   60    45   3
R_alpha_65   6   0.01   1.0   60    65   3
R_alpha_75   6   0.01   1.0   60    75   3
R_alpha_85   6   0.01   1.0   60    85   3
"

# stop before submitting anything if a folder from an earlier Cap07 run with the same settings exists
echo "$JOBLIST" | while read NAME H S TF TH AL M; do
    [ -z "$NAME" ] && continue
    TAG=$(printf '%g' "$S")
    OLD=$(find . -maxdepth 4 -type d -name "*Cap07*tau_f_${TF}__tau_g_0.65*Theta_${TH}.0_alpha${AL}_sigma${TAG}_periodicx_x${M}" 2>/dev/null)
    [ -z "$OLD" ] || { echo "FOLDER EXISTS for $NAME - delete or rename it first: $OLD"; touch .v30_stop; }
done
[ -f .v30_stop ] && { rm -f .v30_stop; echo "STOP: nothing submitted"; exit 1; }

JOBS=""
while read NAME H S TF TH AL M; do
    [ -z "$NAME" ] && continue
    J=$(sbatch --parsable --job-name="$NAME" --time="$H":00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=437 \
        --export=ALL,ALPHA="$AL",RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA="$TH",TAUF="$TF",MULT="$M",SIGMA="$S" \
        "$SCRIPT")
    JOBS="$JOBS,$J"
    echo "submitted $NAME as job $J"
done <<< "$JOBLIST"
JOBS=${JOBS#,}

# diagnostics (read only)
sleep 5
echo "=== squeue (state, reason)"
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.6D %R"
echo "=== sinfo genoa"
sinfo -p genoa -o "%P %T %D"
echo "=== reservations"
scontrol show reservation 2>/dev/null | grep -E "ReservationName|StartTime|EndTime|PartitionName" || echo "none"
