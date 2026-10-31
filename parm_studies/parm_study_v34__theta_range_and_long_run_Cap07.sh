#!/bin/bash
# Three runs for the experiment-matched channel (Cap07 script), all at sigma_lat 0.4 (0.046 N/m), tau_f 0.55, alpha 90, periodic x,
#   nu 1.0e-6, D 1.2 mm (384 cells, dx 3.125 um), tau_g 0.65, rho_L 10, rho_G 0.01, W 10, initial liquid depth 51 cells (159 um).  dt = 1.628e-7 s.
#
#   name         theta   MULT   iterations   physical time   purpose
#   T70_s0.4     70      2      24,002       3.9 ms          meniscus height for the upper end of the contact angle range
#   T62_s0.4     62.5    2      24,002       3.9 ms          meniscus height for the lower end of the range
#   LONG_s0.4    65      12     144,012      23.4 ms         long run: stays it stable, does the meniscus settle (slowest capillary mode about 72,000 iterations)
# The check against the 0.4 mm channel depth: surface_height.npz in each run folder holds the film height across the width for every dump
# (the earlier run EXP_s0.3 peaked at 277 um at the wall after 2.0 ms = about 12,300 iterations).
# About 0.32 s per iteration at 2048 ranks: MULT 2 about 2.2 h, MULT 12 about 13 h.  No node pinning (starts faster), 800 MB per rank (437 gave an out-of-memory kill).
# Output folders: ..._Cap07_..._tau_f_0.55__tau_g_0.65__..._Theta_<theta>_alpha90_sigma0.4_W10_nu1e-06_D1.2_fill51_periodicx_x<MULT>
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v34__theta_range_and_long_run_Cap07.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.py

[ -f "$SCRIPT" ] && [ -f "$PYFILE" ] || { echo "STOP: the Cap07 .py or .sh is not in this folder"; exit 1; }
grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Cap07 Python file has no snapshot_iterations"; exit 1; }
grep -q "(2/3)\*__fi" "$PYFILE" && { echo "STOP: the Cap07 Python file still has the old (2/3) f boundary lines"; exit 1; }
for OPT in "'--vf_W'" "'--nu'" "'--D'" "'--fill'"; do
    grep -q "add_argument($OPT" "$PYFILE" || { echo "STOP: the Cap07 Python file has no $OPT option"; exit 1; }
done
grep -q "def get_iterations_of_interest" "$PYFILE" || { echo "STOP: get_iterations_of_interest is missing in the Cap07 Python file"; exit 1; }
grep -q "def gather_full_volume" "$PYFILE" || { echo "STOP: gather_full_volume is missing in the Cap07 Python file"; exit 1; }
grep -q "Cap07_MPI_bwUniCluster.py" "$SCRIPT" || { echo "STOP: the Cap07 job script does not call the Cap07 Python file"; exit 1; }
for OPT in "--vf_W" "--nu" "--D" "--fill"; do
    grep -q -- "$OPT " "$SCRIPT" || { echo "STOP: the Cap07 job script does not pass $OPT"; exit 1; }
done
grep -q $'\r' "$SCRIPT" && { echo "STOP: the Cap07 job script has CR bytes"; exit 1; }

# name  hours  theta  mult
JOBLIST="
T70_s0.4    4   70    2
T62_s0.4    4   62.5  2
LONG_s0.4   16  65    12
"

# stop before submitting anything if a folder from an earlier run with the same settings exists
echo "$JOBLIST" | while read NAME H TH M; do
    [ -z "$NAME" ] && continue
    OLD=$(find . -maxdepth 4 -type d -name "*Cap07*tau_f_0.55__tau_g_0.65*Theta_${TH}*_alpha90_sigma0.4_W10_nu1e-06_D1.2_fill51_periodicx_x${M}" 2>/dev/null)
    [ -z "$OLD" ] || { echo "FOLDER EXISTS for $NAME - delete or rename it first: $OLD"; touch .v34_stop; }
done
[ -f .v34_stop ] && { rm -f .v34_stop; echo "STOP: nothing submitted"; exit 1; }

JOBS=""
while read NAME H TH M; do
    [ -z "$NAME" ] && continue
    J=$(sbatch --parsable --job-name="$NAME" --time="$H":00:00 --ntasks=2048 --mem-per-cpu=800 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA="$TH",TAUF=0.55,MULT="$M",SIGMA=0.4,VFW=10,NU=1e-6,DD=1.2e-3,FILL=51 \
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
sinfo -p genoa -o "%P %T %D %C"
