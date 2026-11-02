#!/bin/bash
# Contact angle range for the experiment-matched channel (Cap07 script), all at sigma_lat 0.4 (0.046 N/m), tau_f 0.55, alpha 90, periodic x,
#   nu 1.0e-6, D 1.2 mm (384 cells, dx 3.125 um), tau_g 0.65, rho_L 10, rho_G 0.01, W 10, initial liquid depth 51 cells (159 um).  dt = 1.628e-7 s.
#
# Seven runs, theta 62.5, 70, 75, 80, 85, 88, 90, MULT 2 = 24,002 iterations = 3.9 ms each, as they were actually run (the first two were
# submitted by hand, the last five with the same command).  Results: meniscus_theta_range/ (peak wall height 291 um at 62.5 deg, all below 400 um).
# The long run (theta 65, MULT 12, 2048 ranks) is NOT part of this script; it was submitted separately and cancelled at 38,574 iterations.
# 1024 ranks as 16 nodes x 64 tasks, 1600 MB per rank: rank 0 gathers the full volume (MaxRSS 9.1 GB) and Slurm enforces memory per node,
# so a loose layout (--nodes=6-40) can give rank 0's node too few tasks and the job dies with OUT_OF_MEMORY.  About 0.85 s per iteration, 5.5 to 6 h.
# The check against the 0.4 mm channel depth: surface_height.npz in each run folder holds the film height across the width for every dump.
# Output folders: ..._Cap07_..._tau_f_0.55__tau_g_0.65__..._Theta_<theta>_alpha90_sigma0.4_W10_nu1e-06_D1.2_fill51_periodicx_x<MULT>
# genoa QOS limit: 8000 cores per user, so at most 7 x 1024 at the same time.
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
T62.5_s0.4  8   62.5  2
T70_s0.4    8   70    2
T75_s0.4    8   75    2
T80_s0.4    8   80    2
T85_s0.4    8   85    2
T88_s0.4    8   88    2
T90_s0.4    8   90    2
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
    J=$(sbatch --parsable --job-name="$NAME" --time="$H":00:00 --ntasks=1024 --nodes=16 --ntasks-per-node=64 --mem-per-cpu=1600 \
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
