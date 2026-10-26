#!/bin/bash
# Long run of the experiment-matched case (Cap07 script): run until the meniscus height at the walls and at the channel centre has levelled off.
# sigma_lat 0.625, tau_f 0.55, theta 65, alpha 90 (vertical film), periodic x.  The other settings are those of parm_study_v28:
#   nu 1.0e-6, D 1.2 mm (384 cells, dx 3.125 um), tau_g 0.65, rho_L 10, rho_G 0.01, W 10, initial liquid depth 51 cells (159 um).
# dt = 1.628e-7 s.  MULT=50 -> 600,050 iterations = 97.7 ms of physical time.
# Time scale of the film: viscous time h^2/nu = 25.6 ms = 157,000 iterations, so 600,050 iterations are about 3.8 of them.
# About 0.32 s per iteration at 2048 ranks -> about 54 h; the time limit is 80 h (the partition allows 96 h).  There is NO restart: if the job dies, only the dumps written so far remain.
# Dumps: get_iterations_of_interest gives about 60 dump iterations in total.  Early ones (up to 500) are at the start, then about every 10,000 iterations
#   from 45,000 on; the first 24,000 iterations are covered by the shorter jobs EXP_s0.625 / EXP_s0.3 (parm_study_v28).
#   The surface height (wall and centre) is written to surface_height.npz at every dump, so a partial copy can be pulled at any time and the job can be
#   cancelled with scancel as soon as the curves are flat.
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v31__experiment_long_Cap07.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.py
MULT=${MULT:-50}

[ -f "$SCRIPT" ] && [ -f "$PYFILE" ] || { echo "STOP: the Cap07 .py or .sh is not in this folder - upload them first"; exit 1; }
grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Cap07 Python file has no snapshot_iterations"; exit 1; }
grep -q "(2/3)\*__fi" "$PYFILE" && { echo "STOP: the Cap07 Python file still has the old (2/3) f boundary lines"; exit 1; }
[ "$(grep -c '(1/3)\*__fi\[[0-9]*, *[^]]*\] + (1/3)\*__fi' "$PYFILE")" -ge 30 ] || { echo "STOP: fewer than 30 corrected f boundary lines in the Cap07 Python file"; exit 1; }
for OPT in "'--vf_W'" "'--nu'" "'--D'" "'--fill'"; do
    grep -q "add_argument($OPT" "$PYFILE" || { echo "STOP: the Cap07 Python file has no $OPT option - upload the newest file"; exit 1; }
done
grep -q "height=FILL + 0.5 - z_offset" "$PYFILE" || { echo "STOP: the Cap07 Python file does not use --fill in the initial profile"; exit 1; }
grep -q "def get_iterations_of_interest" "$PYFILE" || { echo "STOP: get_iterations_of_interest is missing in the Cap07 Python file"; exit 1; }
grep -q "def gather_full_volume" "$PYFILE" || { echo "STOP: gather_full_volume is missing in the Cap07 Python file"; exit 1; }
grep -q "Cap07_MPI_bwUniCluster.py" "$SCRIPT" || { echo "STOP: the Cap07 job script does not call the Cap07 Python file"; exit 1; }
for OPT in "--vf_W" "--nu" "--D" "--fill"; do
    grep -q -- "$OPT " "$SCRIPT" || { echo "STOP: the Cap07 job script does not pass $OPT - upload the newest file"; exit 1; }
done
grep -q $'\r' "$SCRIPT" && { echo "STOP: the Cap07 job script has CR bytes"; exit 1; }

OLD=$(find . -maxdepth 4 -type d -name "*Cap07*tau_f_0.55*Theta_65.0*sigma0.625_W10_nu1e-06_D1.2_fill51_periodicx_x${MULT}" 2>/dev/null)
[ -z "$OLD" ] || { echo "STOP: an output folder from an earlier run with MULT=$MULT exists - delete or rename it first:"; echo "$OLD"; exit 1; }

JOB=$(sbatch --parsable --job-name=EXP_long --time=80:00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=437 \
    --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=65,TAUF=0.55,MULT="$MULT",SIGMA=0.625,VFW=10,NU=1e-6,DD=1.2e-3,FILL=51 \
    "$SCRIPT")
echo "submitted EXP_long as job $JOB  (MULT=$MULT, $((12001 * MULT)) iterations)"

# diagnostics (read only)
sleep 5
echo "=== squeue (state, reason, start estimate)"
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.6D %.7C %R"
squeue -j "$JOB" --start
echo "=== scontrol show job"
scontrol show job "$JOB" | grep -E "JobState|Reason|StartTime|MinMemory|NumNodes|NumCPUs|TimeLimit|Partition"
echo "=== sinfo genoa"
sinfo -p genoa -o "%P %T %D"
echo "=== reservations"
scontrol show reservation 2>/dev/null | grep -E "ReservationName|StartTime|EndTime|PartitionName" || echo "none"
