#!/bin/bash
# Check of the cut-down script Cap07: same physics as the Cap06 run S0p5_eq30, so the numbers must reproduce.
# sigma_lat 0.5, theta 60, tau_f = tau_g = 0.65, alpha 90, rho_L 10, rho_G 0.01, W = 4, periodic x.   MULT=1 -> 12,001 iterations, about 65 min.
# Reference values from Cap06 (S0p5_eq30): floor min phi 0.99737 at iteration 100, 0.99285 at 3,753, 0.98972 at 7,513, 0.96912 at 12,000;
#                                          4-cell ripple 1.36e-4 at 12,000; about 100 floor cells below 0.999; no bubbles.
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v24__Cap07_check_sigma0.5.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.py

[ -f "$SCRIPT" ] && [ -f "$PYFILE" ] || { echo "STOP: the Cap07 .py or .sh is not in this folder - upload them first"; exit 1; }
grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Cap07 Python file has no snapshot_iterations"; exit 1; }
grep -q "(2/3)\*__fi" "$PYFILE" && { echo "STOP: the Cap07 Python file still has the old (2/3) f boundary lines"; exit 1; }
[ "$(grep -c '(1/3)\*__fi\[[0-9]*, *[^]]*\] + (1/3)\*__fi' "$PYFILE")" -ge 30 ] || { echo "STOP: fewer than 30 corrected f boundary lines in the Cap07 Python file"; exit 1; }
grep -q "fc.vf_W = _args.vf_W" "$PYFILE" || { echo "STOP: the Cap07 Python file has no --vf_W override"; exit 1; }
grep -q "def get_iterations_of_interest" "$PYFILE" || { echo "STOP: get_iterations_of_interest is missing in the Cap07 Python file"; exit 1; }
grep -q "def gather_full_volume" "$PYFILE" || { echo "STOP: gather_full_volume is missing in the Cap07 Python file"; exit 1; }
grep -q "Cap07_MPI_bwUniCluster.py" "$SCRIPT" || { echo "STOP: the Cap07 job script does not call the Cap07 Python file"; exit 1; }
grep -q -- "--vf_W" "$SCRIPT" || { echo "STOP: the Cap07 job script does not pass --vf_W"; exit 1; }
grep -q $'\r' "$SCRIPT" && { echo "STOP: the Cap07 job script has CR bytes"; exit 1; }

OLD=$(find . -maxdepth 4 -type d -name '*Cap07*Theta_60.0_alpha90_sigma0.5_periodicx' 2>/dev/null)
[ -z "$OLD" ] || { echo "STOP: an output folder from an earlier Cap07 run with these parameters exists - delete or rename it first:"; echo "$OLD"; exit 1; }

JOB=$(sbatch --parsable --job-name=S0p5_C07 --time=3:00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=437 \
    --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=60,TAUF=0.65,MULT=1,SIGMA=0.5 \
    "$SCRIPT")
echo "submitted job $JOB"

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
