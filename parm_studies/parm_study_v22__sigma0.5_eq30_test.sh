#!/bin/bash
# Surface-tension test with the corrected f wall boundary (Zhang eq. 30 as printed): sigma_lat 0.5.
# theta 60, tau_f = tau_g = 0.65, alpha 90, rho_L 10, rho_G 0.01, periodic x.   MULT=1 -> 12,001 iterations, about 70-95 min at 0.35-0.48 s per iteration.
# Reference values, old boundary form (tau_f 1.0, theta 60): sigma 1.0 -> floor min phi below 0.9 from iteration 1,800, first bubble at 2,571;
#                                                              sigma 0.1 -> below 0.9 from iteration 22,542, first bubble at 25,922.
# Reference values, corrected form at sigma 0.05 (job A_eq30): floor min phi 0.9991 at iteration 20,641, ripple 4.5e-5, no bubbles.
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v22__sigma0.5_eq30_test.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.py

grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Python file on the server does not have the snapshot edits - upload it first"; exit 1; }
grep -q "np.meshgrid(np.arange(Xn" "$PYFILE" && { echo "STOP: the Python file on the server is not the lean version - upload it first"; exit 1; }
grep -q "(2/3)\*__fi" "$PYFILE" && { echo "STOP: the Python file on the server still has the old (2/3) f boundary lines - upload the edited file first"; exit 1; }
[ "$(grep -c '(1/3)\*__fi\[[0-9]*, *[^]]*\] + (1/3)\*__fi' "$PYFILE")" -ge 30 ] || { echo "STOP: fewer than 30 corrected f boundary lines found in the Python file on the server"; exit 1; }

OLD=$(find . -maxdepth 4 -type d -name '*Theta_60.0_alpha90_sigma0.5_periodicx' 2>/dev/null)
[ -z "$OLD" ] || { echo "STOP: an output folder from an earlier run with these parameters exists - delete or rename it first:"; echo "$OLD"; exit 1; }

JOB=$(sbatch --parsable --job-name=S0p5_eq30 --time=3:00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=437 \
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
