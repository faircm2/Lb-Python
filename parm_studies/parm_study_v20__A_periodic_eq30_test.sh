#!/bin/bash
# Test of the corrected f wall boundary (Zhang eq. 30 as printed: fi(xf) = 1/3 [fi(xff) + fi*(xf) + fi*(xs)]).
# Same parameters as A_periodic: sigma_lat 0.05, theta 60, tau_f = tau_g = 0.65, alpha 90, rho_L 10, rho_G 0.01, periodic x.
# MULT=2 -> 24,002 iterations (MULT=1 would write into the folder of the old test_A run).  About 2.3-3 h at 0.35-0.46 s per iteration.
# Dumps are dense up to iteration 7,200 and spaced about 400 apart up to 16,800, so iterations 9,000 and 15,000 are covered.
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v20__A_periodic_eq30_test.sh
# Reference values for the old boundary form (A_periodic): floor min phi 0.981 at iteration 9,000, 4-cell ripple 1.0e-4 at iteration 450, first bubble at 36,003.
# Reference values for the theta = 90 run (T90_periodic): floor min phi 0.9997 at 450, 0.9988 at 48,000, ripple 1e-5.

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.py

grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Python file on the server does not have the snapshot edits - upload it first"; exit 1; }
grep -q "np.meshgrid(np.arange(Xn" "$PYFILE" && { echo "STOP: the Python file on the server is not the lean version - upload it first"; exit 1; }
grep -q "(2/3)\*__fi" "$PYFILE" && { echo "STOP: the Python file on the server still has the old (2/3) f boundary lines - upload the edited file first"; exit 1; }
[ "$(grep -c '(1/3)\*__fi\[[0-9]*, *[^]]*\] + (1/3)\*__fi' "$PYFILE")" -ge 30 ] || { echo "STOP: fewer than 30 corrected f boundary lines found in the Python file on the server"; exit 1; }

OLD=$(find . -maxdepth 4 -type d -name '*Theta_60.0_alpha90_sigma0.05_periodicx_x2' 2>/dev/null)
[ -z "$OLD" ] || { echo "STOP: an output folder from an earlier x2 run with these parameters exists - delete or rename it first:"; echo "$OLD"; exit 1; }

JOB=$(sbatch --parsable --job-name=A_eq30 --time=5:00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=437 \
    --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=60,TAUF=0.65,MULT=2,SIGMA=0.05 \
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
