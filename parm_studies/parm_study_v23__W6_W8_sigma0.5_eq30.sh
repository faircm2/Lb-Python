#!/bin/bash
# Interface-width test with the corrected f wall boundary (Zhang eq. 30 as printed): sigma_lat 0.5, W = 6 and W = 8.
# Everything except W equals the baseline job S0p5_eq30 (W = 4): theta 60, tau_f = tau_g = 0.65, alpha 90, rho_L 10, rho_G 0.01, periodic x.
# MULT=1 -> 12,001 iterations, about 70-95 min per job at 0.35-0.48 s per iteration (two jobs of 2048 ranks, 32 nodes each).
# Compare floor min phi and the 4-cell ripple with the W = 4 run (sigma 0.5: min phi about 0.993, ripple about 2e-4 at iteration 3,700).
# Output folders: ..._sigma0.5_W6_periodicx and ..._sigma0.5_W8_periodicx
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v23__W6_W8_sigma0.5_eq30.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.py

grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Python file on the server does not have the snapshot edits - upload it first"; exit 1; }
grep -q "np.meshgrid(np.arange(Xn" "$PYFILE" && { echo "STOP: the Python file on the server is not the lean version - upload it first"; exit 1; }
grep -q "(2/3)\*__fi" "$PYFILE" && { echo "STOP: the Python file on the server still has the old (2/3) f boundary lines - upload the edited file first"; exit 1; }
[ "$(grep -c '(1/3)\*__fi\[[0-9]*, *[^]]*\] + (1/3)\*__fi' "$PYFILE")" -ge 30 ] || { echo "STOP: fewer than 30 corrected f boundary lines found in the Python file on the server"; exit 1; }
grep -q "fc.vf_W = _args.vf_W" "$PYFILE" || { echo "STOP: the Python file on the server has no --vf_W override - upload it first"; exit 1; }
grep -q -- "--vf_W" "$SCRIPT" || { echo "STOP: the job script on the server does not pass --vf_W - upload it first"; exit 1; }
grep -q $'\r' "$SCRIPT" && { echo "STOP: the job script on the server has CR bytes - run: sed -i 's/\r\$//' $SCRIPT"; exit 1; }

for W in 6 8; do
    OLD=$(find . -maxdepth 4 -type d -name "*Theta_60.0_alpha90_sigma0.5_W${W}_periodicx" 2>/dev/null)
    [ -z "$OLD" ] || { echo "STOP: an output folder from an earlier W=$W run exists - delete or rename it first:"; echo "$OLD"; exit 1; }
done

submit() {   # W
    sbatch --parsable --job-name="S0p5_W$1" --time=3:00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=437 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=60,TAUF=0.65,MULT=1,SIGMA=0.5,VFW="$1" \
        "$SCRIPT"
}

JOB6=$(submit 6)
JOB8=$(submit 8)
echo "submitted W=6 as job $JOB6, W=8 as job $JOB8"

# diagnostics (read only)
sleep 5
echo "=== squeue (state, reason, start estimate)"
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.6D %.7C %R"
squeue -j "$JOB6,$JOB8" --start
echo "=== scontrol show job"
for J in "$JOB6" "$JOB8"; do
    echo "--- job $J"
    scontrol show job "$J" | grep -E "JobState|Reason|StartTime|MinMemory|NumNodes|NumCPUs|TimeLimit|Partition"
done
echo "=== sinfo genoa"
sinfo -p genoa -o "%P %T %D"
echo "=== reservations"
scontrol show reservation 2>/dev/null | grep -E "ReservationName|StartTime|EndTime|PartitionName" || echo "none"
