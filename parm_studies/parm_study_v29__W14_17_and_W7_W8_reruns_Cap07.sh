#!/bin/bash
# Interface-width series, Cap07 script, sigma_lat 0.5 (corrected f wall boundary):
#   W = 14, 15, 16, 17 : extends the series W = 4..13 to see whether the effect tapers off.
#   W = 7, 8 (reruns)  : the two scattered points (W = 7 better than 6 and 8, W = 8 worse than 7 and 9), run again with Cap07.
#                        Cap07 gives different output folders, so the Cap06 results are kept.  Cap07 reproduces Cap06 exactly (check run S0p5_C07),
#                        so a different result for W = 7 or 8 would point to run-to-run noise, an identical one to a real effect.
# Everything except W equals the baseline S0p5_eq30: theta 60, tau_f = tau_g = 0.65, alpha 90, rho_L 10, rho_G 0.01, nu 2.9e-6, D 1.0 mm, periodic x.
# MULT=1 -> 12,001 iterations, about 65-80 min per job.  Six jobs of 2048 ranks (32 nodes each); the cluster runs three at a time, the others wait.
# Output folders: ..._Cap07_..._sigma0.5_W<W>_periodicx
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v29__W14_17_and_W7_W8_reruns_Cap07.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.py
WIDTHS="14 15 16 17 7 8"

[ -f "$SCRIPT" ] && [ -f "$PYFILE" ] || { echo "STOP: the Cap07 .py or .sh is not in this folder - upload them first"; exit 1; }
grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Cap07 Python file has no snapshot_iterations"; exit 1; }
grep -q "(2/3)\*__fi" "$PYFILE" && { echo "STOP: the Cap07 Python file still has the old (2/3) f boundary lines"; exit 1; }
[ "$(grep -c '(1/3)\*__fi\[[0-9]*, *[^]]*\] + (1/3)\*__fi' "$PYFILE")" -ge 30 ] || { echo "STOP: fewer than 30 corrected f boundary lines in the Cap07 Python file"; exit 1; }
grep -q "add_argument('--vf_W'" "$PYFILE" || { echo "STOP: the Cap07 Python file has no --vf_W option"; exit 1; }
grep -q "def get_iterations_of_interest" "$PYFILE" || { echo "STOP: get_iterations_of_interest is missing in the Cap07 Python file"; exit 1; }
grep -q "def gather_full_volume" "$PYFILE" || { echo "STOP: gather_full_volume is missing in the Cap07 Python file"; exit 1; }
grep -q "Cap07_MPI_bwUniCluster.py" "$SCRIPT" || { echo "STOP: the Cap07 job script does not call the Cap07 Python file"; exit 1; }
grep -q -- "--vf_W " "$SCRIPT" || { echo "STOP: the Cap07 job script does not pass --vf_W"; exit 1; }
grep -q $'\r' "$SCRIPT" && { echo "STOP: the Cap07 job script has CR bytes"; exit 1; }

for W in $WIDTHS; do
    OLD=$(find . -maxdepth 4 -type d -name "*Cap07*Theta_60.0_alpha90_sigma0.5_W${W}_periodicx" 2>/dev/null)
    [ -z "$OLD" ] || { echo "STOP: a Cap07 output folder for W=$W already exists - delete or rename it first:"; echo "$OLD"; exit 1; }
done

submit() {   # W
    sbatch --parsable --job-name="C07_W$1" --time=3:00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=437 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=60,TAUF=0.65,MULT=1,SIGMA=0.5,VFW="$1" \
        "$SCRIPT"
}

JOBS=""
for W in $WIDTHS; do
    J=$(submit "$W")
    JOBS="$JOBS,$J"
    echo "submitted W=$W as job $J"
done
JOBS=${JOBS#,}

# diagnostics (read only)
sleep 5
echo "=== squeue (state, reason, start estimate)"
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.6D %.7C %R"
squeue -j "$JOBS" --start
echo "=== scontrol show job"
for J in ${JOBS//,/ }; do
    echo "--- job $J"
    scontrol show job "$J" | grep -E "JobState|Reason|StartTime|MinMemory|NumNodes|NumCPUs|TimeLimit|Partition"
done
echo "=== sinfo genoa"
sinfo -p genoa -o "%P %T %D"
echo "=== reservations"
scontrol show reservation 2>/dev/null | grep -E "ReservationName|StartTime|EndTime|PartitionName" || echo "none"
