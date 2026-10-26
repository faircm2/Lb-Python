#!/bin/bash
# Surface-tension ladder at the stable interface width W = 10 (corrected f wall boundary, Cap07 script).
# theta 60, tau_f = tau_g = 0.65, alpha 90, rho_L 10, rho_G 0.01, periodic x.   MULT=2 -> 24,002 iterations, about 2.2 h per job (limit 4 h).
# Five jobs of 2048 ranks (32 nodes each); the cluster runs about three at a time, the others wait in the queue.
#
# Bond number Bo = rho g w^2 / sigma of this setup (nu_phys 2.9e-6, physical g, 384 cells across the channel width w):
#   channel width D = 1.0 mm (current script):  Bo = 0.0759 / sigma_lat    sigma_lat 0.39 -> 0.195   0.50 -> 0.152   0.65 -> 0.117   0.80 -> 0.095   1.0 -> 0.076
#   channel width D = 1.2 mm (the experiment):  Bo = 0.1311 / sigma_lat    sigma_lat 0.39 -> 0.336   0.50 -> 0.262   0.65 -> 0.202   0.80 -> 0.164   1.0 -> 0.131
# The experiment has Bo = 0.196: sigma_lat 0.387 for D = 1.0 mm, sigma_lat 0.669 (= water, 0.072 N/m) for D = 1.2 mm.  Both values are inside this ladder.
# The film Reynolds number of these runs is about 0.4 (the experiment: about 14); Re is not matched here.
#
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v27__W10_sigma_ladder_Cap07.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.py
SIGMAS="0.39 0.5 0.65 0.8 1.0"

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

for S in $SIGMAS; do
    TAG=$(printf '%g' "$S")
    OLD=$(find . -maxdepth 4 -type d -name "*Cap07*Theta_60.0_alpha90_sigma${TAG}_W10_periodicx_x2" 2>/dev/null)
    [ -z "$OLD" ] || { echo "STOP: an output folder from an earlier run with sigma=$S exists - delete or rename it first:"; echo "$OLD"; exit 1; }
done

submit() {   # sigma
    sbatch --parsable --job-name="S$1_W10" --time=4:00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=437 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=60,TAUF=0.65,MULT=2,SIGMA="$1",VFW=10 \
        "$SCRIPT"
}

JOBS=""
for S in $SIGMAS; do
    J=$(submit "$S")
    JOBS="$JOBS,$J"
    echo "submitted sigma=$S as job $J"
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
