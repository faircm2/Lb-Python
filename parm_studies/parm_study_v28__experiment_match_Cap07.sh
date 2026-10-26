#!/bin/bash
# Experiment-matched stability test (Saliba, Korvink, Brandner 2024: falling water film, vertical open channel 1.2 mm wide, 1 ml/min), Cap07 script.
# Water properties: nu = 1.0e-6 m^2/s, channel width D = 1.2 mm (384 cells, dx = 3.125 um), tau_f = 0.55, rho_L,lat = 10, interface width W = 10,
# contact angle 65 deg, vertical channel (alpha 90), periodic x, tau_g = 0.65, initial liquid depth 51 cells (159 um).
# Numbers computed for these settings:  dt = 1.628e-7 s,  g_lat = 8.32e-8,  Bo = 0.1225 / sigma_lat,  1 sigma_lat = 0.1152 N/m,
#   film Reynolds number (flat film, g h^3 / (3 nu^2)) = 13.2  (the experiment: about 14),
#   sigma_lat 0.625 -> Bo = 0.196 and sigma = 0.072 N/m (water)   <- the experiment-matched case.
# Two jobs, MULT=2 -> 24,002 iterations each (about 2.2 h, limit 4 h):
#   sigma_lat 0.625 : the experiment-matched case.
#   sigma_lat 0.3   : the safety case at the same tau_f = 0.55; if 0.625 fails and 0.3 runs, sigma is the problem; if both fail, tau_f = 0.55 is.
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v28__experiment_match_Cap07.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster.py
SIGMAS="0.625 0.3"

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

for S in $SIGMAS; do
    TAG=$(printf '%g' "$S")
    OLD=$(find . -maxdepth 4 -type d -name "*Cap07*tau_f_0.55*Theta_65.0*sigma${TAG}_W10_nu1e-06_D1.2_fill51_periodicx_x2" 2>/dev/null)
    [ -z "$OLD" ] || { echo "STOP: an output folder from an earlier run with sigma=$S exists - delete or rename it first:"; echo "$OLD"; exit 1; }
done

submit() {   # sigma
    sbatch --parsable --job-name="EXP_s$1" --time=4:00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=437 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA=65,TAUF=0.55,MULT=2,SIGMA="$1",VFW=10,NU=1e-6,DD=1.2e-3,FILL=51 \
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
