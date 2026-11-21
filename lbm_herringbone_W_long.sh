#!/bin/bash
#SBATCH --job-name=lbm_Wlong
#SBATCH --partition=genoa
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=6000
#SBATCH --time=16:00:00
#SBATCH --array=0-3%2
#SBATCH --exclude=n4102
#SBATCH --signal=B:USR1@180
#SBATCH --output=lbm_Wlong_%A_%a.out
#SBATCH --error=lbm_Wlong_%A_%a.err

# ------------------------------------------------------------------------------------------------------------------------------
# Long confirmation runs for the interface widths that did not fail (W = 13, 14, 15, 16), one array task per W; all other parameters as in the W = 14 amplitude run (job 8885554)
#   (xs1.0 herringbone, sigma 0.1, theta 60, tau_f 0.6, nu 1e-6, fill 9.4, 512 ranks = 0.041 s per iteration).
# The amplitude monitor (steady_amplitude.py) ignores everything before iteration 400000 and then needs 6 consecutive batches of 20000 iterations with amplitudes within 5 %:
#   earliest stop at iteration 520000 (6.1 s physical, about 6 h); --multiples 110 = 1320110 iterations (15.5 s, about 15 h) is the upper limit, the time limit is 16 h.
# The output folders end with _x110.
# Each task stops its run at the first NaN in the log (watchdog) and appends one line to W_series_summary.txt.
# Submit from the workspace folder:   cd /work/classic/fr_jn194-lbm && sbatch lbm_herringbone_W_long.sh
# Throttle: --array=0-3%2 runs at most two W values (1024 cores) at the same time.
# ------------------------------------------------------------------------------------------------------------------------------
WLIST=(13 14 15 16)
W=${WLIST[$SLURM_ARRAY_TASK_ID]}
SOLVER=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap08_MPI_bwUniCluster_herringbone_idealised.py
OUT=lbm_Wlong_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

module purge
module load lang/miniforge3
source /opt/eb/milan/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate lbm
PY=/home/fr/fr_jn194/.conda/envs/lbm/bin/python
MPIRUN=/home/fr/fr_jn194/.conda/envs/lbm/bin/mpirun
export CC=gcc
export CXX=g++
export OMPI_MCA_pml=ucx
export PYTHONUNBUFFERED=1

# geometry and run parameters (environment variables read by the solver)
export D_ND=64
export XN=736
export TP=4.9
export XS=1.0

# amplitude monitor (steady_amplitude.py): stop when the amplitudes of 6 consecutive batches of 20000 iterations agree within 5 %, batches before iteration 400000 ignored
export AMP_CUTOFF=0.05
export AMP_STRIDE=10
export AMP_BATCH=20000
export AMP_NBATCH=6
export AMP_START=400000

echo "W long series (amplitude monitor): array task ${SLURM_ARRAY_TASK_ID}, W = ${W}, ${SLURM_NTASKS} ranks"
echo "Node list: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

# an older output folder with the same W would mix with this run: move it aside
mkdir -p old_runs
for d in FreesurfaceImages/*_W${W}_xs1.0_*_x110*; do
    [ -d "$d" ] && mv "$d" "old_runs/W${W}_x110_before_long_$(date +%m%d%H%M%S)"
done

"$MPIRUN" -n ${SLURM_NTASKS} --bind-to core --map-by core \
    "$PY" "$SOLVER" \
        --alpha 90 --vf_sigma 0.1 --vf_theta 60 --tau_f 0.6 --multiples 110 \
        --rho_G 0.01 --rho_L 10 --periodic_x 1 --vf_W ${W} --nu 1e-6 --D 1.2e-3 --fill 9.4 --h_backstep 15 &
MPI_PID=$!

# time limit close: stop the run cleanly so that the summary line is still written
trap 'echo "WATCHDOG: time limit close, stopping the run"; kill -TERM $MPI_PID' USR1

# watchdog: stop the run at the first NaN in the log
while kill -0 $MPI_PID 2>/dev/null; do
    sleep 20
    if grep -aq -E "Ma=nan|max\|u\|=nan" "$OUT"; then
        echo "WATCHDOG: NaN found in the log, stopping the run"
        kill -TERM $MPI_PID
        sleep 15
        kill -KILL $MPI_PID 2>/dev/null
        break
    fi
done
wait $MPI_PID 2>/dev/null

LAST=$(grep -a '^iter' "$OUT" | grep -av 'nan' | tail -n 1)
FIRST=$(grep -a '^iter' "$OUT" | grep -a -m1 'nan')
echo "W=${W} x110 long amplitude monitor job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | last finite: ${LAST} | first nan: ${FIRST:-none}" >> W_series_summary.txt
echo "Job finished."
