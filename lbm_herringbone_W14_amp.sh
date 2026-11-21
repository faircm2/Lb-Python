#!/bin/bash
#SBATCH --job-name=lbm_W14amp
#SBATCH --partition=genoa
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=6000
#SBATCH --time=20:00:00
#SBATCH --exclude=n4102
#SBATCH --signal=B:USR1@180
#SBATCH --output=lbm_W14amp_%j.out
#SBATCH --error=lbm_W14amp_%j.err

# ------------------------------------------------------------------------------------------------------------------------------
# W = 14 run with the amplitude monitor: no failure in 156013 (series) or 624052 iterations (job 8878871) so far.
# All other parameters as in the series; --multiples 100 = 1200100 iterations (about 14 h at 0.041 s per iteration on 512 ranks) as the upper limit; the amplitude monitor (steady_amplitude.py) ends the run earlier.
# The monitor stops the run when the oscillation amplitude of max|u| is constant (AMP_* variables below); the output folder name ends with _x100.
# The run stops at the first NaN in the log (watchdog); a line is appended to W_series_summary.txt.
# Submit from the workspace folder:   cd /work/classic/fr_jn194-lbm && sbatch lbm_herringbone_W14_long.sh
# ------------------------------------------------------------------------------------------------------------------------------
W=14
MULT=100
SOLVER=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap08_MPI_bwUniCluster_herringbone_idealised.py
OUT=lbm_W14amp_${SLURM_JOB_ID}.out

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

# amplitude monitor (steady_amplitude.py): stop when the amplitudes of 4 consecutive batches of 20000 iterations agree within 5 %, batches before iteration 40000 ignored
export AMP_CUTOFF=0.05
export AMP_STRIDE=10
export AMP_BATCH=20000
export AMP_NBATCH=4
export AMP_START=40000

echo "W=${W} amplitude-monitor run: up to ${MULT} multiples, ${SLURM_NTASKS} ranks"
echo "Node list: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

# an older output folder of the same run would mix with this one: move it aside
mkdir -p old_runs
for d in FreesurfaceImages/*_W${W}_xs1.0_*_x${MULT}*; do
    [ -d "$d" ] && mv "$d" "old_runs/W${W}_x${MULT}_before_long_$(date +%m%d%H%M%S)"
done

"$MPIRUN" -n ${SLURM_NTASKS} --bind-to core --map-by core \
    "$PY" "$SOLVER" \
        --alpha 90 --vf_sigma 0.1 --vf_theta 60 --tau_f 0.6 --multiples ${MULT} \
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
echo "W=${W} x${MULT} amplitude monitor job=${SLURM_JOB_ID} | last finite: ${LAST} | first nan: ${FIRST:-none}" >> W_series_summary.txt
echo "Job finished."
