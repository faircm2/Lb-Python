#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=6000
# ntasks, job name, time limit and output file are given by submit_lib.sh on the sbatch command line

module purge
module load lang/miniforge3
source /opt/eb/milan/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate lbm

# conda activate does not stick in batch shells here: absolute paths
PY=/home/fr/fr_jn194/.conda/envs/lbm/bin/python
MPIRUN=/home/fr/fr_jn194/.conda/envs/lbm/bin/mpirun
export CC=gcc
export CXX=g++
export OMPI_MCA_pml=ucx

cd "$SLURM_SUBMIT_DIR"
export STUDY_OUT="$SLURM_SUBMIT_DIR/runs/${STUDY_TAG}"
mkdir -p "$STUDY_OUT"

echo "tag = ${STUDY_TAG}"
echo "Running with ${SLURM_NTASKS} ranks on ${SLURM_JOB_NUM_NODES} node(s)"
echo "Node list: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
env | grep '^STUDY_' | sort
echo "TAUF=${TAUF:-1.0} TAUG=${TAUG:-0.65} VFW=${VFW:-4} THETA=${THETA:-60} SIGMA=${SIGMA:-0.01}"

"$MPIRUN" -n ${SLURM_NTASKS} --bind-to core --map-by core \
    "$PY" free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap07_MPI_bwUniCluster_channel_study.py \
        --alpha ${ALPHA:-90} \
        --vf_sigma ${SIGMA:-0.01} \
        --vf_theta ${THETA:-60} \
        --tau_f ${TAUF:-1.0} \
        --tau_g ${TAUG:-0.65} \
        --multiples 1 \
        --rho_G ${RHOG:-0.01} \
        --rho_L ${RHOL:-10} \
        --periodic_x ${PERIODIC_X:-1} \
        --vf_W ${VFW:-4} \
        --nu ${NU:-2.9e-6} \
        --D ${DD:-1e-3} \
        --fill ${FILL:-38}

rc=$?
# the crash buffers are only useful after a crash
[ $rc -eq 0 ] && rm -rf "$STUDY_OUT"/*/ring
echo "Job finished, exit code $rc."
