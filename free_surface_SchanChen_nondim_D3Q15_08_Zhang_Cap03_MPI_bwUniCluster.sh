#!/bin/bash
#SBATCH --job-name=lbm_zhang_cap02
#SBATCH --partition=genoa
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=06:00:00
#SBATCH --output=lbm_%j.out
#SBATCH --error=lbm_%j.err

# ------------------------------------------------------------------
# Load modules
# ------------------------------------------------------------------
module purge

module load lang/miniforge3
source /opt/eb/milan/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate lbm

# conda activate does not stick in batch shells here, so call the env
# binaries by absolute path instead.
PY=/home/fr/fr_jn194/.conda/envs/lbm/bin/python
MPIRUN=/home/fr/fr_jn194/.conda/envs/lbm/bin/mpirun

export CC=gcc
export CXX=g++
#export OMPI_MCA_pml=ob1
export OMPI_MCA_pml=ucx

# Verify
echo "CC = $(which $CC)"
echo "CXX = $(which $CXX)"
echo "python = $PY"
"$PY" -c "import numpy, scipy, matplotlib, mpi4py; from mpi4py import MPI; print('imports OK:', MPI.Get_library_version().splitlines()[0])"

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
echo "Running with ${SLURM_NTASKS} ranks on ${SLURM_JOB_NUM_NODES} node(s)"
echo "Node list: $(scontrol show hostnames $SLURM_JOB_NODELIST)"

"$MPIRUN" -n ${SLURM_NTASKS} --bind-to core --map-by core \
    "$PY" free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap03_MPI_bwUniCluster.py

echo "Job finished."