#!/bin/bash
#SBATCH --job-name=lbm_r32
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --time=07:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --output=lbm_%j.out
#SBATCH --error=lbm_%j.err

# ------------------------------------------------------------------
# scaling run: 192^3 lattice, dims = [4,4,2] (32 ranks), 3000 iterations
# ------------------------------------------------------------------
module purge
module load lang/miniforge3
source /opt/eb/milan/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate lbm

PY=/home/fr/fr_jn194/.conda/envs/lbm/bin/python
MPIRUN=/home/fr/fr_jn194/.conda/envs/lbm/bin/mpirun

export CC=gcc
export CXX=g++
export OMPI_MCA_pml=ob1
export UCX_TLS=sm,self

echo "python = $PY"
"$PY" -c "import numpy, scipy, matplotlib, mpi4py; from mpi4py import MPI; print('imports OK:', MPI.Get_library_version().splitlines()[0])"

echo "Running with ${SLURM_NTASKS} ranks on ${SLURM_JOB_NUM_NODES} node(s)"
echo "Node list: $(scontrol show hostnames $SLURM_JOB_NODELIST)"

"$MPIRUN" -n ${SLURM_NTASKS} --bind-to core --map-by core "$PY" sim.py

echo "Job finished."
