#!/bin/bash
#SBATCH --job-name=lbm_s16
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=3000
#SBATCH --output=lbm_%j.out
#SBATCH --error=lbm_%j.err
## for comparable timings pin the node type: check  sinfo -p cpu -o "%n %f"  then uncomment
#SBATCH --constraint=genoa

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
echo "ranks=${SLURM_NTASKS} node=$(scontrol show hostnames $SLURM_JOB_NODELIST)"

"$MPIRUN" -n ${SLURM_NTASKS} --bind-to core --map-by core "$PY" sim.py
echo "Job finished."
