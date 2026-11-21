#!/bin/bash
#SBATCH --job-name=lbm_backstep_a90
#SBATCH --partition=genoa
#SBATCH --ntasks=1024
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=6000
#SBATCH --time=30:00:00
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
# D_ND (grid size) is read by the python script from the environment
export D_ND=${D_ND:-384}
echo "D_ND = ${D_ND}"
echo "script = ${SCRIPT:-free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap08_MPI_bwUniCluster_backstep.py}, PLANE_X = ${PLANE_X:-default}"
echo "Running with ${SLURM_NTASKS} ranks on ${SLURM_JOB_NUM_NODES} node(s)"
echo "Node list: $(scontrol show hostnames $SLURM_JOB_NODELIST)"

"$MPIRUN" -n ${SLURM_NTASKS} --bind-to core --map-by core \
    "$PY" ${SCRIPT:-free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap08_MPI_bwUniCluster_backstep.py} \
        --alpha ${ALPHA:-90} \
        --vf_sigma ${SIGMA:-0.01} \
        --vf_theta ${THETA:-60} \
        --tau_f ${TAUF:-1.0} \
        --multiples ${MULT:-1} \
        --rho_G ${RHOG:-0.01} \
        --rho_L ${RHOL:-10} \
        --periodic_x ${PERIODIC_X:-1} \
        --vf_W ${VFW:-4} \
        --nu ${NU:-2.9e-6} \
        --D ${DD:-1e-3} \
        --fill ${FILL:-38} \
        --h_backstep ${HBS:-25} \
        --x_backstep ${XBS:-200} \
        --x_forwardstep ${XFS:-640}

echo "Job finished."

# ------------------------------------------------------------------
# The python script needs 8, 512, 1024 or 2048 MPI ranks (assert in the script).
# D_ND = 64 runs use 8 ranks (decomposition 2 x 2 x 2).
#
# (a) alpha 90 repeat of: tau_f 1.0, nu 2.9e-6, h 4, fill 20, x1, vf_W 4
# sbatch --ntasks=8 --export=ALL,D_ND=64,ALPHA=90,TAUF=1.0,NU=2.9e-6,MULT=1,FILL=20,HBS=4,XBS=32,XFS=96,VFW=4,SIGMA=0.01,THETA=60,RHOG=0.01,RHOL=10,PERIODIC_X=1,DD=1e-3 free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap08_MPI_bwUniCluster_backstep_a90.sh
#
# (b) alpha 90 repeat of: tau_f 0.6, nu 1.3e-6, h 4, fill 20, x2, vf_W 4
# sbatch --ntasks=8 --export=ALL,D_ND=64,ALPHA=90,TAUF=0.6,NU=1.3e-6,MULT=2,FILL=20,HBS=4,XBS=32,XFS=96,VFW=4,SIGMA=0.01,THETA=60,RHOG=0.01,RHOL=10,PERIODIC_X=1,DD=1e-3 free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap08_MPI_bwUniCluster_backstep_a90.sh
#
# (d4 with 3-D output) same parameters as d4, y-z cross-section at x = 40 (through the vortex)
# sbatch --ntasks=8 --export=ALL,SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap08_MPI_bwUniCluster_backstep_3d.py,PLANE_X=40,D_ND=64,ALPHA=90,TAUF=0.575,NU=1.47e-6,MULT=2,FILL=22,HBS=12,XBS=32,XFS=96,VFW=6,SIGMA=0.01,THETA=60,RHOG=0.01,RHOL=10,PERIODIC_X=1,DD=1e-3 free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap08_MPI_bwUniCluster_backstep_a90.sh
# ------------------------------------------------------------------
