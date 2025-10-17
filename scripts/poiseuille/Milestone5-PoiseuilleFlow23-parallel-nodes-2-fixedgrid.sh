#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=96
#SBATCH --mail-user=johan.neethling@email.uni-freiburg.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mem=380000mb
#SBATCH --job-name=poiseuille_nodes_2

echo "Loading Python module and MPI module"
source ~/env3/bin/activate
module load devel/python/3.12.3_gnu_14.2
module load mpi/openmpi/5.0
module list
startexe="mpirun -n 192 python3 Milestone5-PoiseuilleFlow23-parallel-nodes-2-fixedgrid.py"
exec $startexe