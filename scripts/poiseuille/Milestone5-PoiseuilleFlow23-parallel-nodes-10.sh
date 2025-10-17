#!/bin/bash
#SBATCH --nodes=10
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=96
#SBATCH --mail-user=johan.neethling@email.uni-freiburg.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mem=380000mb
#SBATCH --job-name=poiseuille_nodes_1

# Get NODES from SLURM
NODES=$SLURM_NNODES
if [[ -z "$NODES" ]]; then
    echo "Error: SLURM_NNODES not set"
    exit 1
fi

echo "Loading Python module and MPI module"
source ~/env3/bin/activate
module load devel/python/3.12.3_gnu_14.2
module load mpi/openmpi/5.0
module list

# Fixed parameters
Yn=50
PYFILE="Milestone5-PoiseuilleFlow23-parallel-variable.py"

# Check if Python file exists
if [[ ! -f "$PYFILE" ]]; then
    echo "Error: Python file $PYFILE does not exist"
    exit 1
fi

# Calculate total processes
PROCESSES=$(( NODES * 96 ))

# Loop over L (1 to 4, step 1)
for L in {1..4..1}; do
    # Calculate Xn
    Xn=$(( NODES * 96 * 50 * L ))

    # Export variables for Python script
    export NODES
    export Xn
    export Yn
    export L

    echo "Running simulation with NODES=$NODES, L=$L, Xn=$Xn, Yn=$Yn, PROCESSES=$PROCESSES"
    mpirun -n ${PROCESSES} --report-bindings python3 ./${PYFILE} >> "output_grid_${Yn}x${Xn}_nodes${NODES}_L${L}.log" 2>> "error_nodes_${NODES}_L_${L}.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: mpirun failed for NODES=$NODES, L=$L"
        exit 1
    fi
done