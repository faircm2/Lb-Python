#!/bin/bash
# Memory-lean version.  The script no longer builds full-domain arrays on every rank (measured before: about 4.3 GB per rank),
# so each rank needs only a few hundred MB and the job fits the ~33 GB per node that Slurm can still reserve on busy genoa nodes.
# sbatch settings: genoa (the .sh default), 64 ranks per node, 420 MB per rank (= 26.9 GB per node; Slurm can hand out only about 28.2 GB on a busy node after the 5 GB system reserve; overrides the 6000 MB in the .sh header), 16 h.
# Two runs, both periodic x, alpha 90, tau_f = tau_g = 0.65.
#   T90_periodic : sigma_lat 0.05, theta 90  - bubble test (A_periodic showed its first bubble at iteration 36,003).  MULT=4  -> 48,004 iterations
#   B_periodic   : sigma_lat 0.01, theta 60  - the stable setting.                                                   MULT=10 -> 120,010 iterations (about 11.7 h)
# Run ONCE from /work/classic/fr_jn194-lbm/scaling :   bash parm_study_v18__T90_and_B_periodic_lowmem420.sh

SCRIPT=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.sh
PYFILE=free_surface_SchanChen_nondim_D3Q15_08_Zhang_Cap06_MPI_bwUniCluster.py

grep -q snapshot_iterations "$PYFILE" || { echo "STOP: the Python file on the server does not have the snapshot edits - upload it first"; exit 1; }
grep -q "np.meshgrid(np.arange(Xn" "$PYFILE" && { echo "STOP: the Python file on the server still builds full-domain arrays (needs 4+ GB per rank) - upload the lean version first"; exit 1; }

submit() {   # name  sigma  theta  mult
    sbatch --job-name="$1" --time=16:00:00 --ntasks=2048 --ntasks-per-node=64 --mem-per-cpu=420 \
        --export=ALL,ALPHA=90,RHOG=0.01,RHOL=10,PERIODIC_X=1,THETA="$3",TAUF=0.65,MULT="$4",SIGMA="$2" \
        "$SCRIPT"
}

submit T90_periodic 0.05 90 4
submit B_periodic   0.01 60 10
echo "submitted - check with: squeue -u $USER"
