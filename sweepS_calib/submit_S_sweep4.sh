#!/bin/bash
# Submits the long S sweep (S = 0.50 ... 0.65, 30,000 iterations) to bwUniCluster: one single-core job per S as a Slurm job array, plus an analysis
# job that starts when all of them have ended.  Cap102 is a 2D single-core NumPy script, so every job asks for 1 core and 3000 MB.
# Run ONCE from the folder that holds the five files:  bash submit_S_sweep4.sh
#   free_surface_SchanChen_nondim_D2Q9_08_Cap102.py  plotter_2d.py  run_S_sweep.py  run_S_sweep4.py  submit_S_sweep4.sh
# Results: sweepS_results4/ (calibration.csv, calibration.png, convergence.csv, convergence.png, one .log per run).

PY=/home/fr/fr_jn194/.conda/envs/lbm/bin/python

for F in free_surface_SchanChen_nondim_D2Q9_08_Cap102.py plotter_2d.py run_S_sweep.py run_S_sweep4.py; do
    [ -f "$F" ] || { echo "STOP: $F is not in this folder"; exit 1; }
done
[ -d sweepS_results4 ] && { echo "STOP: sweepS_results4 exists - rename or delete it first"; exit 1; }
"$PY" run_S_sweep4.py prepare || { echo "STOP: prepare failed"; exit 1; }
N=$(wc -l < sweepS_results4/runs.txt)

J=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=Ssweep4
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=08:00:00
#SBATCH --array=1-$N
#SBATCH --output=sweepS_results4/slurm_%A_%a.out
NAME=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" sweepS_results4/runs.txt)
$PY \$NAME.py > sweepS_results4/\$NAME.log 2>&1
EOF
)
echo "submitted array job $J ($N runs)"
A=$(sbatch --parsable --dependency=afterany:$J --job-name=Ssweep4_an --partition=genoa --ntasks=1 --mem-per-cpu=3000 --time=00:20:00 \
    --output=sweepS_results4/slurm_analyse_%j.out --wrap="$PY run_S_sweep4.py analyse")
echo "submitted analysis job $A (starts when the array has ended)"

sleep 5
echo "=== squeue (state, reason)"
squeue -u "$USER" -o "%.12i %.12j %.8T %.10M %.6D %R"
