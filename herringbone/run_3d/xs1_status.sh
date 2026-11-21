#!/bin/bash
# Status of the xs1.0 runs: queue, latest iteration and Ma, first NaN with the Ma history before it, newest dumps, Q table.
# Edit the job list below when jobs change. Usage on NEMO2: bash xs1_status.sh
cd /work/classic/fr_jn194-lbm || exit 1
JOBS="8663380:fill14 8670245:fill9"
date +%H:%M:%S
echo "--- queue (empty = nothing running) ---"
squeue -u fr_jn194 -o "%i %T %M %R" | grep -v 8658711
for p in $JOBS; do
    id=${p%%:*}; name=${p##*:}
    echo "=== $name (job $id) ==="
    grep -a "^iter" lbm_$id.out | awk '{print $2,$7}' > /tmp/ma_$id.txt
    echo "latest logged:      $(tail -n 1 /tmp/ma_$id.txt)"
    n=$(grep -n -m1 nan /tmp/ma_$id.txt | cut -d: -f1)
    if [ -n "$n" ]; then
        echo "first NaN at log line $n; the 8 lines before it (iteration, Ma) and the NaN line:"
        sed -n "$((n>8 ? n-8 : 1)),${n}p" /tmp/ma_$id.txt
    else
        echo "no NaN in the log"
    fi
    echo "newest dump files:"
    ls -lt --time-style=+%H:%M:%S FreesurfaceImages/*xs1.0*_${name}_*/ | sed -n '2,3p'
done
echo "=== Q table ==="
/home/fr/fr_jn194/.conda/envs/lbm/bin/python xs1_status.py
