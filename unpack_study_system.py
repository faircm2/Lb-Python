"""
Unpack study_results.tgz into <out_dir>/<job_name>/ (flat), using the job names of compare_runs_system.py.
The job is recognised from the run-folder name (sigma, tau_f, Theta, alpha, _periodicx tag).
Files older than --since (default: 2026-10-17 11:00 local) are skipped: the closed_x folder shared its name with
an older failed run, whose leftover dumps must not be mixed in.

usage:  python unpack_study_system.py study_results.tgz study
"""
import os
import re
import sys
import tarfile
import time

from compare_runs_system import RUNS

SINCE = time.mktime((2026, 10, 17, 11, 0, 0, 0, 0, -1))


def job_of(folder):
    ramp = re.search(r"_ramp(\d+)", folder)               # runs with a contact-angle ramp share all other parameters with their un-ramped twin
    if ramp:
        return "A_ramp%s" % ramp.group(1)
    g = lambda pat: float(re.search(pat, folder).group(1))
    key = (g(r"_sigma([\d.]+?)(?:_|$)"), g(r"tau_f_([\d.]+?)__"), g(r"Theta_([\d.]+?)_"), g(r"_alpha([\d.]+?)_"),
           1 if "_periodicx" in folder else 0)
    for job, spec in RUNS.items():
        if "ramp" in job:                                   # ramp runs are recognised by the folder name only (above)
            continue
        if tuple(float(v) for v in spec) == key:
            return job
    raise KeyError(folder)


def main(tgz, out):
    n = skipped = 0
    with tarfile.open(tgz) as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            job = job_of(parts[0])
            if m.mtime < SINCE:
                skipped += 1
                continue
            dest = os.path.join(out, job)
            os.makedirs(dest, exist_ok=True)
            with tf.extractfile(m) as src, open(os.path.join(dest, parts[-1]), "wb") as dst:
                dst.write(src.read())
            n += 1
    print("extracted %d files, skipped %d older than 17 Oct 11:00" % (n, skipped))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
