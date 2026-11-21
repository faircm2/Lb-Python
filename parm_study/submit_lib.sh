# sourced by the submit scripts. usage: sub TAG NTASKS TIMELIMIT [VAR=value ...]
# DRY=1 prints the sbatch lines instead of submitting.
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs runs
sub() {
    local tag=$1 ntasks=$2 tlim=$3; shift 3
    local cmd=(env STUDY_TAG="$tag" "$@" sbatch --ntasks="$ntasks" --time="$tlim" --job-name="$tag" --output="logs/${tag}_%j.out" --error="logs/${tag}_%j.err" study_job.sh)
    if [ -n "$DRY" ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
}
