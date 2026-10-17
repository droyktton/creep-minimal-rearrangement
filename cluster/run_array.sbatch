#!/bin/bash
# One run per array task, from a task list (make_tasks.py): line SLURM_ARRAY_TASK_ID is
# "OUTDIR ARGS...", OUTDIR relative to RUNS (default ~/runs/mr).
# usage: sbatch --array=1-$(wc -l < tasks.txt)%200 cluster/run_array.sbatch tasks.txt
# Runs restart from their checkpoint (written every 30 min): resubmit the same list after a
# failure or a cancel; finished runs exit at once.
#SBATCH -p knl_tacc
#SBATCH -J mr_creep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -o logs/%x_%A_%a.out
TASKS=$(readlink -f "$1")
ROOT=${MR_ROOT:-$HOME/Codigos/creep-minimal-rearrangement}
BIN=$ROOT/build/mr_creep_${SLURM_JOB_PARTITION:-knl_tacc}
RUNS=${RUNS:-$HOME/runs/mr}
read -r OUT ARGS <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")"
mkdir -p "$RUNS/$OUT"
echo "$(hostname) $OUT $ARGS"
export OMP_NUM_THREADS=1
# SIGTERM (scancel, node shutdown) writes a checkpoint before exiting
srun "$BIN" $ARGS --restart --checkpoint 1800 --out "$RUNS/$OUT"
echo "exit $?"
