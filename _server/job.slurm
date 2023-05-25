#!/bin/bash
#SBATCH --account=bcamley1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mail-type=end
#SBATCH --mail-user=zadeh@jhu.edu
#SBATCH --time=3-0:00:00
#SBATCH -o ./_outfiles/job_%A_run_%a.out

python3.10 parallel_runs.py $SLURM_ARRAY_TASK_ID