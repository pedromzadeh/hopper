#!/bin/bash
#SBATCH --account=bcamley1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=zadeh@jhu.edu
#SBATCH --time=4:00:00
#SBATCH -o ./_outfiles/job_%A_run_%a.out

python3.10 single_run.py $SLURM_ARRAY_TASK_ID
