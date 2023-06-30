#!/bin/bash
#SBATCH --account=bcamley1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=end
#SBATCH --mail-user=zadeh@jhu.edu
#SBATCH --time=0:05:00
#SBATCH -o ./_outfiles/zipping_job_%A.out

python3.10 zip_data.py
