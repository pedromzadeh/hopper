if [ ! -d "_outfiles" ]; then
  mkdir _outfiles
fi
sbatch --array=$1-$2 single_job.slurm
sqme
