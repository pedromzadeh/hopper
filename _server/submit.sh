if [ ! -d "$_outfiles" ]; then
  mkdir _outfiles
fi
sbatch --array=$1-$2 job.slurm
sqme
