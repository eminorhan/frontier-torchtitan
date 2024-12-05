job_id=$(sbatch train_8B.sh | awk '{print $4}')  # Submit the first job and capture its job ID

for i in {2..10}
do
  job_id=$(sbatch --dependency=afterany:$job_id train_8B.sh | awk '{print $4}')  # Submit job with dependency
done