#!/bin/bash

# Define the batch scripts in the desired order

batch_scripts=(
  "run_DL_PREDICT.sh"
  "run_STATS.sh"
)

#batch_scripts=(
#  "run_DL_HPT.sh"
#  "run_DL_PREDICT.sh"
#  "run_STATS.sh"
#)

# Initialize the dependency variable
previous_job_id=""

# Loop through each batch script and submit them sequentially with dependencies
for script in "${batch_scripts[@]}"
do
  if [ -z "$previous_job_id" ]; then
    # Submit the first job without any dependency
    job=$(sbatch "$script")
  else
    # Submit subsequent jobs with dependency on the previous job
    job=$(sbatch --dependency=afterok:$previous_job_id "$script")
  fi
  
  # Extract the job ID from the submission output
  job_id=$(echo $job | awk '{print $4}')
  
  # Update the previous_job_id for the next iteration
  previous_job_id=$job_id
  
  echo "$script submitted with job ID $job_id"
done

echo "All jobs submitted successfully."
