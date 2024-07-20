#!/bin/sh

#SBATCH --job-name=DL_HPT
#SBATCH --output=LOGS/DL_HPT.out
#SBATCH --error=LOGS/DL_HPT.err
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh

# remove existing directories:
rm -r $HPT_DIR/*

# Record the start time
start_time=$(date +%s)

# Run DL_TRAIN.py for each combination of hyperparameters and lead time in parallel
for dropout in 0.1 0.3 0.5 0.7; do
  for lr in 0.01 0.001 0.0001 0.00001; do
    for bs in 4 8 16 32; do
      for leadtime in {02..10}; do
        echo "Running DL_TRAIN.py for day$leadtime with dropout=$dropout, lr=$lr, bs=$bs ..."
        srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=4 python DL_TRAIN.py --lr $lr --bs $bs --lr_factor $LR_FACTOR --filters $FILTERS --mask_type $MASK_TYPE --HPT_path ${HPT_PATH} --leadtime day$leadtime --dropout $dropout &
        sleep 2 # to make sure all 576 jobs fit in 32 nodes over time.
      done
    done
  done
done

# Wait for all background jobs to finish
wait

srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=4 python EXTRACT_HPT.py # to extract the best hyperparameters in a csv file.

# Wait for all background jobs to finish
wait

end_time=$(date +%s)
total_time=$((end_time - start_time))
total_time_in_minutes=$(echo "scale=2; $total_time / 60" | bc)

echo "Total time taken: $total_time_in_minutes minutes"
