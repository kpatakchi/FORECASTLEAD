#!/bin/sh

#SBATCH --job-name=DL_TRAIN
#SBATCH --output=LOGS/DL_TRAIN.out
#SBATCH --error=LOGS/DL_TRAIN.err
#SBATCH --time=00:05:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv
module load TensorFlow matplotlib xarray Python

# remove existing directories:
rm -r $HPT_DIR/*

# Record the start time
start_time=$(date +%s)

for leadtime in {02..10}; do
    echo "Running DL_TRAIN.py for day$leadtime ..."
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 python DL_TRAIN.py --lr 0.001 --bs 16 --lr_factor 0.5 --filters 64 --mask_type "no_na_land" --HPT_path "HPT/" --leadtime day$leadtime &
    sleep 1
done

# Wait for all background jobs to finish
wait

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
total_time=$((end_time - start_time))

echo "Total time taken: $total_time seconds"