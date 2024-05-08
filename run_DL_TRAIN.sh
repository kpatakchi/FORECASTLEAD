#!/bin/sh

#SBATCH --job-name=DL_TRAIN
#SBATCH --output=LOGS/DL_TRAIN.out
#SBATCH --error=LOGS/DL_TRAIN.err
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
#source /p/project/deepacf/kiste/patakchiyousefi1/SC_VENV/bin/activate prc_env

# remove existing directories:
rm -r $HPT_DIR/*

# Record the start time
start_time=$(date +%s)

for leadtime in {02..10}; do
    echo "Running DL_TRAIN.py for day$leadtime ..."
    srun -N 1 -n 1 --gres=gpu:4 python DL_TRAIN.py --lr 0.01 --bs 16 --lr_factor 0.5 --filters 64 --mask_type "no_na" --HPT_path "HPT/" --leadtime day$leadtime &
    sleep 1
done

# Wait for all background jobs to finish
wait

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
total_time=$((end_time - start_time))

# Convert the total time to minutes
total_time_in_minutes=$(echo "scale=2; $total_time / 60" | bc)

echo "Total time taken: $total_time_in_minutes minutes"