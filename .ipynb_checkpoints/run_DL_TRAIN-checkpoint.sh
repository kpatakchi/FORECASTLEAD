#!/bin/sh

#SBATCH --job-name=DL_TRAIN
#SBATCH --output=LOGS/DL_TRAIN.out
#SBATCH --error=LOGS/DL_TRAIN.err
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh

# remove existing directories:
rm -r $HPT_DIR/*

# Record the start time
start_time=$(date +%s)

# Run DL_TRAIN.py for each lead time in parallel
for leadtime in {02..10}; do
    echo "Running DL_TRAIN.py for day$leadtime ..."
    srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=48 python DL_TRAIN.py --lr $LR --bs $BS --lr_factor $LR_FACTOR --filters $FILTERS --mask_type $MASK_TYPE --HPT_path $HPT_PATH --leadtime day$leadtime &
    sleep 1
done

# Wait for all background jobs to finish
wait

end_time=$(date +%s)
total_time=$((end_time - start_time))
total_time_in_minutes=$(echo "scale=2; $total_time / 60" | bc)

echo "Total time taken: $total_time_in_minutes minutes"
