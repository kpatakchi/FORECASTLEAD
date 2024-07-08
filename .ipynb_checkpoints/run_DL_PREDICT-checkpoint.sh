#!/bin/sh

#SBATCH --job-name=DL_PREDICT
#SBATCH --output=LOGS/DL_PREDICT.out
#SBATCH --error=LOGS/DL_PREDICT.err
#SBATCH --time=01:30:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train


# Run DL_TRAIN.py for each lead time in parallel
for leadtime in {02..10}; do
    lead_day_file="day${leadtime}.csv"
    echo "Setting hyperparameters for $lead_day_file ..."
    
    # Call DL_settings.sh with the lead day file to set the hyperparameters
    source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh "$lead_day_file"
    
    # Now the hyperparameters LR, BS, and DROPOUT should be updated for the current lead day
    echo "Running DL_TRAIN.py for day$leadtime with LR: $LR, BS: $BS, DROPOUT: $DROPOUT ..."
    srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=4 python DL_PREDICT.py --lr $LR --bs $BS --lr_factor $LR_FACTOR --filters $FILTERS --mask_type $MASK_TYPE --HPT_path $HPT_PATH --leadtime "day$leadtime" --dropout $DROPOUT &
    sleep 180
done

