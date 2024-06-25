#!/bin/bash

#SBATCH --job-name=DL_PREDICT
#SBATCH --output=LOGS/DL_PREDICT.out
#SBATCH --error=LOGS/DL_PREDICT.err
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --gres=gpu:1

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh 

rm -r $PREDICT_FILES/*

for leadtime in {02..10}; do
    echo "Running DL_PREDICT.py for day$leadtime ..."
    srun --exclusive --ntasks=1 --nodes=1 --gres=gpu:1 python DL_PREDICT.py --lr $LR --bs $BS --lr_factor $LR_FACTOR --filters $FILTERS --mask_type $MASK_TYPE --HPT_path $HPT_PATH --leadtime day$leadtime --dropout $DROPOUT &
    
    sleep 1
done

# Wait for all background jobs to finish
wait