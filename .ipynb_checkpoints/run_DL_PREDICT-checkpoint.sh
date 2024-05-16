#!/bin/bash

#SBATCH --job-name=DL_PREDICT
#SBATCH --output=LOGS/DL_PREDICT.out
#SBATCH --error=LOGS/DL_PREDICT.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:08:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst
#SBATCH --gres=gpu:4

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train

rm -r $PREDICT_FILES/*
#mkdir $TRAIN_FILES

for leadtime in {10..10}; do
    echo "Running DL_PREDICT.py for day$leadtime ..."
    srun --exclusive --ntasks=1 --nodes=1 --gres=gpu:4 python DL_PREDICT.py --lr 0.01 --bs 16 --lr_factor 0.5 --filters 64 --mask_type "no_na" --HPT_path "HPT/" --leadtime day$leadtime
    sleep 2
done

# Wait for all background jobs to finish
wait