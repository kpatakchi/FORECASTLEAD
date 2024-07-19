#!/bin/bash

#SBATCH --job-name=DL_PREP
#SBATCH --output=LOGS/DL_PREP.out
#SBATCH --error=LOGS/DL_PREP.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh 

rm -r $TRAIN_FILES/*
rm -r $PRODUCE_FILES/*
rm minmax_scaling.csv

#mkdir $TRAIN_FILES

for leadtime in {02..10}; do
    echo "Running DL_PREP.py for day$leadtime ..."
    srun --ntasks=1 --nodes=1 python DL_PREP.py --leadtime day$leadtime --mask_type  $MASK_TYPE &
    sleep 30
done

# Wait for all background jobs to finish
wait