#!/bin/bash

#SBATCH --job-name=DL_PREP
#SBATCH --output=LOGS/DL_PREP.out
#SBATCH --error=LOGS/DL_PREP.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train

rm -r $TRAIN_FILES/*
rm -r $PRODUCE_FILES/*
#mkdir $TRAIN_FILES

for leadtime in {02..10}; do
    echo "Running DL_PREP.py for day$leadtime ..."
    srun --exclusive --ntasks=1 --nodes=1 python DL_PREP.py --leadtime day$leadtime &
    sleep 120
done

# Wait for all background jobs to finish
wait