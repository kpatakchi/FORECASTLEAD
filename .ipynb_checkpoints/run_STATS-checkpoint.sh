#!/bin/sh

#SBATCH --job-name=STATS
#SBATCH --output=LOGS/STATS.out
#SBATCH --error=LOGS/STATS.err
#SBATCH --time=00:05:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train

# remove existing directories:
rm $STATS/*

for leadtime in {02..10}; do
    echo "Running STATS.py for day$leadtime ..."
    # to run with sbatch:
    srun -N 1 -n 1 python STATS.py --leadtime $leadtime &
    # to run with python
    #python STATS.py --leadtime $leadtime &
    sleep 1
done

wait