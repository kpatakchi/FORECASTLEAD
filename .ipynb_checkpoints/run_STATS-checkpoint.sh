#!/bin/sh

#SBATCH --job-name=STATS
#SBATCH --output=LOGS/STATS.out
#SBATCH --error=LOGS/STATS.err
#SBATCH --time=00:10:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train

# remove existing directories:
rm $STATS/*

for leadtime in {02..10}; do
    echo "Running STATS.py for day$leadtime ..."
    srun -N 1 -n 1 python STATS.py --leadtime $leadtime &
    sleep 1000
done