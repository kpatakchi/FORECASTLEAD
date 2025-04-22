#!/bin/sh

#SBATCH --job-name=SSS_STATS
#SBATCH --output=LOGS/SSS_STATS.out
#SBATCH --error=LOGS/SSS_STATS.err
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train

# remove existing directories:
#rm $STATS/*

for leadtime in {02..10}; do
    echo "Running SSS_STATS.py for day$leadtime ..."
    # to run with sbatch:
    srun -N 1 -n 1 python -u SSS_STATS.py --leadtime $leadtime &
    # to run with python
    #python SSS_STATS.py --leadtime $leadtime &
    sleep 2
done

wait