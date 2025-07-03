#!/bin/bash

for leadtime in {02..10}; do
    echo "Submitting job for leadtime $leadtime ..."
    
    sbatch --job-name=SSS_STATS_day${leadtime} \
           --output=LOGS/SSS_STATS_day${leadtime}.out \
           --error=LOGS/SSS_STATS_day${leadtime}.err \
           --time=01:00:00 \
           --partition=booster \
           --mail-user=k.patakchi.yousefi@extern.fz-juelich.de \
           --mail-type=ALL \
           --account=esmtst \
           --nodes=1 \
           --ntasks=1 \
           --wrap="source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train && python -u SSS_STATS.py --leadtime $leadtime"
done
