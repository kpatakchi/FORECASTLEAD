#!/bin/bash

#SBATCH --job-name=DL_PREP
#SBATCH --output=LOGS/DL_PREP.out
#SBATCH --error=LOGS/DL_PREP.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv
#module load TensorFlow matplotlib xarray
source /p/project/deepacf/kiste/patakchiyousefi1/SC_VENV/bin/activate prc_env

rm -r $TRAIN_FILES
mkdir $TRAIN_FILES

for leadtime in {02..02}; do
    echo "Running DL_PREP.py for day$leadtime ..."
    python DL_PREP.py --leadtime day$leadtime
done
