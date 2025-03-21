#!/bin/bash

#SBATCH --job-name=touch
#SBATCH --output=LOGS/touch.out
#SBATCH --error=LOGS/touch.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst

#This is for touching all files in scratch

source directories.sh

cd $PSCRATCH_DIR
cd .. #go back one directory
find -exec touch {} \;

echo "done 1" 

cd $PSCRATCH_DIR2
cd .. # go back one directory
find -exec touch {} \;

echo "done 2" 
