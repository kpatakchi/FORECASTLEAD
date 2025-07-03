#!/bin/bash
#SBATCH --job-name=sss_POST
#SBATCH --output=LOGS/sss_POST.out
#SBATCH --error=LOGS/sss_POST.err
#SBATCH --nodes=1          
#SBATCH --ntasks-per-node=1    
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst

echo "SLURM JOB STARTED: $(date)"
hostname

# Load environment
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train

echo "Starting sss_REF ..."
bash SSS_REF_POST.sh
echo "sss_REF finished."

echo "Starting sss_HRES ..."
#python SSS_HRES_POST.py >> LOGS/sss_POST.out 2>&1
echo "sss_HRES finished."

echo "SLURM JOB COMPLETED: $(date)"
