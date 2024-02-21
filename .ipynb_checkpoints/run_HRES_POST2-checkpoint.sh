#!/bin/bash

#SBATCH --job-name=HRES_POST2_job
#SBATCH --output=LOGS/HRES_POST2_job.out
#SBATCH --error=LOGS/HRES_POST2_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf

# Your script file name
source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv
rm $HRES_POST/* $HRES_DUMP4/* $HRES_LOG/*CDO*

script="/p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/HRES_POST2.sh"

# Loop over years
for year in {2018..2018}; do
    # Loop over months
    for month in {01..12}; do
        start="${year}${month}01 13"
        job_name="HRES_POST2_$(date -d "$start" "+%Y%m%d_%H%M%S")"
        # Check the number of submitted jobs
        while true; do
            num_jobs=$(squeue -u patakchiyousefi1 | wc -l)
            if ((num_jobs <= 3)); then
                echo "submitted $start job!"
                sbatch <<EOL
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=LOGS/HRES_POST2.out
#SBATCH --error=LOGS/HRES_POST2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:15:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf

$script "$start"
EOL
                break
            else
                echo "Too many submitted jobs; Sleeping Zzz ..."
                sleep 600
            fi
        done
    done
done
