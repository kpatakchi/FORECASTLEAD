#!/bin/bash

#outer job for managing the job submissions in the loop

#SBATCH --job-name=HRES_POST_job
#SBATCH --output=LOGS/HRES_POST_job.out
#SBATCH --error=LOGS/HRES_POST_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst

source bashenv

RUN_FIRST_PART=false  # Change to false to disable the first part

if [ "$RUN_FIRST_PART" = true ]; then

    rm $HRES_DUMP/*
    rm $HRES_DUMP2/*
    rm -rf $HRES_DUMP3 && mkdir $HRES_DUMP3
    rm -rf $HRES_DUMP4 && mkdir $HRES_DUMP4
    rm -rf $HRES_DUMP5 && mkdir $HRES_DUMP5
    
    # run the first part of post-processing:
    source HRES_POST.sh &> LOGS/HRES_POST.log
fi

RUN_SECOND_PART=true  # Change to true to enable

# run the second part only if the flag is true
if [ "$RUN_SECOND_PART" = true ]; then

    rm $HRES_POST/*

    script="/p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/HRES_POST2.sh"
    for year in {2021..2021}; do
        for month in {01..12}; do
            start="${year}${month}01 13"
            job_name="HRES_POST_$(date -d "$start" "+%Y%m%d_%H%M%S")"
            while true; do
                num_jobs=$(squeue -u patakchiyousefi1 | wc -l)
                if ((num_jobs <= 25)); then
                    echo "submitted $start job!"
                    sbatch <<EOL
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=LOGS/$job_name.out
#SBATCH --error=LOGS/$job_name.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst

$script "$start"
EOL
                    break
                else
                    echo "Too many submitted jobs; waiting ..."
                    sleep 600
                fi
            done
        done
    done

rm -r $HRES_DUMP $HRES_DUMP2

fi

