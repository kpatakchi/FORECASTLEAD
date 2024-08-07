#!/bin/sh

# Default values
LEADTIME=1
UNET_TYPE=default
TIME_LIMIT=01:00:00

while getopts "l:u:t:" opt; do
    case ${opt} in
        l ) LEADTIME=$OPTARG ;;
        u ) UNET_TYPE=$OPTARG ;;
        t ) TIME_LIMIT=$OPTARG ;;
        \? ) echo "Usage: cmd [-l leadtime] [-u unet_type] [-t time_limit]"
             exit 1 ;;
    esac
done

#SBATCH --job-name=DL_HPT_leadtime_${LEADTIME}_${UNET_TYPE}
#SBATCH --output=LOGS/DL_HPT_${LEADTIME}_${UNET_TYPE}.out
#SBATCH --error=LOGS/DL_HPT_${LEADTIME}_${UNET_TYPE}.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4

# Reload environment settings
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh

# Iterate over dropout, learning rate, and batch size
for dropout_value in 0 0.1 0.2 0.3; do
    for lr_value in 0.01 0.001 0.0001 0.00001; do
        for bs_value in 2 4 8 16; do
            echo "Running DL_TRAIN.py for day${LEADTIME} with dropout=${dropout_value}, lr=${lr_value}, bs=${bs_value}, unet_type=${UNET_TYPE} ..."
            srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=4 \
                python DL_TRAIN.py --lr ${lr_value} --bs ${bs_value} --lr_factor ${LR_FACTOR} \
                --filters ${FILTERS} --mask_type ${MASK_TYPE} --HPT_path ${HPT_PATH} \
                --leadtime day${LEADTIME} --dropout ${dropout_value} --unet_type ${UNET_TYPE} &
            sleep 4
        done
    done
done

# Wait for all background jobs to finish
wait
