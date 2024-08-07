#!/bin/sh

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh

#rm -r $HPT_DIR/*

# Define arrays for parameter values
#unet_type_values=("unet-l" "unet-m" "unet-s" "unet-xs" "unet-l-dw" "unet-m-dw" "unet-s-dw" "unet-xs-dw")
#leadtime_values=("02" "03" "04" "05" "06" "07" "08" "09" "10")  # Adjust as needed

unet_type_values=("trans-unet")
leadtime_values=("10")  # Adjust as needed

# Loop over unet_type and leadtime
for UNET_TYPE in "${unet_type_values[@]}"; do
    for LEADTIME in "${leadtime_values[@]}"; do
        # Determine TIME_LIMIT based on LEADTIME
        case $LEADTIME in
            02|03|04)
                TIME_LIMIT="24:00:00"
                NODES=16
                ;;
            05|06)
                TIME_LIMIT="08:00:00"
                NODES=16
                ;;
            07|08|09|10)
                TIME_LIMIT="00:10:00"
                NODES=32
                ;;
            *)
                echo "Invalid LEADTIME: ${LEADTIME}. Skipping..."
                continue
                ;;
        esac

        # Create a job script with the current parameters
        JOB_SCRIPT="DL_HPT_leadtime_${LEADTIME}_${UNET_TYPE}.sh"
        
        cat <<EOT > $JOB_SCRIPT
#!/bin/bash

#SBATCH --job-name=DL_HPT_leadtime_${LEADTIME}_${UNET_TYPE}_%j
#SBATCH --output=LOGS/DL_HPT_${LEADTIME}_${UNET_TYPE}.out
#SBATCH --error=LOGS/DL_HPT_${LEADTIME}_${UNET_TYPE}.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4

# Reload environment settings
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTSEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh

# Iterate over dropout, learning rate, and batch size
for lr_value in 0.00001 0.0001 0.001 0.01; do
    for bs_value in 2 4 8 16; do
        dropout_value=0
        echo "Running DL_TRAIN.py for day${LEADTIME} with dropout=\${dropout_value}, lr=\${lr_value}, bs=\${bs_value}, unet_type=${UNET_TYPE} ..."
        srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=4 \
            python DL_TRAIN.py --lr \${lr_value} --bs \${bs_value} --lr_factor ${LR_FACTOR} \
            --filters ${FILTERS} --mask_type ${MASK_TYPE} --HPT_path ${HPT_PATH} \
            --leadtime day${LEADTIME} --dropout \${dropout_value} --unet_type ${UNET_TYPE} &
        sleep 0.5
    done
done

# Wait for all background jobs to finish
wait
EOT
        # Submit the job script
        sleep 0.5
        sbatch $JOB_SCRIPT
        rm $JOB_SCRIPT
    done
done
