#!/bin/sh

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh

#rm -r $HPT_DIR/*

# Define arrays for parameter values
#unet_type_values=("unet-xs" "unet-s" "unet-m" "unet-l" "unet-trans-s" "unet-trans-l" "unet-att-s" "unet-att-l" "unet-se")
#leadtime_values=("02" "03" "04" "05" "06" "07" "08" "09" "10") 
unet_type_values=("unet-att-l")
leadtime_values=("03") 

# Define weight dictionaries
declare -A unet_weights=( ["unet-xs"]=4 ["unet-s"]=5 ["unet-m"]=6 ["unet-l"]=7 ["unet-se"]=7 ["unet-trans-s"]=5 ["unet-trans-l"]=6 ["unet-att-s"]=5 ["unet-att-l"]=6)
declare -A leadtime_weights=( ["02"]=10 ["03"]=9 ["04"]=7 ["05"]=5 ["06"]=4 ["07"]=3 ["08"]=2 ["09"]=1 ["10"]=1 )
#declare -A node_weights=( [16]=2 [14]=3 [10]=4 [8]=5 [6]=9 [4]=12 [2]=24) for 16 combo
declare -A node_weights=([9]=2 [7]=3 [5]=4) # for 9 combo

# Define a function to calculate the total weight and determine the time limit and nodes
calculate_weight_product() {
    UNET_TYPE=$1
    LEADTIME=$2
    NODES=$3
    
    unet_weight=${unet_weights[$UNET_TYPE]}
    leadtime_weight=${leadtime_weights[$LEADTIME]}
    node_weight=${node_weights[$NODES]}

    weight_product=$((unet_weight * leadtime_weight * node_weight))

    echo "$weight_product"
}

#Function to convert weight product into HH:MM:SS format
calculate_time_limit() {
    weight_product=$1

    total_seconds=$((weight_product * 960))

    # Cap total_seconds at 86400 (24 hours)
    if [[ $total_seconds -gt 86400 ]]; then
        total_seconds=86400
    fi

    # Convert total_seconds to HH:MM:SS format
    hours=$((total_seconds / 3600))
    minutes=$(((total_seconds % 3600) / 60))
    seconds=$((total_seconds % 60))

    # Format the time as HH:MM:SS
    TIME_LIMIT=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

    echo "$TIME_LIMIT"
}

for UNET_TYPE in "${unet_type_values[@]}"; do
    for LEADTIME in "${leadtime_values[@]}"; do

        # Determine number of nodes based on lead time
        if [[ "$LEADTIME" == "02" || "$LEADTIME" == "03" ]]; then
            NODES=9
        elif [[ "$LEADTIME" == "04" || "$LEADTIME" == "05" ]]; then
            NODES=7
        elif [[ "$LEADTIME" == "06" || "$LEADTIME" == "07" || "$LEADTIME" == "08" ]]; then
            NODES=5
        else
            NODES=5
        fi

        weight_product=$(calculate_weight_product $UNET_TYPE $LEADTIME $NODES)
        echo "Weight product for $UNET_TYPE, leadtime $LEADTIME, nodes $NODES: $weight_product"

        # Calculate TIME_LIMIT based on weight_product        
        TIME_LIMIT=$(calculate_time_limit $weight_product)
        echo $TIME_LIMIT
        
        # Create a job script with the current parameters
        JOB_SCRIPT="DL_HPT_leadtime_${LEADTIME}_${UNET_TYPE}.sh"
        
        cat <<EOT > $JOB_SCRIPT
#!/bin/bash

#SBATCH --job-name=leadtime_${LEADTIME}_${UNET_TYPE}_%j
#SBATCH --output=LOGS/DL_${LEADTIME}_${UNET_TYPE}.out
#SBATCH --error=LOGS/DL_${LEADTIME}_${UNET_TYPE}.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

# Reload environment settings
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh

# Capture start time
start_time=$(date +%s)

# Iterate over dropout, learning rate, and batch size
for lr_value in 0.00001 0.0001 0.001; do
    for bs_value in 2 4 8; do
        dropout_value=0
        echo "Running DL_TRAIN.py for day${LEADTIME} with dropout=\${dropout_value}, lr=\${lr_value}, bs=\${bs_value}, unet_type=${UNET_TYPE} ..."
        srun --nodes=1 --ntasks=1 --gres=gpu:4 \
            python DL_TRAIN.py --lr \${lr_value} --bs \${bs_value} --lr_factor ${LR_FACTOR} \
            --filters ${FILTERS} --mask_type ${MASK_TYPE} --HPT_path ${HPT_PATH} \
            --leadtime day${LEADTIME} --dropout \${dropout_value} --unet_type ${UNET_TYPE} &
        sleep 2
    done
done

# Wait for all background jobs to finish
wait

# Capture end time
end_time=$(date +%s)

# Calculate total runtime in seconds
runtime=$((end_time - start_time))

# Convert runtime to hours, minutes, and seconds
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

# Print the calculated time limit
echo "Time limit for ${UNET_TYPE}, leadtime ${LEADTIME}: ${TIME_LIMIT}"

# Print runtime in a human-readable format
echo "Run time for ${UNET_TYPE} with leadtime ${LEADTIME}: ${runtime} seconds"
EOT
        # Submit the job script
        sleep 0.1
        sbatch $JOB_SCRIPT
        rm $JOB_SCRIPT
    done
done
