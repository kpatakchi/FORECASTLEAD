#!/bin/bash

# DL_settings.sh
# Parameters for DL scripts
# Default values (will be overwritten if best_hyperparameters.csv is found and lead_day is provided)
LR=0.00001
BS=16
LR_FACTOR=0.5
FILTERS=64
MASK_TYPE="no_na_land"
HPT_PATH="HPT/"
DROPOUT=0.3

# Path to the CSV file containing best hyperparameters
HYPERPARAM_CSV="/p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/best_hyperparameters.csv"

# Function to read CSV and extract the best hyperparameters for the given lead day
function set_hyperparameters() {
    local lead_day=$1
    
    if [ -f "$HYPERPARAM_CSV" ]; then
        # Create a temporary file to hold the CSV data without the header
        temp_csv=$(mktemp)
        tail -n +2 "$HYPERPARAM_CSV" > "$temp_csv"
        
        while IFS=, read -r day dropout lr bs val_loss; do
            if [ "$day" == "$lead_day" ]; then
                DROPOUT=$dropout
                LR=$lr
                BS=$bs
                break
            fi
        done < "$temp_csv"
        
        # Remove the temporary file
        rm -f "$temp_csv"
    else
        echo "Warning: $HYPERPARAM_CSV not found. Using default hyperparameters."
    fi
}

# Check if a lead day argument is provided
if [ "$#" -eq 1 ]; then
    # Set hyperparameters for the specified lead day
    set_hyperparameters "$1"
    echo "Using hyperparameters for lead day $1 - LR: $LR, BS: $BS, DROPOUT: $DROPOUT"
else
    # Using default hyperparameters
    echo "No lead day provided. Using default hyperparameters - LR: $LR, BS: $BS, DROPOUT: $DROPOUT"
fi

# Export variables for use in other scripts or commands
export LR
export BS
export DROPOUT
