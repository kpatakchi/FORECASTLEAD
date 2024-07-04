from py_env_train import *
import argparse
import tensorflow as tf
import os
import pandas as pd
import numpy as np

# Define the path to the CSV files
path_to_csv = os.path.join(PPROJECT_DIR2, "HPT")
all_files = [file for file in os.listdir(path_to_csv) if file.endswith('.csv')]
days = [f"day{str(i).zfill(2)}.csv" for i in range(2, 11)]

# Function to extract minimum val_loss for a given day
def extract_min_val_loss(day):
    # Filter out the files ending with the specific day
    filtered_files = [file for file in all_files if file.endswith(day)]
    
    # Initialize lists to store hyperparameters and their corresponding min val_loss
    dropouts_list = []
    lrs_list = []
    bss_list = []
    val_losses_list = []

    # Loop over each file and extract the minimum val_loss
    for file in filtered_files:
        file_path = os.path.join(path_to_csv, file)
        
        # Extract hyperparameters from the file name
        parts = file.split('_')
        dropout = float(parts[10])
        lr = float(parts[2])
        bs = int(parts[6])
                
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Find the minimum validation loss
        min_val_loss = df["val_loss"].min()
        
        # Append the hyperparameters and min_val_loss to the lists
        dropouts_list.append(dropout)
        lrs_list.append(lr)
        bss_list.append(bs)
        val_losses_list.append(min_val_loss)
        
    return np.array(dropouts_list), np.array(lrs_list), np.array(bss_list), np.array(val_losses_list)

# Dictionary to store the best hyperparameters for each lead day
best_hyperparameters = {}

# Loop through each day to find the best hyperparameters
for day in days:
    dropouts, lrs, bss, val_losses = extract_min_val_loss(day)
    
    # Find the index of the minimum val_loss
    min_idx = np.argmin(val_losses)
    
    # Store the best hyperparameters for this lead day
    best_hyperparameters[day] = {
        'dropout': dropouts[min_idx],
        'lr': lrs[min_idx],
        'bs': bss[min_idx],
        'val_loss': val_losses[min_idx]
    }

# Save the best hyperparameters to a file
best_params_df = pd.DataFrame.from_dict(best_hyperparameters, orient='index')
best_params_df.to_csv('best_hyperparameters.csv')
best_params_df
