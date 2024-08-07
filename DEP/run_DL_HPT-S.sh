#!/bin/sh

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh

# remove existing directories:
rm -r $HPT_DIR/*

for leadtime in {02..10}; do
    job_name="DL_HPT_DAY${leadtime}"
    output_file="LOGS/DL_HPT_DAY${leadtime}.out"
    error_file="LOGS/DL_HPT_DAY${leadtime}.err"

    sbatch <<EOL
    
#!/bin/sh

#SBATCH --job-name=$job_name
#SBATCH --output=$output_file
#SBATCH --error=$error_file
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4

source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv-train
source /p/project1/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/DL_settings.sh

# Run DL_TRAIN.py for each combination of hyperparameters and lead time in parallel
for dropout in 0 0.1 0.2 0.3; do
    for lr in 0.01 0.001 0.0001 0.00001; do
        for bs in 2 4 8 16; do
            for unet_type in unet-l unet-m unet-s unet-xs; do
                echo "Running DL_TRAIN.py for day$leadtime with dropout=\$dropout, lr=\$lr, bs=\$bs, unet_type=\$unet_type ..."
                srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=4 python DL_TRAIN.py --lr \$lr --bs \$bs --lr_factor \$LR_FACTOR --filters \$FILTERS --mask_type \$MASK_TYPE --HPT_path \${HPT_PATH} --leadtime day$leadtime --dropout \$dropout --unet_type \$unet_type &
            done
        done
    done
done
EOL

srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=4 python EXTRACT_HPT.py # to extract the best hyperparameters in a csv file.

# Wait for all background jobs to finish
wait

