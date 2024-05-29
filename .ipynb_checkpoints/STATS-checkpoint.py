from py_env_train import *
import argparse

#### Part1) Calculate three error metrics for HRES and HRES_C in a daily scale

# define the parameters
parser = argparse.ArgumentParser(description="Add arguments")
parser.add_argument("--leadtime", type=float, required=True, help="Lead time day")
args = parser.parse_args()

# Load arguments
day = int(args.leadtime)

# Load the reference dataset
REF = xr.open_dataset(HRES_PREP+"/ADAPTER_DE05.day01.merged.nc")

print(f"Processing day {day:02d}...")
MOD = os.path.join(HRES_PREP, f"ADAPTER_DE05.day{day:02d}.merged.nc")
CMOD = os.path.join(PREDICT_FILES, f"ADAPTER_DE05.day{day:02d}.merged.nc.corrected.nc")

# Load the model dataset
MOD = xr.open_dataset(MOD)
CMOD = xr.open_dataset(CMOD)

# Resample both datasets to daily frequency
REF_D = func_stats.resample_dataset(REF, "daily")
MOD_D = func_stats.resample_dataset(MOD, "daily")
CMOD_D = func_stats.resample_dataset(CMOD, "daily")
print("Datasets resampled.")

# Calculate metrics
MOD_METRICS = func_stats.calculate_metrics(REF_D.pr, MOD_D.pr)
CMOD_METRICS = func_stats.calculate_metrics(REF_D.pr, CMOD_D.pr)
print("Metrics calculated. saving ...")

# Save the metrics to the STATS folder
MOD_OUT = os.path.join(STATS, f"ADAPTER_DE05.day{day:02d}_HRES_stats.nc")
CMOD_OUT = os.path.join(STATS, f"ADAPTER_DE05.day{day:02d}_HRES_C_stats.nc")

xr.Dataset(MOD_METRICS).to_netcdf(MOD_OUT)
xr.Dataset(CMOD_METRICS).to_netcdf(CMOD_OUT)

# Close the datasets
MOD.close()
CMOD.close()

    
    

