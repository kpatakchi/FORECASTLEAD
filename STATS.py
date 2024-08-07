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

# Align MOD_D and CMOD_D with REF_D
MOD, CMOD, REF = xr.align(MOD, CMOD, REF, join='inner')

# Apply mask to ensure positive values
MOD = MOD.where(MOD >= 0, drop=True)
CMOD = CMOD.where(CMOD >= 0, drop=True)
REF = REF.where(REF>= 0, drop=True)

# Resample both datasets to daily frequency
REF_D = func_stats.resample_dataset(REF, "daily")
MOD_D = func_stats.resample_dataset(MOD, "daily")
CMOD_D = func_stats.resample_dataset(CMOD, "daily")

# Apply mask to ignore very small values
MOD_D = MOD_D.where(MOD_D >= 0.1, drop=True)
CMOD_D = CMOD_D.where(CMOD_D >= 0.1, drop=True)
REF_D = REF_D.where(REF_D>= 0.1, drop=True)

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

    
    

