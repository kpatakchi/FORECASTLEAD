from py_env_hpc import *

# Load the reference dataset
REF = xr.open_dataset(HRES_PREP+"/ADAPTER_DE05.day01.merged")

# Loop through the model data files from day02 to day10
for day in range(2, 11):
    MOD = os.path.join(HRES_PREP, f"ADAPTER_DE05.day{day:02d}.merged.nc")
    CMOD = os.path.join(PREDICT_FILES, f"ADAPTER_DE05.day{day:02d}.merged.nc")

    # Load the model dataset
    MOD = xr.open_dataset(MOD)
    CMOD = xr.open_dataset(CMOD)

    # Resample both datasets to daily frequency
    REF_D = func_stats.resample_dataset(REF, "daily")
    MOD_D = func_stats.resample_dataset(MOD, "daily")
    CMOD_D = func_stats.resample_dataset(CMOD, "daily")

    # Calculate metrics
    MOD_METRICS = func_stats.calculate_metrics(REF_D, MOD_D)
    CMOD_METRICS = func_stats.calculate_metrics(REF_D, CMOD_D)

    # Save the metrics to the STATS folder
    MOD_OUT = os.path.join(STATS, f"ADAPTER_DE05.day{day:02d}_HRES_stats.nc")
    CMOD_OUT = os.path.join(STATS, f"ADAPTER_DE05.day{day:02d}_HRES_C_stats.nc")

    xr.Dataset(MOD_METRICS).to_netcdf(MOD_OUT)
    xr.Dataset(CMOD_METRICS).to_netcdf(CMOD_OUT)

    # Close the datasets
    MOD.close()
    CMOD.close()
