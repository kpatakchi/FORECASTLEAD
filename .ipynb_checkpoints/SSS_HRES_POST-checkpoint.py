from py_env_train import *
import xarray as xr
import os
import glob

# ========== HRES-RAW ==========
data_dir = "/p/scratch/cesmtst/patakchiyousefi1/FORECASTLEAD_CESMTST_SCRATCH/PF/DE05_HRES-RAW/postpro/data"
output_dir = data_dir
file_list = sorted(glob.glob(os.path.join(data_dir, "sss*0012-0228.nc")))

leadtime_dict = {lead: [] for lead in range(2, 11)}

for file_path in file_list:
    print(f"Processing {file_path}")
    ds = xr.open_dataset(file_path)

    for i in range(9):  # time indices 0–8 → lead times 2–10
        leadtime = i + 2
        da = ds['sss'].isel(time=i)
        leadtime_dict[leadtime].append(da)

for leadtime, da_list in leadtime_dict.items():
    combined = xr.concat(da_list, dim='time')  # default time dim from original slices
    output_file = os.path.join(output_dir, f"sss_HRES_leadtime_{leadtime}.nc")
    combined.to_netcdf(output_file)
    print(f"Saved: {output_file}")

# ========== HRES-COR ==========
data_dir = "/p/scratch/cesmtst/patakchiyousefi1/FORECASTLEAD_CESMTST_SCRATCH/PF/DE05_HRES-COR/postpro/data"
output_dir = data_dir
file_list = sorted(glob.glob(os.path.join(data_dir, "sss*0012-0228.nc")))

leadtime_dict = {lead: [] for lead in range(2, 11)}

for file_path in file_list:
    print(f"Processing {file_path}")
    ds = xr.open_dataset(file_path)

    for i in range(9):  # time indices 0–8 → lead times 2–10
        leadtime = i + 2
        da = ds['sss'].isel(time=i)
        leadtime_dict[leadtime].append(da)

for leadtime, da_list in leadtime_dict.items():
    combined = xr.concat(da_list, dim='time')
    output_file = os.path.join(output_dir, f"sss_HRES_C_leadtime_{leadtime}.nc")
    combined.to_netcdf(output_file)
    print(f"Saved: {output_file}")
