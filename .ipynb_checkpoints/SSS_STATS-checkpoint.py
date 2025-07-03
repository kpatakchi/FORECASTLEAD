from py_env_train import *
import argparse

# define the parameters
parser = argparse.ArgumentParser(description="Add arguments")
parser.add_argument("--leadtime", type=float, required=True, help="Lead time day")
args = parser.parse_args()

# Load arguments
leadtime = int(args.leadtime)

REF_dir="/p/data1/jibg33/patakchiyousefi1/postpro_data/DE05_HRES-RAW/sss_REF_merged.nc"
REF = xr.open_dataset(REF_dir)
REF_SSS = REF.sss

print(f"Calculating SSS stats for {leadtime:02d}...")

HRES_SSS = os.path.join("/p/data1/jibg33/patakchiyousefi1/postpro_data/DE05_HRES-RAW/", f"sss_HRES_leadtime_{leadtime}.nc")
HRES_C_SSS = os.path.join("/p/data1/jibg33/patakchiyousefi1/postpro_data/DE05_HRES-COR/", f"sss_HRES_C_leadtime_{leadtime}.nc")

print("Opening the dataset...")
HRES_SSS = xr.open_dataset(HRES_SSS).sss
HRES_C_SSS = xr.open_dataset(HRES_C_SSS).sss

print("Calculating the metrics...")
# Calculate metrics
HRES_SSS_METRICS = func_stats.calculate_metrics(REF_SSS, HRES_SSS)
HRES_C_SSS_METRICS = func_stats.calculate_metrics(REF_SSS, HRES_C_SSS)
print("Metrics calculated. saving ...")

# Save the metrics to the STATS folder
HRES_SSS_OUT = os.path.join(STATS, f"sss.day{leadtime:02d}_HRES_stats.nc")
HRES_C_SSS_OUT = os.path.join(STATS, f"sss.day{leadtime:02d}_HRES_C_stats.nc")

xr.Dataset(HRES_SSS_METRICS).to_netcdf(HRES_SSS_OUT)
xr.Dataset(HRES_C_SSS_METRICS).to_netcdf(HRES_C_SSS_OUT)

print("Metrics saved!")
