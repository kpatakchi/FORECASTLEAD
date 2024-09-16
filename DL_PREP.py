from py_env_train import *
import argparse

# define the parameters
parser = argparse.ArgumentParser(description="Prepare data for DL")
parser.add_argument("--leadtime", type=str, required=True, help="Specify the lead time for correction (e.g., day02, day03 etc")
parser.add_argument("--mask_type", type=str, required=True, help="Specify mask type")
args = parser.parse_args()

# Define the data specifications:
leadtime=args.leadtime 
mask_type=args.mask_type 

model_data = ["ADAPTER_DE05."+ leadtime + ".merged.nc"]
reference_data = ["ADAPTER_DE05.day01.merged.nc"]
task_name = "model_only"
mm = "MM"  # or DM
date_start = "2018-01-01T13"
date_end = "2022-12-31T23"
date_end2 = "2023-12-31T23"

variable = "pr"
laginensemble = False

# Define the following for network configs:
val_split = 0.5
n_channels = 1
xpixels = 128
ypixels = 256

filename = func_train.data_unique_name_generator(model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble)
data_unique_name = filename[:-4]
print(data_unique_name)

# Create the training data (if doesn't exist)
data_avail = func_train.prepare_train(PPROJECT_DIR, TRAIN_FILES, HRES_PREP, filename, 
                         model_data, reference_data, task_name, mm, date_start,
                         date_end, variable, mask_type, laginensemble, val_split, leadtime)

data_avail = None

scaling_file = f"{PPROJECT_DIR2}/CODES-MS3/FORECASTLEAD/scaling_info.csv"

y_mean, y_std, x_mean, x_std = func_train.get_scaling_params(scaling_file, PPROJECT_DIR2, leadtime)

# Create the production data (if doesn't exist)
data_avail = func_train.prepare_produce(PPROJECT_DIR, PRODUCE_FILES, HRES_PREP, filename,
                     model_data, reference_data, task_name, mm, date_start,
                       date_end2, variable, mask_type, laginensemble, leadtime, y_mean, y_std, x_mean, x_std)
