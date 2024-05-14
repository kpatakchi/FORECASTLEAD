from py_env_train import *
import argparse

# define the parameters
parser = argparse.ArgumentParser(description="Prepare data for DL")
parser.add_argument("--leadtime", type=str, required=True, help="Specify the lead time for correction (e.g., day02, day03 etc")
args = parser.parse_args()

# Define the data specifications:
leadtime=args.leadtime 
model_data = ["ADAPTER_DE05."+ leadtime + ".merged.nc"]
reference_data = ["ADAPTER_DE05.day01.merged.nc"]
task_name = "spatiotemporal"
mm = "MM"  # or DM
date_start = "2018-01-01T13"
date_end = "2023-12-31T23"
variable = "pr"
mask_type = "no_na"
laginensemble = False

# Define the following for network configs:
val_split = 0.25
n_channels = 7
xpixels = 128
ypixels = 256

filename = func_train.data_unique_name_generator(model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble)
data_unique_name = filename[:-4]
print(data_unique_name)

# Create the training data (if doesn't exist)
data_avail = func_train.prepare_train(PPROJECT_DIR, TRAIN_FILES, HRES_PREP, filename, 
                         model_data, reference_data, task_name, mm, date_start,
                         date_end, variable, mask_type, laginensemble, val_split)

data_avail = None

data_avail = func_train.prepare_produce(PPROJECT_DIR, PRODUCE_FILES, HRES_PREP, filename,
                     model_data, reference_data, task_name, mm, date_start,
                       date_end, variable, mask_type, laginensemble)
