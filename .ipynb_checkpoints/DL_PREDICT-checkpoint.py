from py_env_train import *
import argparse

# define the parameters
parser = argparse.ArgumentParser(description="Training hyperparameters")
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
parser.add_argument("--bs", type=int, required=True, help="Batch size")
parser.add_argument("--lr_factor", type=float, required=True, help="Learning rate factor")
parser.add_argument("--filters", type=int, required=True, help="Number of filters")
parser.add_argument("--mask_type", type=str, required=True, help="Mask Type")
parser.add_argument("--HPT_path", type=str, required=True, help="Which HPT path for results?")
parser.add_argument("--leadtime", type=str, required=True, help="Specify the lead time for correction (e.g., day02, day03 etc")
parser.add_argument("--dropout", type=float, required=True, help="specify the dropout rate in U-Net")
parser.add_argument("--unet_type", type=str, required=True, help="specify the type of u-net")

args = parser.parse_args()

leadtime=args.leadtime 
HPT_path=args.HPT_path #HPT_v1 or #HPT_v2
mask_type = args.mask_type
LR=args.lr
BS=args.bs
lr_factor=args.lr_factor
Filters=args.filters
dropout = args.dropout
unet_type = args.unet_type

print(leadtime)
print(LR)
print(BS)
print(unet_type)

# Define the data specifications:
model_data = ["ADAPTER_DE05."+ leadtime + ".merged.nc"]
reference_data = ["ADAPTER_DE05.day01.merged.nc"]
task_name = "model_only"
mm = "MM"  # or DM
date_start = "2018-01-01T13" # first day is not corrected
date_end = "2022-12-31T23"
date_end2 = "2023-12-31T23"
variable = "pr"
laginensemble = False
min_delta_or_lr=0.00000000000000001 #just to avoid any limitations

# Define the following for network configs:
loss_n = "mse"
min_LR = min_delta_or_lr
lr_patience = 4
patience = 16
epochs = 80
val_split = 0.50
n_channels = 1
xpixels = 128
ypixels = 256

if loss_n == "mse-mae":
    def mse_mae_loss(y_true, y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        return mse + mae

    loss = mse_mae_loss
else:
    loss = loss_n
    
filename = func_train.data_unique_name_generator(model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble)
data_unique_name = filename[:-4]
print(data_unique_name)

# load the production data
print("Loading production data...")
produce_files = np.load(PRODUCE_FILES + "/" + "produce_for_" + filename)
train_x = produce_files["canvas_x"]

# Convert numpy arrays to TensorFlow tensors
train_x = tf.data.Dataset.from_tensor_slices(train_x).batch(BS)

training_unique_name = func_train.generate_training_unique_name(loss_n, Filters, LR, min_LR, lr_factor, lr_patience, BS, patience, val_split, epochs, str(dropout), unet_type, leadtime)

# load the model and weights
model = func_train.UNET(xpixels, ypixels, n_channels, Filters, dropout, unet_type)
model_path = PPROJECT_DIR2 + HPT_path + "/" + training_unique_name + '.h5'
model.load_weights(model_path)

# produce 
print("Predicting the mismatches ...")
Y_PRED = model.predict(train_x, verbose=0)
Y_PRED=Y_PRED[..., 0]

#output_npz_path = f"{PREDICT_FILES}/Y_PRED_{training_unique_name}.npz"
#np.savez(output_npz_path, Y_PRED=Y_PRED)

train_x=None

import csv

scaling_file = f"{PPROJECT_DIR2}/CODES-MS3/FORECASTLEAD/scaling_info.csv"

y_min, y_max, x_min, x_max = func_train.get_scaling_params(scaling_file, PPROJECT_DIR2, leadtime)

# Save in PREDICT_FILES
func_train.de_prepare_produce(Y_PRED, PREDICT_FILES + "/", HRES_PREP, filename, 
                              model_data[0], date_start, date_end2, variable, 
                              training_unique_name, reference_data[0], leadtime, y_min, y_max)