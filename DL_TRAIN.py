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
args = parser.parse_args()

leadtime=args.leadtime 
HPT_path=args.HPT_path #HPT_v1 or #HPT_v2
mask_type = args.mask_type
LR=args.lr
BS=args.bs
lr_factor=args.lr_factor
Filters=args.filters

# Define the data specifications:
model_data = ["ADAPTER_DE05."+ leadtime + ".merged.nc"]
reference_data = ["ADAPTER_DE05.day01.merged.nc"]
task_name = "spatiotemporal"
mm = "MM"  # or DM
date_start = "2018-01-01T13"
date_end = "2022-12-31T23" # one year for testing
variable = "pr"
laginensemble = False
min_delta_or_lr=0.0000001 #just to avoid any limitations

# Define the following for network configs:
loss = "mse"
min_LR = min_delta_or_lr
lr_patience = 4
patience = 10
epochs = 128
val_split = 0.50
n_channels = 7
xpixels = 128
ypixels = 256

filename = func_train.data_unique_name_generator(model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble)
data_unique_name = filename[:-4]
print(data_unique_name)

# load the training data
print("Loading training data...")
train_files = np.load(TRAIN_FILES + "/" + filename)

train_x = train_files["train_x"]
train_y = train_files["train_y"]
#train_m = train_files["train_m"]
val_x = train_files["val_x"]
val_y = train_files["val_y"]
#val_m = train_files["val_m"]
print("Data loaded!")

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BS)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(BS)

train_x, train_y, val_x, val_y = None, None, None, None

training_unique_name = func_train.generate_training_unique_name(loss, Filters, LR, min_LR, lr_factor, lr_patience, BS, patience, val_split, epochs)
print(training_unique_name)

model = func_train.UNET_ATT(xpixels, ypixels, n_channels, Filters)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR, name='Adam')
model.compile(optimizer=optimizer, loss=loss)

# Define the model checkpoint and early stopping callbacks
model_path = PPROJECT_DIR2 + HPT_path + training_unique_name + '_' + leadtime + '.h5'
checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss')
callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss')]

# Define the ReduceLROnPlateau callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, min_lr=min_LR, min_delta=min_delta_or_lr)

print("Training the model...")

# Train the model using train_dataset and val_dataset
results = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=2, callbacks=[callbacks, checkpointer, reduce_lr])

# Save and plot the results
print("Saving and plotting the results...")
RESULTS_DF = pd.DataFrame(results.history)
RESULTS_DF.to_csv(PPROJECT_DIR2 + HPT_path + training_unique_name + "_" + leadtime + ".csv")

