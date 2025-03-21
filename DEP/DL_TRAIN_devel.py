from py_env_train import *
import argparse
import tensorflow as tf

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

leadtime = args.leadtime 
HPT_path = args.HPT_path  # HPT_v1 or #HPT_v2
mask_type = args.mask_type
LR = args.lr
BS = args.bs
lr_factor = args.lr_factor
Filters = args.filters
dropout = args.dropout
unet_type = args.unet_type

# Define the data specifications:
model_data = ["ADAPTER_DE05." + leadtime + ".merged.nc"]
reference_data = ["ADAPTER_DE05.day01.merged.nc"]
task_name = "model_only"
mm = "MM"  # or DM
date_start = "2018-01-01T13"
date_end = "2022-12-31T23"  # one year for testing
variable = "pr"
laginensemble = False
min_delta_or_lr = 0.00000000000000001  # just to avoid any limitations

# Define the following for network configs:
loss_n = "mse"
min_LR = min_delta_or_lr
lr_patience = 4
patience = 16
epochs = 128
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

# load the training data
print("Loading training data...")
train_files = np.load(TRAIN_FILES + "/" + filename)

train_x = train_files["train_x"]
train_y = train_files["train_y"]
val_x = train_files["val_x"]
val_y = train_files["val_y"]
train_m = train_files["train_m"]
val_m = train_files["val_m"]

print("Data loaded!")

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_m)).batch(BS)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y, val_m)).batch(BS)

# Disable auto sharding
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

train_dataset = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)

train_x, train_y, val_x, val_y = None, None, None, None

training_unique_name = func_train.generate_training_unique_name(loss_n, Filters, LR, min_LR, lr_factor, lr_patience, BS, patience, val_split, epochs, str(dropout), unet_type, leadtime)
print(training_unique_name)

if unet_type not in ["unet-trans-s", "unet-trans-l"]:
    # Distribute the training across available GPUs
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = func_train.UNET(xpixels, ypixels, n_channels, Filters, dropout, unet_type)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR, name='Adam')
        model.compile(optimizer=optimizer, loss=loss, weighted_metrics=['mse'])
        
        # Define the model checkpoint and early stopping callbacks
        model_path = PPROJECT_DIR2 + HPT_path + training_unique_name + '.h5'
        checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss')
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss')]
        
        # Define the ReduceLROnPlateau callback
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, min_lr=min_LR, min_delta=min_delta_or_lr)
        
        print("Training the model...")
        
        # Train the model using train_dataset and val_dataset
        results = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=0, callbacks=[callbacks, checkpointer, reduce_lr])
else:
    model = func_train.UNET(xpixels, ypixels, n_channels, Filters, dropout, unet_type)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, name='Adam')
    model.compile(optimizer=optimizer, loss=loss, weighted_metrics=['mse'])
    
    # Define the model checkpoint and early stopping callbacks
    model_path = PPROJECT_DIR2 + HPT_path + training_unique_name + '.h5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss')
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss')]
    
    # Define the ReduceLROnPlateau callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, min_lr=min_LR, min_delta=min_delta_or_lr)
    
    print("Training the model...")
    
    # Train the model using train_dataset and val_dataset
    results = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=0, callbacks=[callbacks, checkpointer, reduce_lr])


# Save and plot the results
print("Saving and plotting the results...")
RESULTS_DF = pd.DataFrame(results.history)
RESULTS_DF.to_csv(PPROJECT_DIR2 + HPT_path + training_unique_name + ".csv")
