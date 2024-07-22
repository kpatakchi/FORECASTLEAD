from py_env_hpc import *

# Function to extract minimum val_loss for a given day
def extract_min_val_loss(day, all_files, path_to_csv):
    # Filter out the files ending with the specific day
    filtered_files = [file for file in all_files if file.endswith(day)]
    
    # Initialize lists to store hyperparameters and their corresponding min val_loss
    dropouts_list = []
    lrs_list = []
    bss_list = []
    val_losses_list = []
    unet_types_list = []

    # Loop over each file and extract the minimum val_loss
    for file in filtered_files:
        file_path = os.path.join(path_to_csv, file)
        
        try:
            # Extract hyperparameters from the file name
            parts = file.split('_')
            dropout = float(parts[10])
            lr = float(parts[2])
            bs = int(parts[6])
            unet_type = str(parts[11])

            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Find the minimum validation loss
            min_val_loss = df["val_loss"].min()
            
            # Append the hyperparameters and min_val_loss to the lists
            dropouts_list.append(dropout)
            lrs_list.append(lr)
            bss_list.append(bs)
            val_losses_list.append(min_val_loss)
            unet_types_list.append(unet_type)

        except (IndexError, ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Error processing file {file}: {e}")
            continue
        
    return np.array(dropouts_list), np.array(lrs_list), np.array(bss_list), np.array(val_losses_list), np.array(unet_types_list)


def plot_hpt_scatter_data(ax, day, dropouts, lrs, bss, val_losses, cmap, fs, stepsincolor):
    transformed_val_losses = np.log10(val_losses)
    
    local_vmax = transformed_val_losses.max()
    local_vmin = transformed_val_losses.min()
    
    scatter = ax.scatter(xs=dropouts, ys=np.log10(lrs), zs=np.log2(bss), c=transformed_val_losses, cmap=cmap, vmin=local_vmin, vmax=local_vmax, s=20, edgecolors='k', linewidths=0.5)
    
    min_idx = np.argmin(val_losses)
    ax.scatter(xs=dropouts[min_idx], ys=np.log10(lrs[min_idx]), zs=np.log2(bss[min_idx]), marker='x', c=transformed_val_losses[min_idx], cmap=cmap, vmin=local_vmin, vmax=local_vmax, s=100)
        
    ax.set_xlabel('Dropout', fontsize=8*fs)
    ax.set_ylabel('Learning Rate', fontsize=8*fs)
    ax.set_zlabel('Batch Size', fontsize=8*fs)
    ax.set_title(f'Lead day {day[:-4][-2:]}', fontsize=12*fs)

    ax.set_xticks(np.arange(0.1, 0.8, 0.2))
    ax.set_yticks(np.log10([0.01, 0.001, 0.0001, 0.00001]))
    ax.set_xticklabels(['0.1', '0.3', '0.5', '0.7'], fontsize=10*fs)
    ax.set_yticklabels(['$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$'], fontsize=10*fs)
    ax.set_zticks(np.log2([4, 8, 16, 32]))
    ax.set_zticklabels(['$2^{2}$', '$2^{3}$', '$2^{4}$', '$2^{5}$'], fontsize=10*fs)
    
    ax.set_xlim([0.1, 0.7])
    ax.set_ylim(np.log10(0.01), np.log10(0.00001))
    ax.set_zlim(np.log2([4, 32]))
    
    ax.grid(True, which='both')
    
    ax.plot([dropouts[min_idx], dropouts[min_idx]], [np.log10(lrs[min_idx]), np.log10(lrs[min_idx])], [np.log2(4), np.log2(bss[min_idx])], linestyle='--', color='gray', linewidth=0.5)
    ax.plot([dropouts[min_idx], dropouts[min_idx]], [np.log10(0.00001), np.log10(lrs[min_idx])], [np.log2(bss[min_idx]), np.log2(bss[min_idx])], linestyle='--', color='gray', linewidth=0.5)
    ax.plot([0.1, dropouts[min_idx]], [np.log10(lrs[min_idx]), np.log10(lrs[min_idx])], [np.log2(bss[min_idx]), np.log2(bss[min_idx])], linestyle='--', color='gray', linewidth=0.5)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=local_vmin, vmax=local_vmax))
    sm.set_array(transformed_val_losses)
    
    cb = plt.colorbar(sm, ax=ax, pad=0.12, aspect=40, shrink=0.5, location="bottom")
    cb.set_label('$\\log_{10}(\mathrm{val\_loss})$', fontsize=10*fs)
    cb.ax.tick_params(labelsize=9*fs)

    tick_values = np.linspace(local_vmin, local_vmax, int(stepsincolor/2))
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{tick:.1f}" for tick in tick_values])