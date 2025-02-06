from py_env_hpc import *
from PIL import Image, ImageDraw

# Function to extract minimum val_loss for a given day
def extract_min_val_loss(day, all_files, path_to_csv):
    # Filter out the files ending with the specific day
    filtered_files = [file for file in all_files if file.endswith(day)]
    
    # Initialize lists to store hyperparameters and their corresponding min val_loss
    dropouts_list = []
    lrs_list = []
    bss_list = []
    val_losses_list = []

    # Loop over each file and extract the minimum val_loss
    for file in filtered_files:
        file_path = os.path.join(path_to_csv, file)
        
        try:
            # Extract hyperparameters from the file name
            parts = file.split('_')
            dropout = float(parts[10])
            lr = float(parts[2])
            bs = int(parts[6])

            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Find the minimum validation loss
            min_val_loss = df["val_loss"].min()
            
            # Append the hyperparameters and min_val_loss to the lists
            dropouts_list.append(dropout)
            lrs_list.append(lr)
            bss_list.append(bs)
            val_losses_list.append(min_val_loss)

        except (IndexError, ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Error processing file {file}: {e}")
            continue
        
    return np.array(dropouts_list), np.array(lrs_list), np.array(bss_list), np.array(val_losses_list)


def plot_hpt_3dscatter_data(ax, day, dropouts, lrs, bss, val_losses, cmap, fs, stepsincolor):

    max_lr=0.001
    min_lr=0.00001
    min_bs=2
    max_bs=8
    min_dropout=0
    max_dropout=0.3
    dropoutstep=0.1
    
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

    ax.set_xticks(np.arange(min_dropout, max_dropout+dropoutstep, dropoutstep))
    ax.set_xticks(np.log10([min_lr, 0.0001, 0.001, max_lr]))
    ax.set_xticklabels(['0', '0.1', '0.2', '0.3'], fontsize=10*fs)
    ax.set_yticklabels(['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$'], fontsize=10*fs)
    ax.set_zticks(np.log2([min_bs, 2, 4, max_bs]))
    ax.set_zticklabels(['$2^{0}$', '$2^{1}$', '$2^{2}$', '$2^{3}$'], fontsize=10*fs)
    
    ax.set_xlim([0, max_dropout])
    ax.set_ylim(np.log10(max_lr), np.log10(min_lr))
    ax.set_zlim(np.log2([min_bs, max_bs]))
    
    ax.grid(True, which='both')
    
    ax.plot([dropouts[min_idx], dropouts[min_idx]], [np.log10(lrs[min_idx]), np.log10(lrs[min_idx])], [np.log2(min_bs), np.log2(bss[min_idx])], linestyle='--', color='gray', linewidth=0.5)
    ax.plot([dropouts[min_idx], dropouts[min_idx]], [np.log10(min_lr), np.log10(lrs[min_idx])], [np.log2(bss[min_idx]), np.log2(bss[min_idx])], linestyle='--', color='gray', linewidth=0.5)
    ax.plot([min_dropout, dropouts[min_idx]], [np.log10(lrs[min_idx]), np.log10(lrs[min_idx])], [np.log2(bss[min_idx]), np.log2(bss[min_idx])], linestyle='--', color='gray', linewidth=0.5)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=local_vmin, vmax=local_vmax))
    sm.set_array(transformed_val_losses)
    
    cb = plt.colorbar(sm, ax=ax, pad=0.12, aspect=40, shrink=0.5, location="bottom")
    cb.set_label('$\\log_{10}(\mathrm{val\_loss})$', fontsize=10*fs)
    cb.ax.tick_params(labelsize=9*fs)

    tick_values = np.linspace(local_vmin, local_vmax, int(stepsincolor/2))
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{tick:.1f}" for tick in tick_values])

def plot_hpt_2dscatter_data(ax, day, lrs, bss, val_losses, cmap, fs, stepsincolor, aspect_ratio, show_ylabel):

    max_lr = 0.001
    min_lr = 0.00001
    min_bs = 2
    max_bs = 8
    
    # Log-transform the data
    transformed_val_losses = np.log10(val_losses)
    
    # Determine the range for the color mapping
    local_vmax = transformed_val_losses.max()
    local_vmin = transformed_val_losses.min()
    
    # Create the scatter plot with color-coded points based on the validation loss
    scatter = ax.scatter(np.log10(lrs), np.log2(bss), c=transformed_val_losses, cmap=cmap, 
                         vmin=local_vmin, vmax=local_vmax, s=50, edgecolors='k', linewidths=0.5)
    
    # Highlight the minimum loss point with a different marker
    min_idx = np.argmin(val_losses)
    ax.scatter(np.log10(lrs[min_idx]), np.log2(bss[min_idx]), marker='x', c=transformed_val_losses[min_idx], 
               cmap=cmap, vmin=local_vmin, vmax=local_vmax, s=100)
        
    # Set axis labels and title
    ax.set_xlabel('Learning Rate', fontsize=8*fs)
    if show_ylabel:
        ax.set_ylabel('Batch Size', fontsize=8*fs)
    else:
        ax.set_ylabel('')  # Hide the y-axis label
    
    ax.set_title(f'Lead day {day[:-4][-2:]}', fontsize=12*fs)

    # Configure ticks and labels
    ax.set_xticks(np.log10([min_lr, 0.0001, max_lr]))
    ax.set_xticklabels(['$10^{-5}$', '$10^{-4}$', '$10^{-3}$'], fontsize=10*fs)
    ax.set_yticks(np.log2([min_bs, 4, max_bs]))
    ax.set_yticklabels(['$2^{1}$', '$2^{2}$', '$2^{3}$'], fontsize=10*fs)
    
    # Set limits for the axes
    #ax.set_xlim(np.log10(min_lr)*0.99, np.log10(max_lr)*1.01)
    #ax.set_ylim(np.log2([min_bs*0.9, max_bs])*1.1)
    
    # Add grid
    ax.grid(True, which='both')
    ax.set_aspect(aspect_ratio, adjustable='box')
    
    # Add colorbar to show the range of validation losses
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=local_vmin, vmax=local_vmax))
    sm.set_array(transformed_val_losses)
    
    cb = plt.colorbar(sm, ax=ax, pad=0.18, aspect=40, shrink=0.5, location="bottom")
    cb.set_label('$\\log_{10}(\mathrm{val\_loss})$', fontsize=10*fs)
    cb.ax.tick_params(labelsize=9*fs)

    tick_values = np.linspace(local_vmin, local_vmax, int(stepsincolor/2))
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{tick:.1f}" for tick in tick_values])

def add_border(image, border_color=(255, 0, 0), border_width=5):
    """Add a border around an image."""
    # Get the size of the image
    width, height = image.size
    
    # Create a new image with border
    new_image = Image.new('RGB', (width + 2 * border_width, height + 2 * border_width), border_color)
    
    # Paste the original image onto the new image
    new_image.paste(image, (border_width, border_width))
    
    return new_image

def merge_images(image_files, output_filename, border_color=(255, 255, 255), border_width=1):
    # Open and add borders to the images
    images = [add_border(Image.open(img), border_color, border_width) for img in image_files]
    
    # Determine the size of the final image
    img_width, img_height = images[0].size
    num_images = len(images)
    
    # Determine the layout for the final image
    columns = 3
    rows = (num_images + columns - 1) // columns
    
    # Calculate the size of the final image
    total_width = columns * img_width
    total_height = rows * img_height
    
    # Create a new image with the calculated size
    merged_image = Image.new('RGB', (total_width, total_height))
    
    # Paste each image into the new image
    for index, image in enumerate(images):
        x = (index % columns) * img_width
        y = (index // columns) * img_height
        merged_image.paste(image, (x, y))
    
    # Save the final image
    merged_image.save(output_filename)
    print(f"Saved merged image as {output_filename}")