from py_env_train import *

def resample_dataset(dataset, frequency):
    """
    Resample the given xarray dataset to the specified frequency.
    Parameters:
        dataset (xarray.Dataset): The dataset to be resampled.
        frequency (str): The resampling frequency, either "daily" or "monthly".
    Returns:
        xarray.Dataset: The resampled dataset.
    """
    if frequency == "daily":
        resampling_frequency = "D"
    elif frequency == "monthly":
        resampling_frequency = "M"
    else:
        raise ValueError("Invalid frequency. Please use 'daily' or 'monthly'.")

    resampled_dataset = dataset.resample(time=resampling_frequency).sum()
    return resampled_dataset


def calculate_metrics(reference, model):
    """
    Calculate temporally averaged mean error, root mean squared error, and correlation coefficient
    between corresponding variables in the reference and model datasets.
    
    Parameters:
        reference (xarray.Dataset): The reference dataset.
        model (xarray.Dataset): The model dataset.
    
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Calculate mean error
    mean_error = (model - reference).mean(dim='time')

    # Calculate root mean squared error
    squared_error = (model - reference)**2
    mse = squared_error.mean(dim='time')
    rmse = np.sqrt(mse)

    # Calculate correlation coefficient
    correlation = xr.corr(model, reference, dim='time')

    # Return the calculated metrics as a dictionary
    metrics = {
        'Mean Error': mean_error,
        'Root Mean Squared Error': rmse,
        'Correlation Coefficient': correlation
    }
    return metrics