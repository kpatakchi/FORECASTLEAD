from py_env_train import *
import warnings

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


import numpy as np
import xarray as xr
import warnings

def calculate_metrics(reference, model):
    """
    Calculate temporally averaged mean error, root mean squared error, 
    and correlation coefficient between reference and model datasets, 
    aligning them in time first.

    Parameters:
        reference (xarray.DataArray): The reference dataset.
        model (xarray.DataArray): The model dataset.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """

    # Align datasets on 'time' dimension, keeping only overlapping time steps
    reference_aligned, model_aligned = xr.align(reference, model, join='inner')

    # Load into memory to avoid netCDF4 backend issues
    reference_aligned = reference_aligned.load()
    model_aligned = model_aligned.load()

    # Optional: warn if number of aligned time steps is reduced
    if reference.time.size != reference_aligned.time.size:
        warnings.warn(
            f"Time mismatch detected. Using {reference_aligned.time.size} overlapping time steps "
            f"from {reference.time.size} (ref) and {model.time.size} (model)."
        )

    # Compute metrics
    mean_error = (model_aligned - reference_aligned).mean(dim='time', skipna=True)

    squared_error = (model_aligned - reference_aligned) ** 2
    mse = squared_error.mean(dim='time', skipna=True)
    rmse = np.sqrt(mse)

    correlation = xr.corr(model_aligned, reference_aligned, dim='time')

    # Return as dictionary
    return {
        'Mean Error': mean_error,
        'Root Mean Squared Error': rmse,
        'Correlation Coefficient': correlation
    }
