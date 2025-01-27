#!/bin/bash
module purge  # Clean up loaded modules to avoid conflicts

# Define environment paths
export CONDA_TARGET_DIR=/p/project1/cesmtst/patakchiyousefi1/VENV
rm -r $CONDA_TARGET_DIR/*
CONDA_ENV=forecastlead

# Remove Jupyter kernel if it exists
HOME=/p/home/jusers/patakchiyousefi1/juwels
KERNEL_DIR=$HOME/.local/share/jupyter/kernels/conda_${CONDA_ENV}
if [ -d "$KERNEL_DIR" ]; then
    rm -rf $KERNEL_DIR
fi

# Download and install Miniconda
wget --output-document=$CONDA_TARGET_DIR/Miniconda3.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash $CONDA_TARGET_DIR/Miniconda3.sh -b -u -p ${CONDA_TARGET_DIR}
${CONDA_TARGET_DIR}/bin/conda init bash
${CONDA_TARGET_DIR}/bin/conda config --set auto_activate_base false

# Create a new conda environment with Python 3.9 and ipykernel
${CONDA_TARGET_DIR}/bin/conda create -n ${CONDA_ENV} -y python=3.9 ipykernel

# Set up kernel script for Jupyter
echo '#!/bin/bash
module purge
source '"${CONDA_TARGET_DIR}"'/bin/activate '"${CONDA_ENV}"'
export PYTHONPATH=${CONDA_PREFIX}/lib/python3.8/site-packages:${PYTHONPATH}
exec python -m ipykernel $@' > ${CONDA_TARGET_DIR}/envs/${CONDA_ENV}/kernel.sh
chmod +x ${CONDA_TARGET_DIR}/envs/${CONDA_ENV}/kernel.sh
HOME=/p/home/jusers/patakchiyousefi1/juwels
# Create Jupyter kernel
mkdir -p $HOME/.local/share/jupyter/kernels/conda_${CONDA_ENV}
echo '{
 "argv": [
  "'"${CONDA_TARGET_DIR}"'/envs/'"${CONDA_ENV}"'/kernel.sh",
  "-f",
  "{connection_file}"
 ],
 "display_name": "'"${CONDA_ENV}"'",
 "language": "python"
}' > $HOME/.local/share/jupyter/kernels/conda_${CONDA_ENV}/kernel.json

source ${CONDA_TARGET_DIR}/bin/activate ${CONDA_ENV}

# Update conda to the latest version
#conda update -n base -c defaults -y conda

# Install packages via conda from conda-forge channel
conda install -c conda-forge -y cartopy xarray dask matplotlib pandas basemap scikit-learn opencv holoviews datashader netCDF4 bokeh panel pillow glob2 imageio
