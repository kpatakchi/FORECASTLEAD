#!/bin/bash

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv

# Set the working directory
DATA_DIR="/p/data1/jibg33/patakchiyousefi1/postpro_data/DE05_HRES-RAW"
OUTPUT_FILE="${DATA_DIR}/sss_REF_merged.nc"

# Remove the existing output file, if it exists
rm -f "$OUTPUT_FILE"

# Navigate to the directory
cd "$DATA_DIR" || exit

# List all files matching the pattern and store in a variable
FILES=$(ls sss*0012-0024.nc | sort)

# Merge all files along the time dimension using cdo
cdo mergetime $FILES "$OUTPUT_FILE"

echo "Merged file saved to: $OUTPUT_FILE"
