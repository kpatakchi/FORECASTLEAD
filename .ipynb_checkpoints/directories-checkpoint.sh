#!/bin/bash

# Define Python-related variables:
homedir=""  # Update this with the appropriate value

PSCRATCH_DIR="/p/scratch/deepacf/kiste/patakchiyousefi1/FORECASTLEAD_KISTE_SCRATCH/"
PSCRATCH_DIR2="/p/scratch/cesmtst/patakchiyousefi1/FORECASTLEAD_CESMTST_SCRATCH/"
PPROJECT_DIR="/p/project1/deepacf/kiste/patakchiyousefi1/"
PPROJECT_DIR2="/p/project1/cesmtst/patakchiyousefi1/"

# Define directories:
export PF_FORECASTLEAD="${homedir}${PSCRATCH_DIR2}PF_FORECASTLEAD"

# Define HRES directories:
export HRES_UTI="${homedir}${PPROJECT_DIR}IO/hres_ut/"
export HRES_LOG="${homedir}${PPROJECT_DIR2}CODES-MS3/FORECASTLEAD/LOGS"
export HRES_TOPO="${homedir}${PPROJECT_DIR}HRES_TOPO"
export HRES_RET="/p/project1/pfgpude05/belleflamme1/ADAPTER_DE05_ECMWF-HRES-ENS-SEAS_FZJ-IBG3-ParFlowCLM_atmospheric_forcing/o.data.MARS_retrieval/HRES"
export HRES_OR="${homedir}${PSCRATCH_DIR2}HRES_OR"
export HRES_PREP="${homedir}${PSCRATCH_DIR2}HRES_PREP"
export HRES_DUMP="${homedir}${PSCRATCH_DIR2}HRES_DUMP"
export HRES_DUMP2="${homedir}${PSCRATCH_DIR2}HRES_DUMP2"
export HRES_DUMP3="${homedir}${PSCRATCH_DIR2}HRES_DUMP3"
export HRES_DUMP4="${homedir}${PSCRATCH_DIR2}HRES_DUMP4"
export HRES_DUMP5="${homedir}${PSCRATCH_DIR2}HRES_DUMP5"
export HRES_DUMP6="${homedir}${PSCRATCH_DIR2}HRES_DUMP6"
export HRES_POST="${homedir}${PSCRATCH_DIR2}HRES_POST"
export STATS="${homedir}${PSCRATCH_DIR2}STATS"

# Define DL directories:
export TRAIN_FILES="${homedir}${PSCRATCH_DIR2}TRAIN_FILES"
export HPT_DIR="${homedir}${PPROJECT_DIR2}HPT"
export PREDICT_FILES="${homedir}${PSCRATCH_DIR2}PREDICT_FILES"
export PRODUCE_FILES="${homedir}${PSCRATCH_DIR2}PRODUCE_FILES"


# Create the directories if they don't exist
mkdir -p "$PF_FORECASTLEAD"
mkdir -p "$HRES_UTI"
mkdir -p "$HRES_LOG"
mkdir -p "$HRES_TOPO"
mkdir -p "$HRES_RET"
mkdir -p "$HRES_OR"
mkdir -p "$HRES_PREP"
mkdir -p "$HRES_DUMP"
mkdir -p "$HRES_DUMP2"
mkdir -p "$HRES_DUMP3"
mkdir -p "$HRES_DUMP4"
mkdir -p "$HRES_DUMP5"
mkdir -p "$HRES_DUMP6"
mkdir -p "$HRES_POST"
mkdir -p "$STATS"
mkdir -p "$TRAIN_FILES"
mkdir -p "$HPT_DIR"
mkdir -p "$PREDICT_FILES"
mkdir -p "$PRODUCE_FILES"

echo "All necessary directories have been created or already exist."