#!/bin/bash

# Define Python-related variables:
homedir=""  # Update this with the appropriate value

PSCRATCH_DIR="/p/scratch/deepacf/kiste/patakchiyousefi1/FORECASTLEAD_KISTE_SCRATCH/"
PSCRATCH_DIR2="/p/scratch/cesmtst/patakchiyousefi1/FORECASTLEAD_CESMTST_SCRATCH/"
PPROJECT_DIR="/p/project/deepacf/kiste/patakchiyousefi1/"
PPROJECT_DIR2="/p/project/cesmtst/patakchiyousefi1/"

# Define directories:
export PF_FORECASTLEAD="${homedir}${PSCRATCH_DIR2}PF_FORECASTLEAD"

# Define HRES directories:
export HRES_UTI="${homedir}${PPROJECT_DIR}IO/hres_ut/"
export HRES_LOG="${homedir}${PPROJECT_DIR2}CODES-MS3/FORECASTLEAD/LOGS"
export HRES_TOPO="${homedir}${PPROJECT_DIR}HRES_TOPO"
export HRES_RET="/p/project/pfgpude05/belleflamme1/ADAPTER_DE05_ECMWF-HRES-ENS-SEAS_FZJ-IBG3-ParFlowCLM_atmospheric_forcing/o.data.MARS_retrieval/HRES"
export HRES_OR="${homedir}${PSCRATCH_DIR2}HRES_OR"
export HRES_PREP="${homedir}${PSCRATCH_DIR2}HRES_PREP"
export HRES_DUMP="${homedir}${PSCRATCH_DIR2}HRES_DUMP"
export HRES_DUMP2="${homedir}${PSCRATCH_DIR2}HRES_DUMP2"
export HRES_DUMP3="${homedir}${PSCRATCH_DIR2}HRES_DUMP3"
export HRES_DUMP4="${homedir}${PSCRATCH_DIR2}HRES_DUMP4"
export HRES_DUMP5="${homedir}${PSCRATCH_DIR2}HRES_DUMP5"
export HRES_DUMP6="${homedir}${PSCRATCH_DIR2}HRES_DUMP6"
export HRES_POST="${homedir}${PSCRATCH_DIR2}HRES_POST"

# Define DL directories:
export TRAIN_FILES="${homedir}${PSCRATCH_DIR2}TRAIN_FILES"
export HPT_DIR="${homedir}${PPROJECT_DIR2}HPT"
export PREDICT_FILES="${homedir}${PPROJECT_DIR2}PREDICT_FILES"
export PRODUCE_FILES="${homedir}${PPROJECT_DIR2}PRODUCE_FILES"