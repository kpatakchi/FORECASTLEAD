#homedir="/home/yousefi" #if using local
homedir=""              #if using hpc
#define all the necessary directories:
PPROJECT_DIR="/p/project1/deepacf/kiste/patakchiyousefi1/"
PSCRATCH_DIR="/p/scratch/deepacf/kiste/patakchiyousefi1/"
PPROJECT_DIR2="/p/project1/cesmtst/patakchiyousefi1/"
PSCRATCH_DIR2="/p/scratch/cesmtst/patakchiyousefi1/"
PROJECT_NAME="/FORECASTLEAD/"

# General directories
TRAIN_FILES=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"TRAIN_FILES"
DUMP_PLOT=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"DUMP_PLOT"
PREDICT_FILES=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"PREDICT_FILES"
STATS=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"STATS"
PRODUCE_FILES=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"PRODUCE_FILES"

#HRES directories:
HRES_OR=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"HRES_OR"
HRES_PREP=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"HRES_PREP"
HRES_DUMP=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"HRES_DUMP"
HRES_DUMP2=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"HRES_DUMP2"
HRES_DUMP3=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"HRES_DUMP3"
HRES_DUMP4=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"HRES_DUMP4"
HRES_DUMP5=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"HRES_DUMP5"
HRES_DUMP6=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"HRES_DUMP6"
HRES_POST=homedir+PSCRATCH_DIR2+"/FORECASTLEAD_CESMTST_SCRATCH/"+"HRES_POST"

# PARFLOW Directories
PARFLOWCLM=homedir+PSCRATCH_DIR2+PROJECT_NAME+"PF_FORECASTLEAD"
#ORIG_HRES=PARFLOWCLM+"/sim/ADAPTER_DE05_ECMWF-HRES_detforecast__FZJ-IBG3-ParFlowCLM380D_v03bJuwelsGpuProdClimatologyTl_PRhourly/forcing/o.data.MARS_retrieval"
#COR_HRES=PARFLOWCLM+"/sim/ADAPTER_DE05_ECMWF-HRES_detforecast__FZJ-IBG3-ParFlowCLM380D_v03bJuwelsGpuProdClimatologyTl_PRhourly_PRCORRECTED/forcing/o.data.MARS_retrieval"