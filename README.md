Start from here: 

 

    Modify the environment and directories based on your working system: 

 

Modify `bashenv`, `bashenv-train`, `directories.py` and `directories.sh`accordingly. 

 

    Run `HRES_PREP.sh`for preprocessing HRES data: 

 

This script automates the preprocessing of HRES forecast data. It copies, adjusts, and merges data files, organizing them into daily forecasts. Finally, it adds metadata attributes to the processed data before output. 

 

    Run `DL_PREP.py` using `run_DL_PREP.sh` for preparing the data for training and production in DL: 

This script utilizes argparse to specify lead time, specifications, unique filenames based on parameters, and prepares training and production data. 

 

    Run `DL_TRAIN.py` using `run_DL_TRAIN.sh` for training: 

This script defines training hyperparameters using argparse and prepares data for model training. It loads training data, constructs TensorFlow datasets, initializes a UNet model, compiles it with Adam optimizer, and defines callbacks for model checkpointing and early stopping. It then trains the model using defined datasets and saves the results. 

    Run `DL_PREDICT.py` using `run_DL_PREDICT.sh` for prediction: 

This script loads production data, initializes a UNet model, loads trained weights, and predicts mismatches using the pretrained weights for each lead-time model. 

 

    Run `HRES_POST.sh` and `HRES_POST2.sh` for post-processing the corrected data 