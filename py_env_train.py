# python packages and directories
from directories import *
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
import tensorflow as tf
import numpy as np
import func_train
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import shutil

#------------------------#
#shutil.rmtree(DUMP_PLOT)
#os.mkdir(DUMP_PLOT)
