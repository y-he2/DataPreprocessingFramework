USE_PYTORCH = True

from enum import Enum

import scipy
from scipy import signal
import pywt
import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None
## import ray
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import get_worker
from dask.distributed import Client, LocalCluster

ProgressBar().register()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from pyts.image import GramianAngularField, MarkovTransitionField

from sklearn import preprocessing

if( not USE_PYTORCH ):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import initializers

import json
import pickle
import os
import gc
import winsound
from datetime import datetime, timedelta
import time
import re 
import math

from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp
import itertools
from functools import partial
from collections import OrderedDict


# from sqlalchemy import create_engine
# import pyodbc
# import urllib
# import turbodbc


from IPython.display import clear_output


######## Preprocessor ########
# Preprocessor


######## Trainer ########
# Feature Chunk Names:

# Model Definition
MODEL_DIR = "models/"
FINAL_MODEL_PATH = MODEL_DIR + "model_final.h5"
MIDDLE_MODEL_PATH = MODEL_DIR + "model_middle.h5"
TENSORBOARD_LOG_DIR = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Model Fitting
# BATCH_SIZE = 4 # Ideal: 256?
# TOTAL_EPOCHS = 256 # Ideal: 128

#MAIN_IP_ADRESS = "192.168.1.1"