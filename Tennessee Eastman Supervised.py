# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:44:05 2019

@author: Robot Hands (github.com/nvd919)
"""
# =============================================================================
# The first hurdle to clear is getting all of the data from the 44 files loaded
# into the model. glob.glob() will handle all files fitting a common naming
# convention, but will return them in no particular order. Thus, we use 
# sorted() to counter that. This is of critical importance because the training
# data must correspond to the accompanied test data, otherwise the results
# will probably be totally useless; like studying Spanish for a Math exam.
# =============================================================================

import pandas as pd
import os

DATA_FOLDER = r"E:\ML Projects\Tennessee Eastman\TE_process_dataset"
NUM_FILES = 22

training_files = [os.path.join(DATA_FOLDER, "d{:0>2}.dat".format(i)) for i in range(NUM_FILES)]
test_files = [os.path.join(DATA_FOLDER, "d{:0>2}_te.dat".format(i)) for i in range(NUM_FILES)]

training_data = [pd.read_csv(f, sep=' ', index_col = None) for f in training_files]
test_data = [pd.read_csv(f, sep=' ') for f in test_files]
