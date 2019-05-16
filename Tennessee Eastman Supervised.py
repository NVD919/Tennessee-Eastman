# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:44:05 2019

@author: Robot Hands (github.com/nvd919)
"""
import glob
import numpy as np

# =============================================================================
# The first hurdle to clear is getting all of the data from the 44 files loaded
# into the model. glob.glob() will handle all files fitting a common naming
# convention, but will return them in no particular order. Thus, we use 
# sorted() to counter that. This is of critical importance because the training
# data must correspond to the accompanied test data, otherwise the results
# will probably be totally useless; like studying Spanish for a Math exam.
# =============================================================================

train_path = sorted(glob.glob('E:\ML Projects\Tennessee Eastman\TE_process_dataset\d[00-21].dat'))
test_path = sorted(glob.glob('E:\ML Projects\Tennessee Eastman\TE_process_dataset\d[00-21]_te.dat'))

'''
#Training sets
training = [fault_1_x, fault_2_x, fault_3_x, fault_4_x, fault_5_x, fault_6_x, fault_7_x, fault_8_x, fault_9_x, fault_10_x, fault_11_x,
            fault_12_x, fault_13_x, fault_14_x, fault_15_x, fault_16_x, fault_17_x, fault_18_x, fault_19_x, fault_20_x, fault_21_x, fault_22_x]

 #testing sets
testing = [fault_1_y, fault_2_y, fault_3_y, fault_4_y, fault_5_y, fault_6_y, fault_7_y, fault_8_y, fault_9_y, fault_10_y, fault_11_y,
           fault_12_y, fault_13_y, fault_14_y, fault_15_y, fault_16_y, fault_17_y, fault_18_y, fault_19_y, fault_20_y,fault_21_y, fault_22_y]
'''