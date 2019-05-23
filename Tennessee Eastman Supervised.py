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

train_path = sorted(glob.glob('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\*.dat'))
test_path = sorted(glob.glob('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\*_te.dat'))


"""
I'd like to use an elegant solution to loading in all the data, but for the
moment I have to rely on something that works until I figure out how to do this

#Training sets
training = ["fault_1_x", "fault_2_x", "fault_3_x", "fault_4_x", "fault_5_x", "fault_6_x", "fault_7_x", "fault_8_x", "fault_9_x", "fault_10_x", "fault_11_x",
            "fault_12_x", "fault_13_x", "fault_14_x", "fault_15_x", "fault_16_x", "fault_17_x", "fault_18_x", "fault_19_x", "fault_20_x", "fault_21_x", "fault_22_x"]

# #testing sets
testing = ["fault_1_y", "fault_2_y", "fault_3_y", "fault_4_y", "fault_5_y", "fault_6_y", "fault_7_y", "fault_8_y", "fault_9_y", "fault_10_y", "fault_11_y",
           "fault_12_y", "fault_13_y", "fault_14_y", "fault_15_y", "fault_16_y", "fault_17_y", "fault_18_y", "fault_19_y", "fault_20_y", "fault_21_y", "fault_22_y"]

for name in training:
    np.loadtxt(train_path)
    
for name in testing:
    np.loadtxt(test_path)
"""
#training files
fault_1_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d00.dat', dtype = float)
fault_2_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d01.dat', dtype = float)
fault_3_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d02.dat', dtype = float)
fault_4_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d03.dat', dtype = float)
fault_5_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d04dat', dtype = float)
fault_6_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d05.dat', dtype = float)
fault_7_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d06.dat', dtype = float)
fault_8_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d07.dat', dtype = float)
fault_9_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d08.dat', dtype = float)
fault_10_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d09.dat', dtype = float)
fault_11_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d10.dat', dtype = float)
fault_12_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d11.dat', dtype = float)
fault_13_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d12.dat', dtype = float)
fault_14_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d13.dat', dtype = float)
fault_15_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d14.dat', dtype = float)
fault_16_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d15.dat', dtype = float)
fault_17_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d16.dat', dtype = float)
fault_18_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d17.dat', dtype = float)
fault_19_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d18.dat', dtype = float)
fault_20_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d19.dat', dtype = float)
fault_21_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d20.dat', dtype = float)
fault_22_x = np.loadtxt('E:\ML Projects\Tennessee Eastman\TE_process_dataset\Training Data\d21.dat', dtype = float)

#Testing files
fault_1_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d00_te.dat", dtype= float)
fault_2_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d01_te.dat", dtype= float)
fault_3_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d02_te.dat", dtype= float)
fault_4_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d03_te.dat", dtype= float)
fault_5_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d04_te.dat", dtype= float)
fault_6_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d05_te.dat", dtype= float)
fault_7_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d06_te.dat", dtype= float)
fault_8_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d07_te.dat", dtype= float)
fault_9_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d08_te.dat", dtype= float)
fault_10_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d09_te.dat", dtype= float)
fault_11_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d10_te.dat", dtype= float)
fault_12_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d11_te.dat", dtype= float)
fault_13_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d12_te.dat", dtype= float)
fault_14_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d13_te.dat", dtype= float)
fault_15_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d14_te.dat", dtype= float)
fault_16_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d15_te.dat", dtype= float)
fault_17_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d16_te.dat", dtype= float)
fault_18_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d17_te.dat", dtype= float)
fault_19_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d18_te.dat", dtype= float)
fault_20_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d19_te.dat", dtype= float)
fault_21_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d20_te.dat", dtype= float)
fault_22_y = np.loadtxt("E:\ML Projects\Tennessee Eastman\TE_process_dataset\Test Data\d21_te.dat", dtype= float)
