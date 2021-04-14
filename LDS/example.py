#based on example.m
import numpy as np

#import our library based on MATLAB function
from OnlineLDS_library import *

#settings_file_name = 'MATLAB_func_on_python/data/setting1.mat'
settings_file_name = 'OARIMA_code_data/data/setting1.mat'

# MATLAB:
# load('../data/setting1.mat')
data_in = loadmat(settings_file_name) #loading seq_d0(1x10000), seq_d1(1x10000)
seq_d1 = data_in['seq_d1'][0]

# create object 'options' from class 'ClassOptions'
# to mimic 'options' from MATLAB
options = ClassOptions()

# MATLAB:
# options.mk = 10;
# options.init_w = rand([1, options.mk]);
# options.t_tick = 1;
options.mk = 10

#Uniform distribution array with options.mk number of columns.
options.init_w = np.random.rand(1, options.mk)

options.t_tick = 1
options.lrate = 1

# MATLAB: [RMSE_ogd1,w] = arima_ogd(seq_d1,options);
RMSE_ogd1, w = arima_ogd(seq_d1, options)
print('arima_ogd output: shape(RMSE_ogd1) = %s; shape(w) = %s' %(np.shape(RMSE_ogd1), np.shape(w)))
########

# MATLAB:
#options.lrate = 1.75;
#options.epsilon = 10 ^ -0.5;
#[RMSE_ons1, w] = MATLAB_lib_on_python.arima_ons(seq_d1, options);
###
options.lrate = 1.75
options.epsilon = 10 ** (-0.5)
RMSE_ons1, w = arima_ons(seq_d1, options)
print('arima_ons output: shape(RMSE_ons1) = %s; shape(w) = %s' %(np.shape(RMSE_ons1), np.shape(w)))
########

# MATLAB:
#options.lrate = 10^3;
#options.epsilon=10^-5.5;
#[RMSE_ons0,w] = arima_ons(seq_d0,options);
options.lrate = 10 ** 3
options.epsilon = 10 ** (-5.5)
RMSE_ons1, w = arima_ons(seq_d1, options)
print('arima_ons output: shape(RMSE_ons1) = %s; shape(w) = %s' %(np.shape(RMSE_ons1), np.shape(w)))


