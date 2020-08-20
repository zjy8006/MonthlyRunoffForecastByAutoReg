import matplotlib.pyplot as plt
import pandas as pd
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.models import esvr,multi_optimizer_esvr,esvr_multi_seed
from tools.DNN import BuildDNN

station = 'Huaxian'
predict_pattern = '7_ahead'
train_samples = pd.read_csv(root_path+'/'+station+'/data/'+predict_pattern+'_pacf_lag12/minmax_unsample_train.csv')
dev_samples = pd.read_csv(root_path+'/'+station+'/data/'+predict_pattern+'_pacf_lag12//minmax_unsample_dev.csv')
test_samples = pd.read_csv(root_path+'/'+station+'/data/'+predict_pattern+'_pacf_lag12//minmax_unsample_test.csv')
norm_id = pd.read_csv(root_path+'/'+station+'/data/'+predict_pattern+'_pacf_lag12//norm_unsample_id.csv')

BuildDNN(
    train_samples = train_samples,
    dev_samples = dev_samples,
    test_samples = test_samples,
    norm_id = norm_id,
    # model_path = root_path+'/'+station+'/projects/dnn/'+predict_pattern+'/',
    model_path = root_path+'\\'+station+'\\projects\\dnn\\'+predict_pattern+'\\',
    lags={"ORIG":12},
    seed = 1,
    batch_size=256,
    n_epochs = 500,
    max_trials = 50,
    executions_per_trial=3,
    max_hidden_layers = 2,
    min_units = 8,
    max_units = 32,
    unit_step = 8,
    min_droprate = 0.0,
    max_droprate = 0.5,
    droprate_step=0.1,
    min_learnrate=1e-4,
    max_learnrate=1e-1,
    n_tune_epochs = 500,
    measurement_time = 'month',
    measurement_unit = '$10^8m^3$',
)

