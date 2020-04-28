import matplotlib.pyplot as plt
import pandas as pd
import os
root_path = os.path.dirname(os.path.abspath('__file__')) 
import sys
sys.path.append(root_path)
from tools.ensembler import ensemble
from Zhangjiashan_dwt.projects.variables import variables

# Set the project parameters
ORIGINAL = 'ZhangjiashanRunoff1953-2018(1953-2018).xlsx'
STATION = 'Zhangjiashan'
DECOMPOSER = 'dwt' 
PREDICTOR = 'esvr' # esvr or gbrt or lstm
orig_series = pd.read_excel(root_path+'/time_series/ZhangjiashanRunoff1953-2018(1953-2018).xlsx')['MonthlyRunoff']
orig_series = orig_series[orig_series.shape[0]-variables['full_len']:]
orig_series = orig_series.reset_index(drop=True)



ensemble(
    root_path=root_path,
    original_series=ORIGINAL,
    station=STATION,
    decomposer=DECOMPOSER,
    variables = variables,
    predictor=PREDICTOR,
    predict_pattern='one_step_1_ahead_forecast_pacf_train_val',
)
ensemble(
    root_path=root_path,
    original_series=ORIGINAL,
    station=STATION,
    decomposer=DECOMPOSER,
    variables = variables,
    predictor=PREDICTOR,
    predict_pattern='one_step_1_ahead_forecast_pacf_traindev_test',
)
ensemble(
    root_path=root_path,
    original_series=ORIGINAL,
    station=STATION,
    decomposer=DECOMPOSER,
    variables = variables,
    predictor=PREDICTOR,
    predict_pattern='one_step_1_ahead_forecast_pacf_traindev_append',
)

for lead_time in [1,3,5,7,9]:
    ensemble(
        root_path=root_path,
        original_series=ORIGINAL,
        station=STATION,
        decomposer=DECOMPOSER,
        variables = variables,
        predictor=PREDICTOR,
        predict_pattern='one_step_'+str(lead_time)+'_ahead_forecast_pacf',
    )
for lead_time in [1,3,5,7,9]:
    ensemble(
        root_path=root_path,
        original_series=ORIGINAL,
        station=STATION,
        decomposer=DECOMPOSER,
        variables = variables,
        predictor=PREDICTOR,
        predict_pattern='one_step_'+str(lead_time)+'_ahead_forecast_pcc_local',
    )

ensemble(
        root_path=root_path,
        original_series=ORIGINAL,
        station=STATION,
        decomposer=DECOMPOSER,
        variables = variables,
        predictor=PREDICTOR,
        predict_pattern='one_step_1_ahead_forecast_pacf_pca28',
    )

ensemble(
        root_path=root_path,
        original_series=ORIGINAL,
        station=STATION,
        decomposer=DECOMPOSER,
        variables = variables,
        predictor=PREDICTOR,
        predict_pattern='one_step_1_ahead_forecast_pacf_pcamle',
    )

num_in_one = sum(variables['lags_dict']['db10-2'].values())
for n_components in range(num_in_one-16,num_in_one+1):
    ensemble(
        root_path=root_path,
        original_series=ORIGINAL,
        station=STATION,
        decomposer=DECOMPOSER,
        variables = variables,
        predictor=PREDICTOR,
        predict_pattern='one_step_1_ahead_forecast_pacf_pca'+str(n_components),
    )
ensemble(
    root_path=root_path,
    original_series=orig_series,
    station=STATION,
    decomposer=DECOMPOSER,
    variables = variables,
    predictor=PREDICTOR,
    predict_pattern='multi_step_1_ahead_forecast_pacf',
)