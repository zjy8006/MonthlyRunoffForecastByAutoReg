import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))

import sys
sys.path.append(root_path)
from tools.models import multi_step_esvr,multi_step_esvr_multi_seed
from Huaxian_dwt.projects.variables import variables

if __name__ == '__main__':
    lags=list((variables['lags_dict']['db10-2']).values())
    for i in range(1,len(lags)+1):
        multi_step_esvr(
            root_path=root_path,
            station='Huaxian',
            decomposer='dwt',
            predict_pattern='multi_step_1_ahead_forecast_pacf',
            lags = variables['lags_dict']['db10-2'],
            model_id=i,
            n_calls=100,
        )
    
    
    