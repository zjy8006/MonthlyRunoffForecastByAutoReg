import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.models import multi_step_esvr,multi_step_esvr_multi_seed
from Zhangjiashan_vmd.projects.variables import variables


if __name__ == '__main__':
    lags=list((variables['lags_dict']).values())
    for i in range(1,len(lags)+1):
        multi_step_esvr(
            root_path=root_path,
            station='Zhangjiashan',
            decomposer='vmd',
            predict_pattern='multi_step_1_ahead_forecast_pacf',
            lags = variables['lags_dict'],
            model_id=i,
            n_calls=100,
        )
    