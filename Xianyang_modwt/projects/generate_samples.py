import pandas as pd
import sys
from variables import variables
import os
root_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(root_path)
from tools.samples_generator import gen_direct_forecast_samples
from tools.samples_generator import gen_wddff_samples

wavelet_level='db1-4'
lag = 12
data = pd.read_csv(root_path+'/Xianyang_modwt/data-wddff/'+wavelet_level+'/1_ahead_lag12_calibration.csv')
X = data.drop('Y',axis=1)
num_in_one = X.shape[1]

for lead_time in [1,3,5,7]:
    gen_wddff_samples(
        data_path = root_path+'/Xianyang_modwt/data-wddff/'+wavelet_level+'/',
        lead_time=lead_time,
        mode='MI',
        lag=lag,
        use_full=True,
    )



