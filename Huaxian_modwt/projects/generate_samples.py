import pandas as pd

import sys
from variables import variables
import os
root_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(root_path)
from tools.samples_generator import gen_multi_forecast_samples # is forecast for modwt
from tools.samples_generator import gen_direct_forecast_samples # is forecast for modwt
from tools.samples_generator import gen_wddff_samples # is forecast for modwt

# generate samples for WDDFF framework
wavelet_level='db1-4'
lag = 12
# data = pd.read_csv(root_path+'/Huaxian_modwt/data-wddff/'+wavelet_level+'/1_ahead_lag'+str(lag)+'_calibration.csv')
# X = data.drop('Y',axis=1)
# num_in_one = X.shape[1]

for lead_time in [1,3,5,7]:
    gen_wddff_samples(
            data_path = root_path+'/Huaxian_modwt/data-wddff/'+wavelet_level+'/',
            lead_time=lead_time,
            lag=lag,
            use_full=True,
            mode='MI',
    )

# gen_wddff_samples(
#     data_path = root_path+'/Huaxian_modwt/data-wddff/'+wavelet_level+'/',
#     lead_time=1,
#     mode=None,
# )
 
# for lead_time in [1,3,5,7,9]:
#     for threshold in [0.1,0.2,0.3,0.4,0.5]:
#         gen_wddff_samples(
#             data_path = root_path+'/Huaxian_modwt/data-wddff/'+wavelet_level+'/',
#             lead_time=lead_time,
#             threshold=threshold,
#         )

# for n_components in range(num_in_one-16,num_in_one+1):
#     gen_wddff_samples(
#         data_path = root_path+'/Huaxian_modwt/data-wddff/'+wavelet_level+'/',
#         lead_time=1,
#         mode='PCA',
#         n_components=n_components,
#     )

# Generate smaples for TSDP framework
from tools.samples_generator import gen_multi_forecast_samples
from tools.samples_generator import gen_direct_forecast_samples
from tools.samples_generator import gen_direct_hindcast_samples

# gen_direct_forecast_samples_triandev_test(
#     station="Huaxian",
#     decomposer="modwt",
#     lags_dict=variables['lags_dict'],
#     input_columns=['D1', 'D2', 'A2', ],
#     output_column=['ORIG'],
#     start=533,
#     stop=792,
#     test_len=120,
# )

# for lead_time in [1,]:
#     gen_direct_forecast_samples(
#         station='Huaxian',
#         decomposer="modwt",
#         lags_dict=variables['lags_dict'],
#         input_columns=['D1', 'D2', 'A2', ],
#         output_column=['ORIG'],
#         start=533,
#         stop=792,
#         test_len=120,
#         mode='PACF',
#         lead_time=lead_time,

#     )
# gen_direct_hindcast_samples(
#         station='Huaxian',
#         decomposer="modwt",
#         lags_dict=variables['lags_dict'],
#         input_columns=['D1', 'D2', 'D3','D4','A4', ],
#         output_column=['ORIG'],
#         test_len=120,
#         mode='PACF',
#         lead_time=1,
#         wavelet_level='db10-4',
#     )
# for lead_time in [3, 5, 7, 9]:
#     for threshold in [0.1,0.2,0.3,0.4,0.5]:
#         gen_direct_forecast_samples(
#             station='Huaxian',
#             decomposer="modwt",
#             lags_dict=variables['lags_dict'],
#             input_columns=['D1', 'D2', 'A2', ],
#             output_column=['ORIG'],
#             start=533,
#             stop=792,
#             test_len=120,
#             mode='Pearson',
#             filter_boundary=threshold,
#             lead_time=lead_time,
#         )


# gen_multi_forecast_samples(
#     station='Huaxian',
#     decomposer="modwt",
#     lags_dict=variables['lags_dict'],
#     columns=['D1', 'D2', 'A2', ],
#     start=533,
#     stop=792,
#     test_len=120,
# )

# gen_direct_forecast_samples(
#     station='Huaxian',
#     decomposer="modwt",
#     lags_dict=variables['lags_dict'],
#     input_columns=['D1', 'D2', 'A2', ],
#     output_column=['ORIG'],
#     start=533,
#     stop=792,
#     test_len=120,
#     mode='PACF',
#     lead_time=1,
#     n_components='mle',
# )

# gen_direct_forecast_samples(
#     station='Huaxian',
#     decomposer="modwt",
#     lags_dict=variables['lags_dict'],
#     input_columns=['D1', 'D2', 'A2', ],
#     output_column=['ORIG'],
#     start=533,
#     stop=792,
#     test_len=120,
#     mode='PACF',
#     lead_time=1,
#     n_components=28,
# )

# num_in_one = sum(variables['lags_dict']['db10-2'].values())
# for n_components in range(num_in_one-16,num_in_one+1):
#     gen_direct_forecast_samples(
#         station='Huaxian',
#         decomposer="modwt",
#         lags_dict=variables['lags_dict'],
#         input_columns=['D1', 'D2', 'A2', ],
#         output_column=['ORIG'],
#         start=533,
#         stop=792,
#         test_len=120,
#         mode='PACF',
#         lead_time=1,
#         n_components=n_components,
#     )