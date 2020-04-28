import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__')) 
import sys
sys.path.append(root_path)
from tools.ensembler import ensemble
from Zhangjiashan_modwt.projects.variables import variables

# Set the project parameters
ORIGINAL = 'Zhangjiashan1953-2018(1953-2018).xlsx'
STATION = 'Zhangjiashan'
DECOMPOSER = 'modwt' 
PREDICTOR = 'esvr' # esvr or gbrt or lstm
wavelet_level='db1-4'
# ensemble(
#         root_path=root_path,
#         original_series=ORIGINAL,
#         station=STATION,
#         decomposer=DECOMPOSER,
#         variables = variables,
#         predictor=PREDICTOR,
#         predict_pattern='single_hybrid_1_ahead',
#         wavelet_level=wavelet_level,
#     )
for lead_time in [1,3,5,7]:
    ensemble(
            root_path=root_path,
            original_series=ORIGINAL,
            station=STATION,
            decomposer=DECOMPOSER,
            variables = variables,
            predictor=PREDICTOR,
            predict_pattern='single_hybrid_'+str(lead_time)+'_ahead_lag12_mi_ts0.1',
            wavelet_level=wavelet_level,
        )
# for n_components in range(32,49):
#     ensemble(
#         root_path=root_path,
#         original_series=ORIGINAL,
#         station=STATION,
#         decomposer=DECOMPOSER,
#         variables = variables,
#         predictor=PREDICTOR,
#         predict_pattern='single_hybrid_1_ahead_pca'+str(n_components),
#         wavelet_level=wavelet_level,
#     )
