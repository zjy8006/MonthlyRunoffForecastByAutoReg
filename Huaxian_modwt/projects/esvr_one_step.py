
import sys
import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(root_path)
from tools.models import one_step_esvr, one_step_esvr_multi_seed
from Huaxian_modwt.projects.variables import variables

if __name__ == '__main__':


    for lead_time in [1,3,5,7]:
        one_step_esvr_multi_seed(
            root_path=root_path,
            station='Huaxian',
            decomposer='modwt',
            predict_pattern='single_hybrid_'+str(lead_time)+'_ahead_lag12_mi_ts0.1',# forecast or forecast or forecast_with_pca_mle or forecast_with_pca_mle
            n_calls=100,
            wavelet_level='db1-4',
        )

    # one_step_esvr_multi_seed(
    #     root_path=root_path,
    #     station='Huaxian',
    #     decomposer='modwt',
    #     predict_pattern='single_hybrid_1_ahead',# forecast or forecast or forecast_with_pca_mle or forecast_with_pca_mle
    #     n_calls=100,
    #     wavelet_level=wavelet_level,
    # )
    # for lead_time in [1,3,5,7,9]:
    #     one_step_esvr_multi_seed(
    #         root_path=root_path,
    #         station='Huaxian',
    #         decomposer='modwt',
    #         predict_pattern='single_hybrid_'+str(lead_time)+'_ahead_mi_ts0.1',# forecast or forecast or forecast_with_pca_mle or forecast_with_pca_mle
    #         n_calls=100,
    #         wavelet_level=wavelet_level,
    #     )
    
    # for n_components in range(32,49):
    #     one_step_esvr_multi_seed(
    #         root_path=root_path,
    #         station='Huaxian',
    #         decomposer='modwt',
    #         predict_pattern='single_hybrid_1_ahead_pca'+str(n_components),# forecast or forecast or forecast_with_pca_mle or forecast_with_pca_mle
    #         n_calls=100,
    #         wavelet_level=wavelet_level,
    #     )

    

    