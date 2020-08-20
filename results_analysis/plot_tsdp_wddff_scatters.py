import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
plt.rcParams['lines.markersize'] =4
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/graphs/'
print(root_path)
import sys
sys.path.append(root_path)
from tools.results_reader import read_two_stage,read_pure_esvr,read_pure_arma
from tools.fit_line import compute_linear_fit,compute_multi_linear_fit

h_arma = read_pure_arma("Huaxian")
x_arma = read_pure_arma("Xianyang")
z_arma = read_pure_arma("Zhangjiashan")

h_svr_1 = pd.read_csv(root_path+'/Huaxian/projects/esvr/1_ahead_pacf_lag12/optimal_model_results.csv')
h_svr_3 = pd.read_csv(root_path+'/Huaxian/projects/esvr/3_ahead_pacf_lag12/optimal_model_results.csv')
h_svr_5 = pd.read_csv(root_path+'/Huaxian/projects/esvr/5_ahead_pacf_lag12/optimal_model_results.csv')
h_svr_7 = pd.read_csv(root_path+'/Huaxian/projects/esvr/7_ahead_pacf_lag12/optimal_model_results.csv')


x_svr_1 = pd.read_csv(root_path+'/Xianyang/projects/esvr/1_ahead_pacf_lag12/optimal_model_results.csv')
x_svr_3 = pd.read_csv(root_path+'/Xianyang/projects/esvr/3_ahead_pacf_lag12/optimal_model_results.csv')
x_svr_5 = pd.read_csv(root_path+'/Xianyang/projects/esvr/5_ahead_pacf_lag12/optimal_model_results.csv')
x_svr_7 = pd.read_csv(root_path+'/Xianyang/projects/esvr/7_ahead_pacf_lag12/optimal_model_results.csv')


z_svr_1 = pd.read_csv(root_path+'/Zhangjiashan/projects/esvr/1_ahead_pacf_lag12/optimal_model_results.csv')
z_svr_3 = pd.read_csv(root_path+'/Zhangjiashan/projects/esvr/3_ahead_pacf_lag12/optimal_model_results.csv')
z_svr_5 = pd.read_csv(root_path+'/Zhangjiashan/projects/esvr/5_ahead_pacf_lag12/optimal_model_results.csv')
z_svr_7 = pd.read_csv(root_path+'/Zhangjiashan/projects/esvr/7_ahead_pacf_lag12/optimal_model_results.csv')

h_lstm_1 = pd.read_csv(root_path+'/Huaxian/projects/lstm/1_ahead/optimal/opt_pred.csv')
h_lstm_3 = pd.read_csv(root_path+'/Huaxian/projects/lstm/3_ahead/optimal/opt_pred.csv')
h_lstm_5 = pd.read_csv(root_path+'/Huaxian/projects/lstm/5_ahead/optimal/opt_pred.csv')
h_lstm_7 = pd.read_csv(root_path+'/Huaxian/projects/lstm/7_ahead/optimal/opt_pred.csv')

x_lstm_1 = pd.read_csv(root_path+'/Xianyang/projects/lstm/1_ahead/optimal/opt_pred.csv')
x_lstm_3 = pd.read_csv(root_path+'/Xianyang/projects/lstm/3_ahead/optimal/opt_pred.csv')
x_lstm_5 = pd.read_csv(root_path+'/Xianyang/projects/lstm/5_ahead/optimal/opt_pred.csv')
x_lstm_7 = pd.read_csv(root_path+'/Xianyang/projects/lstm/7_ahead/optimal/opt_pred.csv')

z_lstm_1 = pd.read_csv(root_path+'/Zhangjiashan/projects/lstm/1_ahead/optimal/opt_pred.csv')
z_lstm_3 = pd.read_csv(root_path+'/Zhangjiashan/projects/lstm/3_ahead/optimal/opt_pred.csv')
z_lstm_5 = pd.read_csv(root_path+'/Zhangjiashan/projects/lstm/5_ahead/optimal/opt_pred.csv')
z_lstm_7 = pd.read_csv(root_path+'/Zhangjiashan/projects/lstm/7_ahead/optimal/opt_pred.csv')


h_dnn_1 = pd.read_csv(root_path+'/Huaxian/projects/dnn/1_ahead/optimal/opt_pred.csv')
h_dnn_3 = pd.read_csv(root_path+'/Huaxian/projects/dnn/3_ahead/optimal/opt_pred.csv')
h_dnn_5 = pd.read_csv(root_path+'/Huaxian/projects/dnn/5_ahead/optimal/opt_pred.csv')
h_dnn_7 = pd.read_csv(root_path+'/Huaxian/projects/dnn/7_ahead/optimal/opt_pred.csv')

x_dnn_1 = pd.read_csv(root_path+'/Xianyang/projects/dnn/1_ahead/optimal/opt_pred.csv')
x_dnn_3 = pd.read_csv(root_path+'/Xianyang/projects/dnn/3_ahead/optimal/opt_pred.csv')
x_dnn_5 = pd.read_csv(root_path+'/Xianyang/projects/dnn/5_ahead/optimal/opt_pred.csv')
x_dnn_7 = pd.read_csv(root_path+'/Xianyang/projects/dnn/7_ahead/optimal/opt_pred.csv')

z_dnn_1 = pd.read_csv(root_path+'/Zhangjiashan/projects/dnn/1_ahead/optimal/opt_pred.csv')
z_dnn_3 = pd.read_csv(root_path+'/Zhangjiashan/projects/dnn/3_ahead/optimal/opt_pred.csv')
z_dnn_5 = pd.read_csv(root_path+'/Zhangjiashan/projects/dnn/5_ahead/optimal/opt_pred.csv')
z_dnn_7 = pd.read_csv(root_path+'/Zhangjiashan/projects/dnn/7_ahead/optimal/opt_pred.csv')


h_d_1 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_d_3 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
h_d_5 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
h_d_7 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
h_e_1 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_e_3 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
h_e_5 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
h_e_7 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
h_s_1 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_s_3 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
h_s_5 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
h_s_7 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
h_v_1 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_v_3 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
h_v_5 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
h_v_7 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
h_m_1 = pd.read_csv(root_path+'/Huaxian_modwt/projects/esvr-wddff/db1-4/single_hybrid_1_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
h_m_3 = pd.read_csv(root_path+'/Huaxian_modwt/projects/esvr-wddff/db1-4/single_hybrid_3_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
h_m_5 = pd.read_csv(root_path+'/Huaxian_modwt/projects/esvr-wddff/db1-4/single_hybrid_5_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
h_m_7 = pd.read_csv(root_path+'/Huaxian_modwt/projects/esvr-wddff/db1-4/single_hybrid_7_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
x_d_1 = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_d_3 = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
x_d_5 = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
x_d_7 = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
x_e_1 = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_e_3 = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
x_e_5 = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
x_e_7 = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
x_s_1 = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_s_3 = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
x_s_5 = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
x_s_7 = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
x_v_1 = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_v_3 = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
x_v_5 = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
x_v_7 = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
x_m_1 = pd.read_csv(root_path+'/Xianyang_modwt/projects/esvr-wddff/db1-4/single_hybrid_1_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
x_m_3 = pd.read_csv(root_path+'/Xianyang_modwt/projects/esvr-wddff/db1-4/single_hybrid_3_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
x_m_5 = pd.read_csv(root_path+'/Xianyang_modwt/projects/esvr-wddff/db1-4/single_hybrid_5_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
x_m_7 = pd.read_csv(root_path+'/Xianyang_modwt/projects/esvr-wddff/db1-4/single_hybrid_7_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
z_d_1 = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_d_3 = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
z_d_5 = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
z_d_7 = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
z_e_1 = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_e_3 = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
z_e_5 = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
z_e_7 = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
z_s_1 = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_s_3 = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
z_s_5 = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
z_s_7 = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
z_v_1 = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_v_3 = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
z_v_5 = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
z_v_7 = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
z_m_1 = pd.read_csv(root_path+'/Zhangjiashan_modwt/projects/esvr-wddff/db1-4/single_hybrid_1_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
z_m_3 = pd.read_csv(root_path+'/Zhangjiashan_modwt/projects/esvr-wddff/db1-4/single_hybrid_3_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
z_m_5 = pd.read_csv(root_path+'/Zhangjiashan_modwt/projects/esvr-wddff/db1-4/single_hybrid_5_ahead_lag12_mi_ts0.1/optimal_model_results.csv')
z_m_7 = pd.read_csv(root_path+'/Zhangjiashan_modwt/projects/esvr-wddff/db1-4/single_hybrid_7_ahead_lag12_mi_ts0.1/optimal_model_results.csv')

records={
    'h_arma': h_arma['test_y'],
    'h_svr_1' : h_svr_1['test_y'][0:120].values,
    'h_svr_3' : h_svr_3['test_y'][0:120].values,
    'h_svr_5' : h_svr_5['test_y'][0:120].values,
    'h_svr_7' : h_svr_7['test_y'][0:120].values,
    'h_dnn_1' : h_dnn_1['test_y'][0:120].values,
    'h_dnn_3' : h_dnn_3['test_y'][0:120].values,
    'h_dnn_5' : h_dnn_5['test_y'][0:120].values,
    'h_dnn_7' : h_dnn_7['test_y'][0:120].values,
    'h_lstm_1' : h_lstm_1['test_y'][0:120].values,
    'h_lstm_3' : h_lstm_3['test_y'][0:120].values,
    'h_lstm_5' : h_lstm_5['test_y'][0:120].values,
    'h_lstm_7' : h_lstm_7['test_y'][0:120].values,
    'h_m_1' : h_m_1['test_y'][0:120].values,
    'h_d_1' : h_d_1['test_y'][0:120].values,
    'h_e_1' : h_e_1['test_y'][0:120].values,
    'h_s_1' : h_s_1['test_y'][0:120].values,
    'h_v_1' : h_v_1['test_y'][0:120].values,
    'x_arma': x_arma['test_y'],
    'x_svr_1' : x_svr_1['test_y'][0:120].values,
    'x_svr_3' : x_svr_3['test_y'][0:120].values,
    'x_svr_5' : x_svr_5['test_y'][0:120].values,
    'x_svr_7' : x_svr_7['test_y'][0:120].values,
    'x_dnn_1' : x_dnn_1['test_y'][0:120].values,
    'x_dnn_3' : x_dnn_3['test_y'][0:120].values,
    'x_dnn_5' : x_dnn_5['test_y'][0:120].values,
    'x_dnn_7' : x_dnn_7['test_y'][0:120].values,
    'x_lstm_1' : x_lstm_1['test_y'][0:120].values,
    'x_lstm_3' : x_lstm_3['test_y'][0:120].values,
    'x_lstm_5' : x_lstm_5['test_y'][0:120].values,
    'x_lstm_7' : x_lstm_7['test_y'][0:120].values,
    'x_m_1' : x_m_1['test_y'][0:120].values,
    'x_d_1' : x_d_1['test_y'][0:120].values,
    'x_e_1' : x_e_1['test_y'][0:120].values,
    'x_s_1' : x_s_1['test_y'][0:120].values,
    'x_v_1' : x_v_1['test_y'][0:120].values,
    'z_arma': z_arma['test_y'],
    'z_svr_1' : z_svr_1['test_y'][0:120].values,
    'z_svr_3' : z_svr_3['test_y'][0:120].values,
    'z_svr_5' : z_svr_5['test_y'][0:120].values,
    'z_svr_7' : z_svr_7['test_y'][0:120].values,
    'z_dnn_1' : z_dnn_1['test_y'][0:120].values,
    'z_dnn_3' : z_dnn_3['test_y'][0:120].values,
    'z_dnn_5' : z_dnn_5['test_y'][0:120].values,
    'z_dnn_7' : z_dnn_7['test_y'][0:120].values,
    'z_lstm_1' : z_lstm_1['test_y'][0:120].values,
    'z_lstm_3' : z_lstm_3['test_y'][0:120].values,
    'z_lstm_5' : z_lstm_5['test_y'][0:120].values,
    'z_lstm_7' : z_lstm_7['test_y'][0:120].values,
    'z_m_1' : z_m_1['test_y'][0:120].values,
    'z_d_1' : z_d_1['test_y'][0:120].values,
    'z_e_1' : z_e_1['test_y'][0:120].values,
    'z_s_1' : z_s_1['test_y'][0:120].values,
    'z_v_1' : z_v_1['test_y'][0:120].values,
    'h_m_3' : h_m_3['test_y'][0:120].values,
    'h_d_3' : h_d_3['test_y'][0:120].values,
    'h_e_3' : h_e_3['test_y'][0:120].values,
    'h_s_3' : h_s_3['test_y'][0:120].values,
    'h_v_3' : h_v_3['test_y'][0:120].values,
    'x_m_3' : x_m_3['test_y'][0:120].values,
    'x_d_3' : x_d_3['test_y'][0:120].values,
    'x_e_3' : x_e_3['test_y'][0:120].values,
    'x_s_3' : x_s_3['test_y'][0:120].values,
    'x_v_3' : x_v_3['test_y'][0:120].values,
    'z_m_3' : z_m_3['test_y'][0:120].values,
    'z_d_3' : z_d_3['test_y'][0:120].values,
    'z_e_3' : z_e_3['test_y'][0:120].values,
    'z_s_3' : z_s_3['test_y'][0:120].values,
    'z_v_3' : z_v_3['test_y'][0:120].values,
    'h_m_5' : h_m_5['test_y'][0:120].values,
    'h_d_5' : h_d_5['test_y'][0:120].values,
    'h_e_5' : h_e_5['test_y'][0:120].values,
    'h_s_5' : h_s_5['test_y'][0:120].values,
    'h_v_5' : h_v_5['test_y'][0:120].values,
    'x_m_5' : x_m_5['test_y'][0:120].values,
    'x_d_5' : x_d_5['test_y'][0:120].values,
    'x_e_5' : x_e_5['test_y'][0:120].values,
    'x_s_5' : x_s_5['test_y'][0:120].values,
    'x_v_5' : x_v_5['test_y'][0:120].values,
    'z_m_5' : z_m_5['test_y'][0:120].values,
    'z_d_5' : z_d_5['test_y'][0:120].values,
    'z_e_5' : z_e_5['test_y'][0:120].values,
    'z_s_5' : z_s_5['test_y'][0:120].values,
    'z_v_5' : z_v_5['test_y'][0:120].values,
    'h_m_7' : h_m_7['test_y'][0:120].values,
    'h_d_7' : h_d_7['test_y'][0:120].values,
    'h_e_7' : h_e_7['test_y'][0:120].values,
    'h_s_7' : h_s_7['test_y'][0:120].values,
    'h_v_7' : h_v_7['test_y'][0:120].values,
    'x_m_7' : x_m_7['test_y'][0:120].values,
    'x_d_7' : x_d_7['test_y'][0:120].values,
    'x_e_7' : x_e_7['test_y'][0:120].values,
    'x_s_7' : x_s_7['test_y'][0:120].values,
    'x_v_7' : x_v_7['test_y'][0:120].values,
    'z_m_7' : z_m_7['test_y'][0:120].values,
    'z_d_7' : z_d_7['test_y'][0:120].values,
    'z_e_7' : z_e_7['test_y'][0:120].values,
    'z_s_7' : z_s_7['test_y'][0:120].values,
    'z_v_7' : z_v_7['test_y'][0:120].values,
}


preds={
    'h_arma': h_arma['test_pred'],
    'h_svr_1' : h_svr_1['test_pred'][0:120].values,
    'h_svr_3' : h_svr_3['test_pred'][0:120].values,
    'h_svr_5' : h_svr_5['test_pred'][0:120].values,
    'h_svr_7' : h_svr_7['test_pred'][0:120].values,
    'h_dnn_1' : h_dnn_1['test_pred'][0:120].values,
    'h_dnn_3' : h_dnn_3['test_pred'][0:120].values,
    'h_dnn_5' : h_dnn_5['test_pred'][0:120].values,
    'h_dnn_7' : h_dnn_7['test_pred'][0:120].values,
    'h_lstm_1' : h_lstm_1['test_pred'][0:120].values,
    'h_lstm_3' : h_lstm_3['test_pred'][0:120].values,
    'h_lstm_5' : h_lstm_5['test_pred'][0:120].values,
    'h_lstm_7' : h_lstm_7['test_pred'][0:120].values,
    'h_m_1' : h_m_1['test_pred'][0:120].values,
    'h_d_1' : h_d_1['test_pred'][0:120].values,
    'h_e_1' : h_e_1['test_pred'][0:120].values,
    'h_s_1' : h_s_1['test_pred'][0:120].values,
    'h_v_1' : h_v_1['test_pred'][0:120].values,
    'x_arma': x_arma['test_pred'],
    'x_svr_1' : x_svr_1['test_pred'][0:120].values,
    'x_svr_3' : x_svr_3['test_pred'][0:120].values,
    'x_svr_5' : x_svr_5['test_pred'][0:120].values,
    'x_svr_7' : x_svr_7['test_pred'][0:120].values,
    'x_dnn_1' : x_dnn_1['test_pred'][0:120].values,
    'x_dnn_3' : x_dnn_3['test_pred'][0:120].values,
    'x_dnn_5' : x_dnn_5['test_pred'][0:120].values,
    'x_dnn_7' : x_dnn_7['test_pred'][0:120].values,
    'x_lstm_1' : x_lstm_1['test_pred'][0:120].values,
    'x_lstm_3' : x_lstm_3['test_pred'][0:120].values,
    'x_lstm_5' : x_lstm_5['test_pred'][0:120].values,
    'x_lstm_7' : x_lstm_7['test_pred'][0:120].values,
    'x_m_1' : x_m_1['test_pred'][0:120].values,
    'x_d_1' : x_d_1['test_pred'][0:120].values,
    'x_e_1' : x_e_1['test_pred'][0:120].values,
    'x_s_1' : x_s_1['test_pred'][0:120].values,
    'x_v_1' : x_v_1['test_pred'][0:120].values,
    'z_arma': z_arma['test_pred'],
    'z_svr_1' : z_svr_1['test_pred'][0:120].values,
    'z_svr_3' : z_svr_3['test_pred'][0:120].values,
    'z_svr_5' : z_svr_5['test_pred'][0:120].values,
    'z_svr_7' : z_svr_7['test_pred'][0:120].values,
    'z_dnn_1' : z_dnn_1['test_pred'][0:120].values,
    'z_dnn_3' : z_dnn_3['test_pred'][0:120].values,
    'z_dnn_5' : z_dnn_5['test_pred'][0:120].values,
    'z_dnn_7' : z_dnn_7['test_pred'][0:120].values,
    'z_lstm_1' : z_lstm_1['test_pred'][0:120].values,
    'z_lstm_3' : z_lstm_3['test_pred'][0:120].values,
    'z_lstm_5' : z_lstm_5['test_pred'][0:120].values,
    'z_lstm_7' : z_lstm_7['test_pred'][0:120].values,
    'z_m_1' : z_m_1['test_pred'][0:120].values,
    'z_d_1' : z_d_1['test_pred'][0:120].values,
    'z_e_1' : z_e_1['test_pred'][0:120].values,
    'z_s_1' : z_s_1['test_pred'][0:120].values,
    'z_v_1' : z_v_1['test_pred'][0:120].values,
    'h_m_3' : h_m_3['test_pred'][0:120].values,
    'h_d_3' : h_d_3['test_pred'][0:120].values,
    'h_e_3' : h_e_3['test_pred'][0:120].values,
    'h_s_3' : h_s_3['test_pred'][0:120].values,
    'h_v_3' : h_v_3['test_pred'][0:120].values,
    'x_m_3' : x_m_3['test_pred'][0:120].values,
    'x_d_3' : x_d_3['test_pred'][0:120].values,
    'x_e_3' : x_e_3['test_pred'][0:120].values,
    'x_s_3' : x_s_3['test_pred'][0:120].values,
    'x_v_3' : x_v_3['test_pred'][0:120].values,
    'z_m_3' : z_m_3['test_pred'][0:120].values,
    'z_d_3' : z_d_3['test_pred'][0:120].values,
    'z_e_3' : z_e_3['test_pred'][0:120].values,
    'z_s_3' : z_s_3['test_pred'][0:120].values,
    'z_v_3' : z_v_3['test_pred'][0:120].values,
    'h_m_5' : h_m_5['test_pred'][0:120].values,
    'h_d_5' : h_d_5['test_pred'][0:120].values,
    'h_e_5' : h_e_5['test_pred'][0:120].values,
    'h_s_5' : h_s_5['test_pred'][0:120].values,
    'h_v_5' : h_v_5['test_pred'][0:120].values,
    'x_m_5' : x_m_5['test_pred'][0:120].values,
    'x_d_5' : x_d_5['test_pred'][0:120].values,
    'x_e_5' : x_e_5['test_pred'][0:120].values,
    'x_s_5' : x_s_5['test_pred'][0:120].values,
    'x_v_5' : x_v_5['test_pred'][0:120].values,
    'z_m_5' : z_m_5['test_pred'][0:120].values,
    'z_d_5' : z_d_5['test_pred'][0:120].values,
    'z_e_5' : z_e_5['test_pred'][0:120].values,
    'z_s_5' : z_s_5['test_pred'][0:120].values,
    'z_v_5' : z_v_5['test_pred'][0:120].values,
    'h_m_7' : h_m_7['test_pred'][0:120].values,
    'h_d_7' : h_d_7['test_pred'][0:120].values,
    'h_e_7' : h_e_7['test_pred'][0:120].values,
    'h_s_7' : h_s_7['test_pred'][0:120].values,
    'h_v_7' : h_v_7['test_pred'][0:120].values,
    'x_m_7' : x_m_7['test_pred'][0:120].values,
    'x_d_7' : x_d_7['test_pred'][0:120].values,
    'x_e_7' : x_e_7['test_pred'][0:120].values,
    'x_s_7' : x_s_7['test_pred'][0:120].values,
    'x_v_7' : x_v_7['test_pred'][0:120].values,
    'z_m_7' : z_m_7['test_pred'][0:120].values,
    'z_d_7' : z_d_7['test_pred'][0:120].values,
    'z_e_7' : z_e_7['test_pred'][0:120].values,
    'z_s_7' : z_s_7['test_pred'][0:120].values,
    'z_v_7' : z_v_7['test_pred'][0:120].values,
}
xx,linear_list,xymin,xymax = compute_multi_linear_fit(records,preds)
h_records={
    'h_arma': h_arma['test_y'],
    'h_svr_1' : h_svr_1['test_y'][0:120].values,
    'h_dnn_1' : h_dnn_1['test_y'][0:120].values,
    'h_lstm_1' : h_lstm_1['test_y'][0:120].values,
    'h_m_1' : h_m_1['test_y'][0:120].values,
    'h_d_1' : h_d_1['test_y'][0:120].values,
    'h_e_1' : h_e_1['test_y'][0:120].values,
    'h_s_1' : h_s_1['test_y'][0:120].values,
    'h_v_1' : h_v_1['test_y'][0:120].values,
    'h_svr_3' : h_svr_3['test_y'][0:120].values,
    'h_dnn_3' : h_dnn_3['test_y'][0:120].values,
    'h_lstm_3' : h_lstm_3['test_y'][0:120].values,
    'h_m_3' : h_m_3['test_y'][0:120].values,
    'h_d_3' : h_d_3['test_y'][0:120].values,
    'h_e_3' : h_e_3['test_y'][0:120].values,
    'h_s_3' : h_s_3['test_y'][0:120].values,
    'h_v_3' : h_v_3['test_y'][0:120].values,
    'h_svr_5' : h_svr_5['test_y'][0:120].values,
    'h_dnn_5' : h_dnn_5['test_y'][0:120].values,
    'h_lstm_5' : h_lstm_5['test_y'][0:120].values,
    'h_m_5' : h_m_5['test_y'][0:120].values,
    'h_d_5' : h_d_5['test_y'][0:120].values,
    'h_e_5' : h_e_5['test_y'][0:120].values,
    'h_s_5' : h_s_5['test_y'][0:120].values,
    'h_v_5' : h_v_5['test_y'][0:120].values,
    'h_svr_7' : h_svr_7['test_y'][0:120].values,
    'h_dnn_7' : h_dnn_7['test_y'][0:120].values,
    'h_lstm_7' : h_lstm_7['test_y'][0:120].values,
    'h_m_7' : h_m_7['test_y'][0:120].values,
    'h_d_7' : h_d_7['test_y'][0:120].values,
    'h_e_7' : h_e_7['test_y'][0:120].values,
    'h_s_7' : h_s_7['test_y'][0:120].values,
    'h_v_7' : h_v_7['test_y'][0:120].values,
}

h_preds={
    'h_arma': h_arma['test_pred'],
    'h_svr_1' : h_svr_1['test_pred'][0:120].values,
    'h_dnn_1' : h_dnn_1['test_pred'][0:120].values,
    'h_lstm_1' : h_lstm_1['test_pred'][0:120].values,
    'h_m_1' : h_m_1['test_pred'][0:120].values,
    'h_d_1' : h_d_1['test_pred'][0:120].values,
    'h_e_1' : h_e_1['test_pred'][0:120].values,
    'h_s_1' : h_s_1['test_pred'][0:120].values,
    'h_v_1' : h_v_1['test_pred'][0:120].values,
    'h_svr_3' : h_svr_3['test_pred'][0:120].values,
    'h_dnn_3' : h_dnn_3['test_pred'][0:120].values,
    'h_lstm_3' : h_lstm_3['test_pred'][0:120].values,
    'h_m_3' : h_m_3['test_pred'][0:120].values,
    'h_d_3' : h_d_3['test_pred'][0:120].values,
    'h_e_3' : h_e_3['test_pred'][0:120].values,
    'h_s_3' : h_s_3['test_pred'][0:120].values,
    'h_v_3' : h_v_3['test_pred'][0:120].values,
    'h_svr_5' : h_svr_5['test_pred'][0:120].values,
    'h_dnn_5' : h_dnn_5['test_pred'][0:120].values,
    'h_lstm_5' : h_lstm_5['test_pred'][0:120].values,
    'h_m_5' : h_m_5['test_pred'][0:120].values,
    'h_d_5' : h_d_5['test_pred'][0:120].values,
    'h_e_5' : h_e_5['test_pred'][0:120].values,
    'h_s_5' : h_s_5['test_pred'][0:120].values,
    'h_v_5' : h_v_5['test_pred'][0:120].values,
    'h_svr_7' : h_svr_7['test_pred'][0:120].values,
    'h_dnn_7' : h_dnn_7['test_pred'][0:120].values,
    'h_lstm_7' : h_lstm_7['test_pred'][0:120].values,
    'h_m_7' : h_m_7['test_pred'][0:120].values,
    'h_d_7' : h_d_7['test_pred'][0:120].values,
    'h_e_7' : h_e_7['test_pred'][0:120].values,
    'h_s_7' : h_s_7['test_pred'][0:120].values,
    'h_v_7' : h_v_7['test_pred'][0:120].values,
}
h_labels=[
    ['h_arma','h_svr_1','h_dnn_1','h_lstm_1','h_m_1','h_e_1','h_s_1','h_d_1','h_v_1',],
    ['h_svr_3','h_dnn_3','h_lstm_3','h_m_3','h_e_3','h_s_3','h_d_3','h_v_3',],
    ['h_svr_5','h_dnn_5','h_lstm_5','h_m_5','h_e_5','h_s_5','h_d_5','h_v_5',],
    ['h_svr_7','h_dnn_7','h_lstm_7','h_m_7','h_e_7','h_s_7','h_d_7','h_v_7',],
]
h_xx,h_linear_list,h_xymin,h_xymax = compute_multi_linear_fit(h_records,h_preds)
markers0=['<','d','v','p','s','*','+','^','o']
markers=['d','v','p','s','*','+','^','o']
colors0=['tab:red','tab:brown','tab:green','tab:olive','tab:orange','tab:cyan','tab:purple','tab:gray','tab:blue']
colors=['tab:brown','tab:green','tab:olive','tab:orange','tab:cyan','tab:purple','tab:gray','tab:blue']
zorders0=[0,1,2,3,4,5,6,7,8]
zorders=[1,2,3,5,4,6,7,8]
# alpha0 = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
# alpha = [0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
alpha0 = [0.52,0.58,0.64,0.7,0.76,0.82,0.88,0.94,1]
alpha = [0.58,0.64,0.7,0.76,0.82,0.88,0.94,1]
models0=['ARMA','SVR','BPNN','LSTM','WDDFF (BCMODWT-SVR)','TSDP (EEMD-SVR)','TSDP (SSA-SVR)','TSDP (DWT-SVR)','TSDP (VMD-SVR)']
models=['SVR','BPNN','LSTM','WDDFF (BCMODWT-SVR)','TSDP (EEMD-SVR)','TSDP (SSA-SVR)','TSDP (DWT-SVR)','TSDP (VMD-SVR)']
titles=['(a) 1-month ahead','(b) 3-month ahead','(c) 5-month ahead','(d) 7-month ahead']
plt.figure(figsize=(7.48,8.05))
for i in range(len(h_labels)):
    plt.subplot(2,2,i+1,aspect='equal')
    plt.text(25,1,titles[i])
    plt.plot([h_xymin,h_xymax], [h_xymin,h_xymax], '-', color='black', label='Ideal fit',)
    plt.xlim([h_xymin,h_xymax])
    plt.ylim([h_xymin,h_xymax])
    if i==len(h_labels)-1 or i==len(h_labels)-2:
        plt.xlabel('Predictions (' + r'$10^8m^3$' +')')
    else:
        plt.xticks([])
    if i in [0,2]:
        plt.ylabel('Records (' + r'$10^8m^3$' + ')', )
    else:
        plt.yticks([])
    for j in range(len(h_labels[i])):
        if i==0:
            plt.scatter(preds[h_labels[i][j]], records[h_labels[i][j]],c=colors0[j],label=models0[j],marker=markers0[j],zorder=zorders0[j],alpha=alpha0[j])
            plt.plot(xx, linear_list[h_labels[i][j]], '--',c=colors0[j],label=models0[j])
        else:
            plt.scatter(preds[h_labels[i][j]], records[h_labels[i][j]],c=colors[j],label=models[j],marker=markers[j],zorder=zorders[j],alpha=alpha[j])
            plt.plot(xx, linear_list[h_labels[i][j]], '--',c=colors[j],label=models[j])
    if i==0:
        plt.legend(
                    loc='upper left',
                    # bbox_to_anchor=(0.08,1.01, 1,0.101),
                    bbox_to_anchor=(0.001,1.15),
                    ncol=6,
                    shadow=False,
                    frameon=True,
                    )
plt.subplots_adjust(left=0.06, bottom=0.05, right=0.99,top=0.94, hspace=0.02, wspace=0.02)
plt.savefig(graphs_path+'Scatter plots for TSDP and WDDFF at Huaxian.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'Scatter plots for TSDP and WDDFF at Huaxian.tif',format='TIFF',dpi=500)
plt.savefig(graphs_path+'Scatter plots for TSDP and WDDFF at Huaxian.pdf',format='PDF',dpi=1200)


models0=['ARMA','SVR','BPNN','LSTM','BCMODWT-SVR','EEMD-SVR','SSA-SVR','DWT-SVR','VMD-SVR']
plt.figure(figsize=(7.48,6.6))
for i in range(len(h_labels)):
    plt.subplot(2,2,i+1,aspect='equal')
    plt.text(24,1,titles[i])
    plt.plot([h_xymin,h_xymax], [h_xymin,h_xymax], '-', color='black', label='Ideal fit',)
    plt.xlim([h_xymin,h_xymax])
    plt.ylim([h_xymin,h_xymax])
    if i==len(h_labels)-1 or i==len(h_labels)-2:
        plt.xlabel('Predictions (' + r'$10^8m^3$' +')')
    else:
        plt.xticks([])
    if i in [0,2]:
        plt.ylabel('Records (' + r'$10^8m^3$' + ')', )
    else:
        plt.yticks([])
    for j in range(len(h_labels[i])):
        if i==0:
            plt.scatter(preds[h_labels[i][j]], records[h_labels[i][j]],c=colors0[j],label=models0[j],marker=markers0[j],zorder=zorders0[j],alpha=alpha0[j])
            plt.plot(xx, linear_list[h_labels[i][j]], '--',c=colors0[j],label=models0[j])
        else:
            plt.scatter(preds[h_labels[i][j]], records[h_labels[i][j]],c=colors[j],label=models[j],marker=markers[j],zorder=zorders[j],alpha=alpha[j])
            plt.plot(xx, linear_list[h_labels[i][j]], '--',c=colors[j],label=models[j])
    if i==0:
        plt.legend(
                    loc='upper left',
                    # bbox_to_anchor=(2.03,0.8, 1,0.5),
                    bbox_to_anchor=(2.03,1.012),
                    ncol=1,
                    shadow=False,
                    frameon=True,
                    labelspacing=2.98,
                    )
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.86,top=0.99, hspace=0.02, wspace=0.02)
plt.savefig(graphs_path+'Scatter plots for TSDP and WDDFF at Huaxian.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'Scatter plots for TSDP and WDDFF at Huaxian.tif',format='TIFF',dpi=500)
plt.savefig(graphs_path+'Scatter plots for TSDP and WDDFF at Huaxian.pdf',format='PDF',dpi=1200)
plt.show()

data=[
    [h_arma,h_svr_1,h_m_1,h_e_1,h_s_1,h_d_1,h_v_1],
    [x_arma,x_svr_1,x_m_1,x_e_1,x_s_1,x_d_1,x_v_1],
    [z_arma,z_svr_1,z_m_1,z_e_1,z_s_1,z_d_1,z_v_1],
    [h_svr_3,h_m_3,h_e_3,h_s_3,h_d_3,h_v_3],
    [x_svr_3,x_m_3,x_e_3,x_s_3,x_d_3,x_v_3],
    [z_svr_3,z_m_3,z_e_3,z_s_3,z_d_3,z_v_3],
    [h_svr_5,h_m_5,h_e_5,h_s_5,h_d_5,h_v_5],
    [x_svr_5,x_m_5,x_e_5,x_s_5,x_d_5,x_v_5],
    [z_svr_5,z_m_5,z_e_5,z_s_5,z_d_5,z_v_5],
    [h_svr_7,h_m_7,h_e_7,h_s_7,h_d_7,h_v_7],
    [x_svr_7,x_m_7,x_e_7,x_s_7,x_d_7,x_v_7],
    [z_svr_7,z_m_7,z_e_7,z_s_7,z_d_7,z_v_7],
]

h_data_rds=[
    [h_arma['test_y'],h_svr_1['test_y'][0:120].values,h_m_1['test_y'][0:120].values,h_e_1['test_y'][0:120].values,h_s_1['test_y'][0:120].values,h_d_1['test_y'][0:120].values,h_v_1['test_y'][0:120].values],
    [h_m_3['test_y'][0:120].values,h_e_3['test_y'][0:120].values,h_s_3['test_y'][0:120].values,h_d_3['test_y'][0:120].values,h_v_3['test_y'][0:120].values],
    [h_m_5['test_y'][0:120].values,h_e_5['test_y'][0:120].values,h_s_5['test_y'][0:120].values,h_d_5['test_y'][0:120].values,h_v_5['test_y'][0:120].values],
    [h_m_7['test_y'][0:120].values,h_e_7['test_y'][0:120].values,h_s_7['test_y'][0:120].values,h_d_7['test_y'][0:120].values,h_v_7['test_y'][0:120].values],
]

h_data_pds=[
    [h_arma['test_pred'],h_svr_1['test_pred'][0:120].values,h_m_1['test_pred'][0:120].values,h_e_1['test_pred'][0:120].values,h_s_1['test_pred'][0:120].values,h_d_1['test_pred'][0:120].values,h_v_1['test_pred'][0:120].values],
    [h_m_3['test_pred'][0:120].values,h_e_3['test_pred'][0:120].values,h_s_3['test_pred'][0:120].values,h_d_3['test_pred'][0:120].values,h_v_3['test_pred'][0:120].values],
    [h_m_5['test_pred'][0:120].values,h_e_5['test_pred'][0:120].values,h_s_5['test_pred'][0:120].values,h_d_5['test_pred'][0:120].values,h_v_5['test_pred'][0:120].values],
    [h_m_7['test_pred'][0:120].values,h_e_7['test_pred'][0:120].values,h_s_7['test_pred'][0:120].values,h_d_7['test_pred'][0:120].values,h_v_7['test_pred'][0:120].values],
]



# models=['ARMA','SVR','WDDFF(MODWT-SVR)','TSDP(EEMD-SVR)','TSDP(SSA-SVR)','TSDP(DWT-SVR)','TSDP(VMD-SVR)']

# idx = [
#     [1,2,3],
#     [4,5,6],
#     [7,8,9],
#     [10,11,12],
#     [13,14,15],
#     [16,17,18],
#     [19,20,21],
# ]


# fig_idx=['(a)','(b)','(c)']
# plt.figure(figsize=(7.48,7.48))
# for i in range(len(data)):
#     for j in range(len(data[i])):
#         plt.subplot(7,3,idx[i][j])
#         plt.plot(data[i][j]['test_y'],label='Records',c='tab:blue',lw=0.8)
#         plt.plot(data[i][j]['test_pred'],'--',label=models[i],c='tab:red',lw=0.8)
#         plt.ylabel("Flow(" + r"$10^8m^3$" + ")")
#         if i==len(data)-1:
#             plt.xlabel('Time(month)\n'+fig_idx[j])
#         plt.legend(ncol=2)
# plt.tight_layout()
# plt.savefig(graphs_path+'two_stage_predictions.eps',format='EPS',dpi=2000)
# plt.savefig(graphs_path+'two_stage_predictions.tif',format='TIFF',dpi=1200)
# # plt.show()


# records_data=[
#     [h_arma['test_y'],h_esvr['test_y'],h_modwt_esvr['test_y'],h_eemd_esvr['test_y'],h_ssa_esvr['test_y'],h_dwt_esvr['test_y'],h_vmd_esvr['test_y']],
#     [x_arma['test_y'],x_esvr['test_y'],x_modwt_esvr['test_y'],x_eemd_esvr['test_y'],x_ssa_esvr['test_y'],x_dwt_esvr['test_y'],x_vmd_esvr['test_y']],
#     [z_arma['test_y'],z_esvr['test_y'],z_modwt_esvr['test_y'],z_eemd_esvr['test_y'],z_ssa_esvr['test_y'],z_dwt_esvr['test_y'],z_vmd_esvr['test_y']],
# ]
# preds_data=[
#     [h_arma['test_pred'],h_esvr['test_pred'],h_modwt_esvr['test_pred'],h_eemd_esvr['test_pred'],h_ssa_esvr['test_pred'],h_dwt_esvr['test_pred'],h_vmd_esvr['test_pred']],
#     [x_arma['test_pred'],x_esvr['test_pred'],x_modwt_esvr['test_pred'],x_eemd_esvr['test_pred'],x_ssa_esvr['test_pred'],x_dwt_esvr['test_pred'],x_vmd_esvr['test_pred']],
#     [z_arma['test_pred'],z_esvr['test_pred'],z_modwt_esvr['test_pred'],z_eemd_esvr['test_pred'],z_ssa_esvr['test_pred'],z_dwt_esvr['test_pred'],z_vmd_esvr['test_pred']],
# ]
# markers=['<','v','s','*','+','d','o']
# colors=['r','g','teal','cyan','purple','gold','blue']
# zorders=[0,1,2,3,4,5,6]
# plt.figure(figsize=(7.48,3.28))
# for i in range(len(records_data)):
#     plt.subplot(1,3,i+1,aspect='equal')
#     records_list=records_data[i]
#     predictions_list=preds_data[i]
#     xx,linear_list,xymin,xymax=compute_multi_linear_fit(
#         records_list=records_list,
#         predictions_list=predictions_list,
#     )
#     plt.xlabel('Predictions(' + r'$10^8m^3$' +')\n'+fig_idx[i], )
#     if i==0:
#         plt.ylabel('Records(' + r'$10^8m^3$' + ')', )
#     for j in range(len(predictions_list)):
#         # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
#         plt.scatter(predictions_list[j], records_list[j],label=models[j],marker=markers[j],zorder=zorders[j])
#         plt.plot(xx, linear_list[j], '--', label=models[j],linewidth=1.0,zorder=zorders[j])
#     plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
#     plt.xlim([xymin,xymax])
#     plt.ylim([xymin,xymax])
#     if i==1:
#         plt.legend(
#                     loc='upper center',
#                     # bbox_to_anchor=(0.08,1.01, 1,0.101),
#                     bbox_to_anchor=(0.5,1.25),
#                     ncol=5,
#                     shadow=False,
#                     frameon=True,
#                     )
# # plt.tight_layout()
# plt.subplots_adjust(left=0.06, bottom=0.08, right=0.99,top=0.94, hspace=0.2, wspace=0.15)
# plt.savefig(graphs_path+'Scatter plots for TSDP and WDDFF.eps',format='EPS',dpi=2000)
# plt.savefig(graphs_path+'Scatter plots for TSDP and WDDFF.tif',format='TIFF',dpi=1200)
# plt.savefig(graphs_path+'Scatter plots for TSDP and WDDFF.pdf',format='PDF',dpi=1200)
# plt.show()