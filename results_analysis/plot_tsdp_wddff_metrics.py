import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.size'] = 6
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/graphs/'
results_path = root_path+'/results_analysis/results/'
print("root path:{}".format(root_path))
sys.path.append(root_path)
from tools.results_reader import read_two_stage, read_pure_esvr,read_pure_arma
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






def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height, 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height/2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def autolabels(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height, 2)
        ax.text(
            x=rect.get_x() + rect.get_width() / 2,
            y=height,
            s='{}'.format(height),
            rotation=90,
            ha='center', va='bottom',
        )
nse_arma_data = [h_arma['test_nse'],x_arma['test_nse'],z_arma['test_nse']]
nse_svr_data1 = [h_svr_1['test_nse'][0],x_svr_1['test_nse'][0],z_svr_1['test_nse'][0]]
nse_svr_data3 = [h_svr_3['test_nse'][0],x_svr_3['test_nse'][0],z_svr_3['test_nse'][0]]
nse_svr_data5 = [h_svr_5['test_nse'][0],x_svr_5['test_nse'][0],z_svr_5['test_nse'][0]]
nse_svr_data7 = [h_svr_7['test_nse'][0],x_svr_7['test_nse'][0],z_svr_7['test_nse'][0]]

nse_dnn_data1 = [h_dnn_1['test_nse'][0],x_dnn_1['test_nse'][0],z_dnn_1['test_nse'][0]]
nse_dnn_data3 = [h_dnn_3['test_nse'][0],x_dnn_3['test_nse'][0],z_dnn_3['test_nse'][0]]
nse_dnn_data5 = [h_dnn_5['test_nse'][0],x_dnn_5['test_nse'][0],z_dnn_5['test_nse'][0]]
nse_dnn_data7 = [h_dnn_7['test_nse'][0],x_dnn_7['test_nse'][0],z_dnn_7['test_nse'][0]]

nse_lstm_data1 = [h_lstm_1['test_nse'][0],x_lstm_1['test_nse'][0],z_lstm_1['test_nse'][0]]
nse_lstm_data3 = [h_lstm_3['test_nse'][0],x_lstm_3['test_nse'][0],z_lstm_3['test_nse'][0]]
nse_lstm_data5 = [h_lstm_5['test_nse'][0],x_lstm_5['test_nse'][0],z_lstm_5['test_nse'][0]]
nse_lstm_data7 = [h_lstm_7['test_nse'][0],x_lstm_7['test_nse'][0],z_lstm_7['test_nse'][0]]

nse_eemd_data1=[h_e_1['test_nse'][0],x_e_1['test_nse'][0],z_e_1['test_nse'][0]]
nse_eemd_data3=[h_e_3['test_nse'][0],x_e_3['test_nse'][0],z_e_3['test_nse'][0]]
nse_eemd_data5=[h_e_5['test_nse'][0],x_e_5['test_nse'][0],z_e_5['test_nse'][0]]
nse_eemd_data7=[h_e_7['test_nse'][0],x_e_7['test_nse'][0],z_e_7['test_nse'][0]]

nse_ssa_data1=[h_s_1['test_nse'][0],x_s_1['test_nse'][0],z_s_1['test_nse'][0]]
nse_ssa_data3=[h_s_3['test_nse'][0],x_s_3['test_nse'][0],z_s_3['test_nse'][0]]
nse_ssa_data5=[h_s_5['test_nse'][0],x_s_5['test_nse'][0],z_s_5['test_nse'][0]]
nse_ssa_data7=[h_s_7['test_nse'][0],x_s_7['test_nse'][0],z_s_7['test_nse'][0]]

nse_vmd_data1=[h_v_1['test_nse'][0],x_v_1['test_nse'][0],z_v_1['test_nse'][0]]
nse_vmd_data3=[h_v_3['test_nse'][0],x_v_3['test_nse'][0],z_v_3['test_nse'][0]]
nse_vmd_data5=[h_v_5['test_nse'][0],x_v_5['test_nse'][0],z_v_5['test_nse'][0]]
nse_vmd_data7=[h_v_7['test_nse'][0],x_v_7['test_nse'][0],z_v_7['test_nse'][0]]

nse_dwt_data1=[h_d_1['test_nse'][0],x_d_1['test_nse'][0],z_d_1['test_nse'][0]]
nse_dwt_data3=[h_d_3['test_nse'][0],x_d_3['test_nse'][0],z_d_3['test_nse'][0]]
nse_dwt_data5=[h_d_5['test_nse'][0],x_d_5['test_nse'][0],z_d_5['test_nse'][0]]
nse_dwt_data7=[h_d_7['test_nse'][0],x_d_7['test_nse'][0],z_d_7['test_nse'][0]]

nse_modwt_data1=[h_m_1['test_nse'][0],x_m_1['test_nse'][0],z_m_1['test_nse'][0]]
nse_modwt_data3=[h_m_3['test_nse'][0],x_m_3['test_nse'][0],z_m_3['test_nse'][0]]
nse_modwt_data5=[h_m_5['test_nse'][0],x_m_5['test_nse'][0],z_m_5['test_nse'][0]]
nse_modwt_data7=[h_m_7['test_nse'][0],x_m_7['test_nse'][0],z_m_7['test_nse'][0]]

nse_data=[
    nse_arma_data,
    nse_svr_data1,
    nse_svr_data3,
    nse_svr_data5,
    nse_svr_data7,
    nse_dnn_data1,
    nse_dnn_data3,
    nse_dnn_data5,
    nse_dnn_data7,
    nse_lstm_data1,
    nse_lstm_data3,
    nse_lstm_data5,
    nse_lstm_data7,
    nse_eemd_data1,
    nse_eemd_data3,
    nse_eemd_data5,
    nse_eemd_data7,
    nse_ssa_data1,
    nse_ssa_data3,
    nse_ssa_data5,
    nse_ssa_data7,
    nse_vmd_data1,
    nse_vmd_data3,
    nse_vmd_data5,
    nse_vmd_data7,
    nse_dwt_data1,
    nse_dwt_data3,
    nse_dwt_data5,
    nse_dwt_data7,
    nse_modwt_data1,
    nse_modwt_data3,
    nse_modwt_data5,
    nse_modwt_data7,
]

arima_mean_nse = sum(nse_arma_data)/len(nse_arma_data)

svr_mean_nse = [
    sum(nse_svr_data1)/len(nse_svr_data1),
    sum(nse_svr_data3)/len(nse_svr_data3),
    sum(nse_svr_data5)/len(nse_svr_data5),
    sum(nse_svr_data7)/len(nse_svr_data7),
]
eemd_mean_nse=[
    sum(nse_eemd_data1)/len(nse_eemd_data1),
    sum(nse_eemd_data3)/len(nse_eemd_data3),
    sum(nse_eemd_data5)/len(nse_eemd_data5),
    sum(nse_eemd_data7)/len(nse_eemd_data7),
]
ssa_mean_nse=[
    sum(nse_ssa_data1)/len(nse_ssa_data1),
    sum(nse_ssa_data3)/len(nse_ssa_data3),
    sum(nse_ssa_data5)/len(nse_ssa_data5),
    sum(nse_ssa_data7)/len(nse_ssa_data7),
]
vmd_mean_nse=[
    sum(nse_vmd_data1)/len(nse_vmd_data1),
    sum(nse_vmd_data3)/len(nse_vmd_data3),
    sum(nse_vmd_data5)/len(nse_vmd_data5),
    sum(nse_vmd_data7)/len(nse_vmd_data7),
]
dwt_mean_nse=[
    sum(nse_dwt_data1)/len(nse_dwt_data1),
    sum(nse_dwt_data3)/len(nse_dwt_data3),
    sum(nse_dwt_data5)/len(nse_dwt_data5),
    sum(nse_dwt_data7)/len(nse_dwt_data7),
]

modwt_mean_nse=[
    sum(nse_modwt_data1)/len(nse_modwt_data1),
    sum(nse_modwt_data3)/len(nse_modwt_data3),
    sum(nse_modwt_data5)/len(nse_modwt_data5),
    sum(nse_modwt_data7)/len(nse_modwt_data7),
]

eemd_nse_ins = [(nse-arima_mean_nse)/arima_mean_nse*100 for nse in eemd_mean_nse]
ssa_nse_ins = [(nse-arima_mean_nse)/arima_mean_nse*100 for nse in ssa_mean_nse]
dwt_nse_ins = [(nse-arima_mean_nse)/arima_mean_nse*100 for nse in dwt_mean_nse]
vmd_nse_ins = [(nse-arima_mean_nse)/arima_mean_nse*100 for nse in vmd_mean_nse]
modwt_nse_ins = [(nse-arima_mean_nse)/arima_mean_nse*100 for nse in modwt_mean_nse]

nse_ins={
    'EEMD-SVR':eemd_nse_ins,
    'SSA-SVR':ssa_nse_ins,
    'DWT-SVR':dwt_nse_ins,
    'VMD-SVR':vmd_nse_ins,
    'BCMODWT-SVR':modwt_nse_ins,
}

nse_ins_df = pd.DataFrame(nse_ins,index=['1-month ahead','3-month ahead','5-month ahead','7-month ahead'])
nse_ins_df.to_csv(root_path+'/results_analysis/results/mean_nse_increasement_based_arima.csv')
nrmse_arma_data = [h_arma['test_nrmse'],x_arma['test_nrmse'],z_arma['test_nrmse']]

nrmse_svr_data1 = [h_svr_1['test_nrmse'][0],x_svr_1['test_nrmse'][0],z_svr_1['test_nrmse'][0]]
nrmse_svr_data3 = [h_svr_3['test_nrmse'][0],x_svr_3['test_nrmse'][0],z_svr_3['test_nrmse'][0]]
nrmse_svr_data5 = [h_svr_5['test_nrmse'][0],x_svr_5['test_nrmse'][0],z_svr_5['test_nrmse'][0]]
nrmse_svr_data7 = [h_svr_7['test_nrmse'][0],x_svr_7['test_nrmse'][0],z_svr_7['test_nrmse'][0]]

nrmse_dnn_data1 = [h_dnn_1['test_nrmse'][0],x_dnn_1['test_nrmse'][0],z_dnn_1['test_nrmse'][0]]
nrmse_dnn_data3 = [h_dnn_3['test_nrmse'][0],x_dnn_3['test_nrmse'][0],z_dnn_3['test_nrmse'][0]]
nrmse_dnn_data5 = [h_dnn_5['test_nrmse'][0],x_dnn_5['test_nrmse'][0],z_dnn_5['test_nrmse'][0]]
nrmse_dnn_data7 = [h_dnn_7['test_nrmse'][0],x_dnn_7['test_nrmse'][0],z_dnn_7['test_nrmse'][0]]

nrmse_lstm_data1 = [h_lstm_1['test_nrmse'][0],x_lstm_1['test_nrmse'][0],z_lstm_1['test_nrmse'][0]]
nrmse_lstm_data3 = [h_lstm_3['test_nrmse'][0],x_lstm_3['test_nrmse'][0],z_lstm_3['test_nrmse'][0]]
nrmse_lstm_data5 = [h_lstm_5['test_nrmse'][0],x_lstm_5['test_nrmse'][0],z_lstm_5['test_nrmse'][0]]
nrmse_lstm_data7 = [h_lstm_7['test_nrmse'][0],x_lstm_7['test_nrmse'][0],z_lstm_7['test_nrmse'][0]]

nrmse_eemd_data1=[h_e_1['test_nrmse'][0],x_e_1['test_nrmse'][0],z_e_1['test_nrmse'][0]]
nrmse_eemd_data3=[h_e_3['test_nrmse'][0],x_e_3['test_nrmse'][0],z_e_3['test_nrmse'][0]]
nrmse_eemd_data5=[h_e_5['test_nrmse'][0],x_e_5['test_nrmse'][0],z_e_5['test_nrmse'][0]]
nrmse_eemd_data7=[h_e_7['test_nrmse'][0],x_e_7['test_nrmse'][0],z_e_7['test_nrmse'][0]]

nrmse_ssa_data1=[h_s_1['test_nrmse'][0],x_s_1['test_nrmse'][0],z_s_1['test_nrmse'][0]]
nrmse_ssa_data3=[h_s_3['test_nrmse'][0],x_s_3['test_nrmse'][0],z_s_3['test_nrmse'][0]]
nrmse_ssa_data5=[h_s_5['test_nrmse'][0],x_s_5['test_nrmse'][0],z_s_5['test_nrmse'][0]]
nrmse_ssa_data7=[h_s_7['test_nrmse'][0],x_s_7['test_nrmse'][0],z_s_7['test_nrmse'][0]]

nrmse_vmd_data1=[h_v_1['test_nrmse'][0],x_v_1['test_nrmse'][0],z_v_1['test_nrmse'][0]]
nrmse_vmd_data3=[h_v_3['test_nrmse'][0],x_v_3['test_nrmse'][0],z_v_3['test_nrmse'][0]]
nrmse_vmd_data5=[h_v_5['test_nrmse'][0],x_v_5['test_nrmse'][0],z_v_5['test_nrmse'][0]]
nrmse_vmd_data7=[h_v_7['test_nrmse'][0],x_v_7['test_nrmse'][0],z_v_7['test_nrmse'][0]]

nrmse_dwt_data1=[h_d_1['test_nrmse'][0],x_d_1['test_nrmse'][0],z_d_1['test_nrmse'][0]]
nrmse_dwt_data3=[h_d_3['test_nrmse'][0],x_d_3['test_nrmse'][0],z_d_3['test_nrmse'][0]]
nrmse_dwt_data5=[h_d_5['test_nrmse'][0],x_d_5['test_nrmse'][0],z_d_5['test_nrmse'][0]]
nrmse_dwt_data7=[h_d_7['test_nrmse'][0],x_d_7['test_nrmse'][0],z_d_7['test_nrmse'][0]]

nrmse_modwt_data1=[h_m_1['test_nrmse'][0],x_m_1['test_nrmse'][0],z_m_1['test_nrmse'][0]]
nrmse_modwt_data3=[h_m_3['test_nrmse'][0],x_m_3['test_nrmse'][0],z_m_3['test_nrmse'][0]]
nrmse_modwt_data5=[h_m_5['test_nrmse'][0],x_m_5['test_nrmse'][0],z_m_5['test_nrmse'][0]]
nrmse_modwt_data7=[h_m_7['test_nrmse'][0],x_m_7['test_nrmse'][0],z_m_7['test_nrmse'][0]]
nrmse_data=[
    nrmse_arma_data,
    nrmse_svr_data1,
    nrmse_svr_data3,
    nrmse_svr_data5,
    nrmse_svr_data7,
    nrmse_dnn_data1,
    nrmse_dnn_data3,
    nrmse_dnn_data5,
    nrmse_dnn_data7,
    nrmse_lstm_data1,
    nrmse_lstm_data3,
    nrmse_lstm_data5,
    nrmse_lstm_data7,
    nrmse_eemd_data1,
    nrmse_eemd_data3,
    nrmse_eemd_data5,
    nrmse_eemd_data7,
    nrmse_ssa_data1,
    nrmse_ssa_data3,
    nrmse_ssa_data5,
    nrmse_ssa_data7,
    nrmse_vmd_data1,
    nrmse_vmd_data3,
    nrmse_vmd_data5,
    nrmse_vmd_data7,
    nrmse_dwt_data1,
    nrmse_dwt_data3,
    nrmse_dwt_data5,
    nrmse_dwt_data7,
    nrmse_modwt_data1,
    nrmse_modwt_data3,
    nrmse_modwt_data5,
    nrmse_modwt_data7,
]
eemd_mean_nrmse=[
    sum(nrmse_eemd_data1)/len(nrmse_eemd_data1),
    sum(nrmse_eemd_data3)/len(nrmse_eemd_data3),
    sum(nrmse_eemd_data5)/len(nrmse_eemd_data5),
    sum(nrmse_eemd_data7)/len(nrmse_eemd_data7),
]
ssa_mean_nrmse=[
    sum(nrmse_ssa_data1)/len(nrmse_ssa_data1),
    sum(nrmse_ssa_data3)/len(nrmse_ssa_data3),
    sum(nrmse_ssa_data5)/len(nrmse_ssa_data5),
    sum(nrmse_ssa_data7)/len(nrmse_ssa_data7),
]
vmd_mean_nrmse=[
    sum(nrmse_vmd_data1)/len(nrmse_vmd_data1),
    sum(nrmse_vmd_data3)/len(nrmse_vmd_data3),
    sum(nrmse_vmd_data5)/len(nrmse_vmd_data5),
    sum(nrmse_vmd_data7)/len(nrmse_vmd_data7),
]
dwt_mean_nrmse=[
    sum(nrmse_dwt_data1)/len(nrmse_dwt_data1),
    sum(nrmse_dwt_data3)/len(nrmse_dwt_data3),
    sum(nrmse_dwt_data5)/len(nrmse_dwt_data5),
    sum(nrmse_dwt_data7)/len(nrmse_dwt_data7),
]
modwt_mean_nrmse=[
    sum(nrmse_modwt_data1)/len(nrmse_modwt_data1),
    sum(nrmse_modwt_data3)/len(nrmse_modwt_data3),
    sum(nrmse_modwt_data5)/len(nrmse_modwt_data5),
    sum(nrmse_modwt_data7)/len(nrmse_modwt_data7),
]
ppts_arma_data = [h_arma['test_ppts'],x_arma['test_ppts'],z_arma['test_ppts']]
ppts_svr_data1 = [h_svr_1['test_ppts'][0],x_svr_1['test_ppts'][0],z_svr_1['test_ppts'][0]]
ppts_svr_data3 = [h_svr_3['test_ppts'][0],x_svr_3['test_ppts'][0],z_svr_3['test_ppts'][0]]
ppts_svr_data5 = [h_svr_5['test_ppts'][0],x_svr_5['test_ppts'][0],z_svr_5['test_ppts'][0]]
ppts_svr_data7 = [h_svr_7['test_ppts'][0],x_svr_7['test_ppts'][0],z_svr_7['test_ppts'][0]]

ppts_dnn_data1 = [h_dnn_1['test_ppts'][0],x_dnn_1['test_ppts'][0],z_dnn_1['test_ppts'][0]]
ppts_dnn_data3 = [h_dnn_3['test_ppts'][0],x_dnn_3['test_ppts'][0],z_dnn_3['test_ppts'][0]]
ppts_dnn_data5 = [h_dnn_5['test_ppts'][0],x_dnn_5['test_ppts'][0],z_dnn_5['test_ppts'][0]]
ppts_dnn_data7 = [h_dnn_7['test_ppts'][0],x_dnn_7['test_ppts'][0],z_dnn_7['test_ppts'][0]]


ppts_lstm_data1 = [h_lstm_1['test_ppts'][0],x_lstm_1['test_ppts'][0],z_lstm_1['test_ppts'][0]]
ppts_lstm_data3 = [h_lstm_3['test_ppts'][0],x_lstm_3['test_ppts'][0],z_lstm_3['test_ppts'][0]]
ppts_lstm_data5 = [h_lstm_5['test_ppts'][0],x_lstm_5['test_ppts'][0],z_lstm_5['test_ppts'][0]]
ppts_lstm_data7 = [h_lstm_7['test_ppts'][0],x_lstm_7['test_ppts'][0],z_lstm_7['test_ppts'][0]]


ppts_eemd_data1=[h_e_1['test_ppts'][0],x_e_1['test_ppts'][0],z_e_1['test_ppts'][0]]
ppts_eemd_data3=[h_e_3['test_ppts'][0],x_e_3['test_ppts'][0],z_e_3['test_ppts'][0]]
ppts_eemd_data5=[h_e_5['test_ppts'][0],x_e_5['test_ppts'][0],z_e_5['test_ppts'][0]]
ppts_eemd_data7=[h_e_7['test_ppts'][0],x_e_7['test_ppts'][0],z_e_7['test_ppts'][0]]
ppts_ssa_data1=[h_s_1['test_ppts'][0],x_s_1['test_ppts'][0],z_s_1['test_ppts'][0]]
ppts_ssa_data3=[h_s_3['test_ppts'][0],x_s_3['test_ppts'][0],z_s_3['test_ppts'][0]]
ppts_ssa_data5=[h_s_5['test_ppts'][0],x_s_5['test_ppts'][0],z_s_5['test_ppts'][0]]
ppts_ssa_data7=[h_s_7['test_ppts'][0],x_s_7['test_ppts'][0],z_s_7['test_ppts'][0]]
ppts_vmd_data1=[h_v_1['test_ppts'][0],x_v_1['test_ppts'][0],z_v_1['test_ppts'][0]]
ppts_vmd_data3=[h_v_3['test_ppts'][0],x_v_3['test_ppts'][0],z_v_3['test_ppts'][0]]
ppts_vmd_data5=[h_v_5['test_ppts'][0],x_v_5['test_ppts'][0],z_v_5['test_ppts'][0]]
ppts_vmd_data7=[h_v_7['test_ppts'][0],x_v_7['test_ppts'][0],z_v_7['test_ppts'][0]]
ppts_dwt_data1=[h_d_1['test_ppts'][0],x_d_1['test_ppts'][0],z_d_1['test_ppts'][0]]
ppts_dwt_data3=[h_d_3['test_ppts'][0],x_d_3['test_ppts'][0],z_d_3['test_ppts'][0]]
ppts_dwt_data5=[h_d_5['test_ppts'][0],x_d_5['test_ppts'][0],z_d_5['test_ppts'][0]]
ppts_dwt_data7=[h_d_7['test_ppts'][0],x_d_7['test_ppts'][0],z_d_7['test_ppts'][0]]
ppts_modwt_data1=[h_m_1['test_ppts'][0],x_m_1['test_ppts'][0],z_m_1['test_ppts'][0]]
ppts_modwt_data3=[h_m_3['test_ppts'][0],x_m_3['test_ppts'][0],z_m_3['test_ppts'][0]]
ppts_modwt_data5=[h_m_5['test_ppts'][0],x_m_5['test_ppts'][0],z_m_5['test_ppts'][0]]
ppts_modwt_data7=[h_m_7['test_ppts'][0],x_m_7['test_ppts'][0],z_m_7['test_ppts'][0]]
ppts_data=[
    ppts_arma_data,
    ppts_svr_data1,
    ppts_svr_data3,
    ppts_svr_data5,
    ppts_svr_data7,
    ppts_dnn_data1,
    ppts_dnn_data3,
    ppts_dnn_data5,
    ppts_dnn_data7,
    ppts_lstm_data1,
    ppts_lstm_data3,
    ppts_lstm_data5,
    ppts_lstm_data7,
    ppts_eemd_data1,
    ppts_eemd_data3,
    ppts_eemd_data5,
    ppts_eemd_data7,
    ppts_ssa_data1,
    ppts_ssa_data3,
    ppts_ssa_data5,
    ppts_ssa_data7,
    ppts_vmd_data1,
    ppts_vmd_data3,
    ppts_vmd_data5,
    ppts_vmd_data7,
    ppts_dwt_data1,
    ppts_dwt_data3,
    ppts_dwt_data5,
    ppts_dwt_data7,
    ppts_modwt_data1,
    ppts_modwt_data3,
    ppts_modwt_data5,
    ppts_modwt_data7,
]
eemd_mean_ppts=[
    sum(ppts_eemd_data1)/len(ppts_eemd_data1),
    sum(ppts_eemd_data3)/len(ppts_eemd_data3),
    sum(ppts_eemd_data5)/len(ppts_eemd_data5),
    sum(ppts_eemd_data7)/len(ppts_eemd_data7),
]
ssa_mean_ppts=[
    sum(ppts_ssa_data1)/len(ppts_ssa_data1),
    sum(ppts_ssa_data3)/len(ppts_ssa_data3),
    sum(ppts_ssa_data5)/len(ppts_ssa_data5),
    sum(ppts_ssa_data7)/len(ppts_ssa_data7),
]
vmd_mean_ppts=[
    sum(ppts_vmd_data1)/len(ppts_vmd_data1),
    sum(ppts_vmd_data3)/len(ppts_vmd_data3),
    sum(ppts_vmd_data5)/len(ppts_vmd_data5),
    sum(ppts_vmd_data7)/len(ppts_vmd_data7),
]
dwt_mean_ppts=[
    sum(ppts_dwt_data1)/len(ppts_dwt_data1),
    sum(ppts_dwt_data3)/len(ppts_dwt_data3),
    sum(ppts_dwt_data5)/len(ppts_dwt_data5),
    sum(ppts_dwt_data7)/len(ppts_dwt_data7),
]
modwt_mean_ppts=[
    sum(ppts_modwt_data1)/len(ppts_modwt_data1),
    sum(ppts_modwt_data3)/len(ppts_modwt_data3),
    sum(ppts_modwt_data5)/len(ppts_modwt_data5),
    sum(ppts_modwt_data7)/len(ppts_modwt_data7),
]
nse_lines=[
    eemd_mean_nse,
    ssa_mean_nse,
    vmd_mean_nse,
    dwt_mean_nse,
    modwt_mean_nse,
]
nrmse_lines=[
    eemd_mean_nrmse,
    ssa_mean_nrmse,
    vmd_mean_nrmse,
    dwt_mean_nrmse,
    modwt_mean_nrmse,
]
ppts_lines=[
    eemd_mean_ppts,
    ssa_mean_ppts,
    vmd_mean_ppts,
    dwt_mean_ppts,
    modwt_mean_ppts,
]
lines=[
    nse_lines,
    nrmse_lines,
    ppts_lines,
]
print("NSE"+"-"*100)
base_nse = vmd_mean_nse[0]
for i in range(1,len(eemd_mean_nse)):
    ratio = (vmd_mean_nse[i]-base_nse)/base_nse*100
    print("VMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (ssa_mean_nse[i]-base_nse)/base_nse*100
    print("SSA-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (eemd_mean_nse[i]-base_nse)/base_nse*100
    print("EEMMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (dwt_mean_nse[i]-base_nse)/base_nse*100
    print("DWT-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (modwt_mean_nse[i]-base_nse)/base_nse*100
    print("MODWT-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
print("NRMSE"+"-"*100)
base_nrmse = vmd_mean_nrmse[0]
for i in range(1,len(eemd_mean_nrmse)):
    ratio = (vmd_mean_nrmse[i]-base_nrmse)/base_nrmse*100
    print("VMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (ssa_mean_nrmse[i]-base_nrmse)/base_nrmse*100
    print("SSA-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (eemd_mean_nrmse[i]-base_nrmse)/base_nrmse*100
    print("EEMMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (dwt_mean_nrmse[i]-base_nrmse)/base_nrmse*100
    print("DWT-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (modwt_mean_nrmse[i]-base_nrmse)/base_nrmse*100
    print("MODWT-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
print("PPTS"+"-"*100)
base_ppts = vmd_mean_ppts[0]
for i in range(1,len(eemd_mean_ppts)):
    ratio = (vmd_mean_ppts[i]-base_ppts)/base_ppts*100
    print("VMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (ssa_mean_ppts[i]-base_ppts)/base_ppts*100
    print("SSA-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (eemd_mean_ppts[i]-base_ppts)/base_ppts*100
    print("EEMMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (dwt_mean_ppts[i]-base_ppts)/base_ppts*100
    print("DWT-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    ratio = (modwt_mean_ppts[i]-base_ppts)/base_ppts*100
    print("MODWT-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
all_datas = [nse_data,nrmse_data,ppts_data]
fig_index=["(a)","(b)","(c)"]
labels=[
    'ARIMA, 1',
    'SVR, 1',
    'SVR, 3',
    'SVR, 5',
    'SVR, 7',
    'BPNN, 1',
    'BPNN, 3',
    'BPNN, 5',
    'BPNN, 7',
    'LSTM, 1',
    'LSTM, 3',
    'LSTM, 5',
    'LSTM, 7',
    "EEMD-SVR, 1",
    "EEMD-SVR, 3",
    "EEMD-SVR, 5",
    "EEMD-SVR, 7",
    "SSA-SVR, 1",
    "SSA-SVR, 3",
    "SSA-SVR, 5",
    "SSA-SVR, 7",
    "VMD-SVR, 1",
    "VMD-SVR, 3",
    "VMD-SVR, 5",
    "VMD-SVR, 7",
    "DWT-SVR, 1",
    "DWT-SVR, 3",
    "DWT-SVR, 5",
    "DWT-SVR, 7",
    "BCMODWT-SVR, 1",
    "BCMODWT-SVR, 3",
    "BCMODWT-SVR, 5",
    "BCMODWT-SVR, 7",
]
face_colors={
    0:'#DA70D6',#arima
    1:'#B0C4DE',#svr
    2:'#B0C4DE',#svr
    3:'#B0C4DE',#svr
    4:'#B0C4DE',#svr
    5:'#F08080',#bpnn 
    6:'#F08080',#bpnn 
    7:'#F08080',#bpnn 
    8:'#F08080',#bpnn 
    9:'#87CEFA',#lstm
    10:'#87CEFA',#lstm
    11:'#87CEFA',#lstm
    12:'#87CEFA',#lstm
    13:'#00FF9F',#eemd-svr
    14:'#00FF9F',#eemd-svr
    15:'#00FF9F',#eemd-svr
    16:'#00FF9F',#eemd-svr
    17:'#FFE4E1',#ssa-svr
    18:'#FFE4E1',#ssa-svr
    19:'#FFE4E1',#ssa-svr
    20:'#FFE4E1',#ssa-svr
    21:'#FF4500',#vmd-svr
    22:'#FF4500',#vmd-svr
    23:'#FF4500',#vmd-svr
    24:'#FF4500',#vmd-svr
    25:'#FAA460',#dwt-svr
    26:'#FAA460',#dwt-svr
    27:'#FAA460',#dwt-svr
    28:'#FAA460',#dwt-svr
    29:'#ADFF2F',#dwt-svr
    30:'#ADFF2F',#dwt-svr
    31:'#ADFF2F',#dwt-svr
    32:'#ADFF2F',#dwt-svr
}
x = list(range(len(labels)))
ylabels=[
    r"$NSE$",r"$NRMSE\ (10^8m^3)$",r"$PPTS(5)\ (\%)$",
]
x_s=[-1.1,-1.1,-1.1]
y_s=[0.85,1.55,70]
plt.figure(figsize=(7.48, 4.54))
# plt.figure(figsize=(7.48, 7.48))
for i in range(len(all_datas)):
    ax1 = plt.subplot(3, 1, i+1)
    ax1.yaxis.grid(True)
    if i in [0,1]:
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.text(x_s[i],y_s[i],fig_index[i],fontsize=7)
    vplot1 = plt.violinplot(
        dataset=all_datas[i],
        positions=x,
        showmeans=True,
    )
    L=len(lines[i][0])
    # ax1.plot(list(range(2,L+2)),lines[i][0],'--',lw=0.5,color='blue')
    # ax1.plot(list(range(L+2,2*L+2)),lines[i][1],'--',lw=0.5,color='blue')
    # ax1.plot(list(range(2*L+2,3*L+2)),lines[i][2],'--',lw=0.5,color='blue')
    # ax1.plot(list(range(3*L+2,4*L+2)),lines[i][3],'--',lw=0.5,color='blue')
    # ax1.plot(list(range(4*L+2,5*L+2)),lines[i][4],'--',lw=0.5,color='blue')
    print(type(vplot1["cmeans"]))
    plt.ylabel(ylabels[i])
    if i==len(all_datas)-1:
        plt.xticks(x, labels, rotation=45)
    else:
        plt.xticks([])
    for pc,k in zip(vplot1['bodies'],range(len(face_colors))):
        print(pc)
        # pc.set_facecolor('#D43F3A')
        pc.set_facecolor(face_colors[k])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
# plt.tight_layout()
plt.subplots_adjust(left=0.066, bottom=0.16, right=0.99,top=0.99, hspace=0.05, wspace=0.25)
plt.savefig(graphs_path+'Violin plots for TSDP and WDDFF at Huaxian.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'Violin plots for TSDP and WDDFF at Huaxian.tif',format='TIFF',dpi=500)
plt.savefig(graphs_path+'Violin plots for TSDP and WDDFF at Huaxian.pdf',format='PDF',dpi=1200)
plt.show()