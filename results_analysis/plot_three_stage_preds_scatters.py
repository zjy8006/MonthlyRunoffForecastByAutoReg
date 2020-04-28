import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
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

h_esvr=read_pure_esvr("Huaxian")
x_esvr=read_pure_esvr("Xianyang")
z_esvr=read_pure_esvr("Zhangjiashan")

h_eemd_esvr = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
h_ssa_esvr = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
h_vmd_esvr = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
h_dwt_esvr = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')

x_eemd_esvr = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
x_ssa_esvr = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
x_vmd_esvr = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
x_dwt_esvr = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')

z_eemd_esvr = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
z_ssa_esvr = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
z_vmd_esvr = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
z_dwt_esvr = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')


data = [
    [h_arma,x_arma,z_arma],
    [h_esvr,x_esvr,z_esvr],
    [h_eemd_esvr,x_eemd_esvr,z_eemd_esvr],
    [h_ssa_esvr,x_ssa_esvr,z_ssa_esvr],
    [h_dwt_esvr,x_dwt_esvr,z_dwt_esvr],
    [h_vmd_esvr,x_vmd_esvr,z_vmd_esvr],
]

models=['ARMA','SVR','EEMD-SVR-A','SSA-SVR-A','DWT-SVR-A','VMD-SVR-A']

idx = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12],
    [13,14,15],
    [16,17,18],
]


fig_idx=['(a)','(b)','(c)']
plt.figure(figsize=(7.48,7.48))
for i in range(len(data)):
    for j in range(len(data[i])):
        plt.subplot(7,3,idx[i][j])
        plt.plot(data[i][j]['test_y'],label='Records',c='tab:blue',lw=0.8)
        plt.plot(data[i][j]['test_pred'],'--',label=models[i],c='tab:red',lw=0.8)
        plt.ylabel("Flow(" + r"$10^8m^3$" + ")")
        if i==len(data)-1:
            plt.xlabel('Time(month)\n'+fig_idx[j])
        plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(graphs_path+'two_stage_predictions.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'two_stage_predictions.tif',format='TIFF',dpi=1200)
# plt.show()



records_data=[
    [h_arma['test_y'],h_esvr['test_y'],h_eemd_esvr['test_y'][0:120].values,h_ssa_esvr['test_y'][0:120].values,h_dwt_esvr['test_y'][0:120].values,h_vmd_esvr['test_y'][0:120].values],
    [x_arma['test_y'],x_esvr['test_y'],x_eemd_esvr['test_y'][0:120].values,x_ssa_esvr['test_y'][0:120].values,x_dwt_esvr['test_y'][0:120].values,x_vmd_esvr['test_y'][0:120].values],
    [z_arma['test_y'],z_esvr['test_y'],z_eemd_esvr['test_y'][0:120].values,z_ssa_esvr['test_y'][0:120].values,z_dwt_esvr['test_y'][0:120].values,z_vmd_esvr['test_y'][0:120].values],
]
preds_data=[
    [h_arma['test_pred'],h_esvr['test_pred'],h_eemd_esvr['test_pred'][0:120].values,h_ssa_esvr['test_pred'][0:120].values,h_dwt_esvr['test_pred'][0:120].values,h_vmd_esvr['test_pred'][0:120].values],
    [x_arma['test_pred'],x_esvr['test_pred'],x_eemd_esvr['test_pred'][0:120].values,x_ssa_esvr['test_pred'][0:120].values,x_dwt_esvr['test_pred'][0:120].values,x_vmd_esvr['test_pred'][0:120].values],
    [z_arma['test_pred'],z_esvr['test_pred'],z_eemd_esvr['test_pred'][0:120].values,z_ssa_esvr['test_pred'][0:120].values,z_dwt_esvr['test_pred'][0:120].values,z_vmd_esvr['test_pred'][0:120].values],
]

markers=['<','v','s','+','d','o']
colors=['r','g','teal','purple','gold','blue']
zorders=[0,1,2,3,4,5,6]
plt.figure(figsize=(7.48,3.2))
for i in range(len(records_data)):
    plt.subplot(1,3,i+1,aspect='equal')
    records_list=records_data[i]
    predictions_list=preds_data[i]
    xx,linear_list,xymin,xymax=compute_multi_linear_fit(
        records=records_list,
        predictions=predictions_list,
    )
    plt.xlabel('Predictions(' + r'$10^8m^3$' +')\n'+fig_idx[i], )
    if i==0:
        plt.ylabel('Records(' + r'$10^8m^3$' + ')', )
    for j in range(len(predictions_list)):
        # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
        plt.scatter(predictions_list[j], records_list[j],label=models[j],marker=markers[j],zorder=zorders[j])
        plt.plot(xx, linear_list[j], '--', label=models[j],linewidth=1.0,zorder=zorders[j])
    plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
    plt.xlim([xymin,xymax])
    plt.ylim([xymin,xymax])
    if i==1:
        plt.legend(
                    loc='upper center',
                    # bbox_to_anchor=(0.08,1.01, 1,0.101),
                    bbox_to_anchor=(0.5,1.18),
                    ncol=8,
                    shadow=False,
                    frameon=True,
                    )
# plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.1, right=0.99,top=0.96, hspace=0.2, wspace=0.15)
plt.savefig(graphs_path+'Scatter plots for TSDP.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'Scatter plots for TSDP.tif',format='TIFF',dpi=1200)
plt.savefig(graphs_path+'Scatter plots for TSDP.pdf',format='PDF',dpi=1200)
plt.show()

