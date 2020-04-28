import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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
from tools.results_reader import read_two_stage,read_pure_esvr
from tools.fit_line import compute_linear_fit,compute_multi_linear_fit

# h_records,h_predictions,h_r2,h_nrmse,h_mae,h_mape,h_ppts,h_timecost=read_pure_esvr("Huaxian")
# x_records,x_predictions,x_r2,x_nrmse,x_mae,x_mape,x_ppts,x_timecost=read_pure_esvr("Xianyang")
# z_records,z_predictions,z_r2,z_nrmse,z_mae,z_mape,z_ppts,z_timecost=read_pure_esvr("Zhangjiashan")

h_vmd= read_two_stage(station="Huaxian",decomposer="vmd",predict_pattern="one_step_1_ahead_forecast_pacf")
x_vmd= read_two_stage(station="Xianyang",decomposer="vmd",predict_pattern="one_step_1_ahead_forecast_pacf")
z_vmd= read_two_stage(station="Zhangjiashan",decomposer="vmd",predict_pattern="one_step_1_ahead_forecast_pacf")

h_eemd= read_two_stage(station="Huaxian",decomposer="eemd",predict_pattern="one_step_1_ahead_forecast_pacf")
x_eemd= read_two_stage(station="Xianyang",decomposer="eemd",predict_pattern="one_step_1_ahead_forecast_pacf")
z_eemd= read_two_stage(station="Zhangjiashan",decomposer="eemd",predict_pattern="one_step_1_ahead_forecast_pacf")

h_ssa= read_two_stage(station="Huaxian",decomposer="ssa",predict_pattern="one_step_1_ahead_forecast_pacf")
x_ssa= read_two_stage(station="Xianyang",decomposer="ssa",predict_pattern="one_step_1_ahead_forecast_pacf")
z_ssa= read_two_stage(station="Zhangjiashan",decomposer="ssa",predict_pattern="one_step_1_ahead_forecast_pacf")

h_dwt= read_two_stage(station="Huaxian",decomposer="dwt",predict_pattern="one_step_1_ahead_forecast_pacf")
x_dwt= read_two_stage(station="Xianyang",decomposer="dwt",predict_pattern="one_step_1_ahead_forecast_pacf")
z_dwt= read_two_stage(station="Zhangjiashan",decomposer="dwt",predict_pattern="one_step_1_ahead_forecast_pacf")

h_eemd_a = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
h_ssa_a = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
h_vmd_a = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
h_dwt_a = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')

x_eemd_a = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
x_ssa_a = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
x_vmd_a = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
x_dwt_a = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')

z_eemd_a = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
z_ssa_a = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
z_vmd_a = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
z_dwt_a = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')


##########################################################################################################
records_sets=[
    [
        [h_eemd['test_y'],h_eemd_a['test_y'][0:120].values,],
        [h_ssa['test_y'],h_ssa_a['test_y'][0:120].values,],
        [h_vmd['test_y'],h_vmd_a['test_y'][0:120].values,],
        [h_dwt['test_y'],h_dwt_a['test_y'][0:120].values,],
    ],
    [
        [x_eemd['test_y'],x_eemd_a['test_y'][0:120].values,],
        [x_ssa['test_y'],x_ssa_a['test_y'][0:120].values,],
        [x_vmd['test_y'],x_vmd_a['test_y'][0:120].values,],
        [x_dwt['test_y'],x_dwt_a['test_y'][0:120].values,],
    ],
    [
        [z_eemd['test_y'],z_eemd_a['test_y'][0:120].values,],
        [z_ssa['test_y'],z_ssa_a['test_y'][0:120].values,],
        [z_vmd['test_y'],z_vmd_a['test_y'][0:120].values,],
        [z_dwt['test_y'],z_dwt_a['test_y'][0:120].values,],  
    ],
]
predictions_sets=[
    [
        [h_eemd['test_pred'],h_eemd_a['test_pred'][0:120].values,],
        [h_ssa['test_pred'],h_ssa_a['test_pred'][0:120].values,],
        [h_vmd['test_pred'],h_vmd_a['test_pred'][0:120].values,],
        [h_dwt['test_pred'],h_dwt_a['test_pred'][0:120].values,],
    ],
    [
        [x_eemd['test_pred'],x_eemd_a['test_pred'][0:120].values,],
        [x_ssa['test_pred'],x_ssa_a['test_pred'][0:120].values,],
        [x_vmd['test_pred'],x_vmd_a['test_pred'][0:120].values,],
        [x_dwt['test_pred'],x_dwt_a['test_pred'][0:120].values,],
    ],
    [
        [z_eemd['test_pred'],z_eemd_a['test_pred'][0:120].values,],
        [z_ssa['test_pred'],z_ssa_a['test_pred'][0:120].values,],
        [z_vmd['test_pred'],z_vmd_a['test_pred'][0:120].values,],
        [z_dwt['test_pred'],z_dwt_a['test_pred'][0:120].values,],  
    ],
]

models_labels=[
    ['EEMD-SVR','EEMD-SVR-A',],
    ['SSA-SVR','SSA-SVR-A',],
    ['VMD-SVR','VMD-SVR-A',],
    ['DWT-SVR','DWT-SVR-A',],
]

x_sets=[
    [29.5,29.5,29.5,30],
    [17.5,17.5,17.9,19.5],
    [4.5,4.5,4.6,4.8],
]
y_sets=[
    [1.4,2,1.75,1.75],
    [0.9,0.9,0.9,0.9],
    [0.2,0.18,0.18,0.18],
]
stations=['Huaxian','Xianyang','Zhangjiashan']
fig_idx=['(a)','(b)','(c)','(d)']
for k in range(len(records_sets)):
    records_list=records_sets[k]
    predictions_list=predictions_sets[k]
    x_s = x_sets[k]
    y_s = y_sets[k]
    plt.figure(figsize=(5.5,5.5))
    for j in range(len(records_list)):
        ax=plt.subplot(2,2,j+1, aspect='equal')
        xx,linear_list,xymin,xymax=compute_multi_linear_fit(
            records=records_list[j],
            predictions=predictions_list[j],
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # if j in range(8,12):
        plt.xlabel('Predictions(' + r'$10^8m^3$' +')', )
        # if j in [0,4,8]:
        plt.ylabel('Records(' + r'$10^8m^3$' + ')', )
        models=models_labels[j]
        markers=['o','v',]
        zorders=[1,0]
        plt.text(x_s[j],y_s[j],fig_idx[j],fontweight='normal',fontsize=7)
        for i in range(len(predictions_list[j])):
            # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
            plt.scatter(predictions_list[j][i], records_list[j][i],marker=markers[i],zorder=zorders[i],)
            plt.plot(xx, linear_list[i], '--', label=models[i],linewidth=1.0,zorder=zorders[i])
        plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
        plt.xlim([xymin,xymax])
        plt.ylim([xymin,xymax])
        plt.legend(
                    loc=0,
                    # bbox_to_anchor=(0.08,1.01, 1,0.101),
                    # bbox_to_anchor=(1,1),
                    # ncol=2,
                    shadow=False,
                    frameon=False,
                    fontsize=6,
                    )
    plt.tight_layout()
    # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.99,top=0.99, hspace=0.15, wspace=0.1)
    plt.savefig(graphs_path+'Scatters of TSDP vs TSDPE at '+stations[k]+'.eps',format='EPS',dpi=2000)
    plt.savefig(graphs_path+'Scatters of TSDP vs TSDPE at '+stations[k]+'.tif',format='TIFF',dpi=1200)
plt.show()


records_list=[
    
        [h_eemd['test_y'],h_eemd_a['test_y'][0:120].values,],
        [h_ssa['test_y'],h_ssa_a['test_y'][0:120].values,],
        [h_vmd['test_y'],h_vmd_a['test_y'][0:120].values,],
        [h_dwt['test_y'],h_dwt_a['test_y'][0:120].values,],
        [x_eemd['test_y'],x_eemd_a['test_y'][0:120].values,],
        [x_ssa['test_y'],x_ssa_a['test_y'][0:120].values,],
        [x_vmd['test_y'],x_vmd_a['test_y'][0:120].values,],
        [x_dwt['test_y'],x_dwt_a['test_y'][0:120].values,],
        [z_eemd['test_y'],z_eemd_a['test_y'][0:120].values,],
        [z_ssa['test_y'],z_ssa_a['test_y'][0:120].values,],
        [z_vmd['test_y'],z_vmd_a['test_y'][0:120].values,],
        [z_dwt['test_y'],z_dwt_a['test_y'][0:120].values,],  
    
]
predictions_list=[
    
        [h_eemd['test_pred'],h_eemd_a['test_pred'][0:120].values,],
        [h_ssa['test_pred'],h_ssa_a['test_pred'][0:120].values,],
        [h_vmd['test_pred'],h_vmd_a['test_pred'][0:120].values,],
        [h_dwt['test_pred'],h_dwt_a['test_pred'][0:120].values,],
        [x_eemd['test_pred'],x_eemd_a['test_pred'][0:120].values,],
        [x_ssa['test_pred'],x_ssa_a['test_pred'][0:120].values,],
        [x_vmd['test_pred'],x_vmd_a['test_pred'][0:120].values,],
        [x_dwt['test_pred'],x_dwt_a['test_pred'][0:120].values,],
        [z_eemd['test_pred'],z_eemd_a['test_pred'][0:120].values,],
        [z_ssa['test_pred'],z_ssa_a['test_pred'][0:120].values,],
        [z_vmd['test_pred'],z_vmd_a['test_pred'][0:120].values,],
        [z_dwt['test_pred'],z_dwt_a['test_pred'][0:120].values,],  
]

models_labels=[
    ['EEMD-SVR','EEMD-SVR-A',],
    ['SSA-SVR','SSA-SVR-A',],
    ['VMD-SVR','VMD-SVR-A',],
    ['DWT-SVR','DWT-SVR-A',],
    ['EEMD-SVR','EEMD-SVR-A',],
    ['SSA-SVR','SSA-SVR-A',],
    ['VMD-SVR','VMD-SVR-A',],
    ['DWT-SVR','DWT-SVR-A',],
    ['EEMD-SVR','EEMD-SVR-A',],
    ['SSA-SVR','SSA-SVR-A',],
    ['VMD-SVR','VMD-SVR-A',],
    ['DWT-SVR','DWT-SVR-A',],
]
x_s=[
    27.8,28.,28.,28.5,
    16.5,16.5,16.9,18.5,
    4.3,4.3,4.4,4.6,
]
y_s=[
    1.4,2,1.75,1.75,
    0.9,0.9,0.9,0.9,
    0.2,0.18,0.18,0.18,
]
stations=['Huaxian','Xianyang','Zhangjiashan']
fig_idx=[
    '(a1)','(a2)','(a3)','(a4)',
    '(b1)','(b2)','(b3)','(b4)',
    '(c1)','(c2)','(c3)','(c4)',
    '(d1)','(d2)','(d3)','(d4)',
]
# models_labels=[
#     ['EEMD-SVR','EEMD-SVR-A',],
#     ['EEMD-SVR','EEMD-SVR-A',],
#     ['EEMD-SVR','EEMD-SVR-A',],
#     ['SSA-SVR','SSA-SVR-A',],
#     ['SSA-SVR','SSA-SVR-A',],
#     ['SSA-SVR','SSA-SVR-A',],
#     ['VMD-SVR','VMD-SVR-A',],
#     ['VMD-SVR','VMD-SVR-A',],
#     ['VMD-SVR','VMD-SVR-A',],
#     ['DWT-SVR','DWT-SVR-A',],
#     ['DWT-SVR','DWT-SVR-A',],
#     ['DWT-SVR','DWT-SVR-A',],
# ]
# fig_idx=[
#     '(a1)','(b1)','(c1)',
#     '(a2)','(b2)','(c2)',
#     '(a3)','(b3)','(c3)',
#     '(a4)','(b4)','(c4)',
# ]
# x_s=[
#     29.,17.,4.4,
#     29.,17.,4.4,
#     29.,17.4,4.5,
#     29.,19.,4.7,
# ]
# y_s=[
#     1.4, 0.8,0.2,
#     2,   0.8,0.18,
#     1.75,0.8,0.18,
#     1.74,0.8,0.18,
# ]

plt.figure(figsize=(7.4861,6))
for j in range(len(records_list)):
    ax=plt.subplot(3,4,j+1, aspect='equal')
    xx,linear_list,xymin,xymax=compute_multi_linear_fit(
        records=records_list[j],
        predictions=predictions_list[j],
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    if j in range(8,12):
        plt.xlabel('Predictions(' + r'$10^8m^3$' +')', )
    if j in [0,4,8]:
        plt.ylabel('Records(' + r'$10^8m^3$' + ')', )
    models=models_labels[j]
    markers=['o','v',]
    zorders=[1,0]
    plt.text(x_s[j],y_s[j],fig_idx[j],fontweight='normal',fontsize=7)
    for i in range(len(predictions_list[j])):
        # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
        plt.scatter(predictions_list[j][i], records_list[j][i],marker=markers[i],zorder=zorders[i],)
        plt.plot(xx, linear_list[i], '--', label=models[i],linewidth=1.0,zorder=zorders[i])
    plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
    plt.xlim([xymin,xymax])
    plt.ylim([xymin,xymax])
    plt.legend(
                loc=0,
                # bbox_to_anchor=(0.08,1.01, 1,0.101),
                # bbox_to_anchor=(1,1),
                # ncol=2,
                shadow=False,
                frameon=False,
                fontsize=6,
                )
plt.subplots_adjust(left=0.06, bottom=0.08, right=0.99,top=0.99, hspace=0.1, wspace=0.15)
plt.savefig(graphs_path+'Scatters of TSDP vs TSDPE.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'Scatters of TSDP vs TSDPE.tif',format='TIFF',dpi=1200)
plt.savefig(graphs_path+'Scatters of TSDP vs TSDPE.pdf',format='PDF',dpi=1200)
plt.show()
