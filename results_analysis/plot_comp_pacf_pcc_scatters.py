import sys
import os
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
root_path = os.path.dirname(os.path.abspath('__file__'))
graphs_path = root_path+'/graphs/'
print("root path:{}".format(root_path))
sys.path.append(root_path)
from tools.fit_line import compute_linear_fit,compute_multi_linear_fit

h_pacf_dwt_1 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_dwt_1 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_dwt_3 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_dwt_3 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_3_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_dwt_5 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_dwt_5 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_5_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_dwt_7 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_dwt_7 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_7_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_dwt_9 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_9_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_dwt_9 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_9_ahead_forecast_pcc_local/optimal_model_results.csv')

h_pacf_eemd_1 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_eemd_1 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_1_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_eemd_3 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_eemd_3 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_3_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_eemd_5 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_eemd_5 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_5_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_eemd_7 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_eemd_7 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_7_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_eemd_9 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_9_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_eemd_9 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_9_ahead_forecast_pcc_local/optimal_model_results.csv')

h_pacf_ssa_1 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_ssa_1 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_1_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_ssa_3 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_ssa_3 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_3_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_ssa_5 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_ssa_5 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_5_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_ssa_7 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_ssa_7 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_7_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_ssa_9 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_9_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_ssa_9 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_9_ahead_forecast_pcc_local/optimal_model_results.csv')

h_pacf_vmd_1 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_vmd_1 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_vmd_3 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_3_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_vmd_3 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_3_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_vmd_5 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_5_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_vmd_5 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_5_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_vmd_7 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_7_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_vmd_7 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_7_ahead_forecast_pcc_local/optimal_model_results.csv')
h_pacf_vmd_9 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_9_ahead_forecast_pacf/optimal_model_results.csv')
h_pcc_vmd_9 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_9_ahead_forecast_pcc_local/optimal_model_results.csv')

pacf_data=[
    [h_pacf_dwt_1,h_pacf_dwt_3,h_pacf_dwt_5,h_pacf_dwt_7],
    [h_pacf_eemd_1,h_pacf_eemd_3,h_pacf_eemd_5,h_pacf_eemd_7],
    [h_pacf_ssa_1,h_pacf_ssa_3,h_pacf_ssa_5,h_pacf_ssa_7],
    [h_pacf_vmd_1,h_pacf_vmd_3,h_pacf_vmd_5,h_pacf_vmd_7],
]

pcc_data=[
    [h_pcc_dwt_1,h_pcc_dwt_3,h_pcc_dwt_5,h_pcc_dwt_7],
    [h_pcc_eemd_1,h_pcc_eemd_3,h_pcc_eemd_5,h_pcc_eemd_7],
    [h_pcc_ssa_1,h_pcc_ssa_3,h_pcc_ssa_5,h_pcc_ssa_7],
    [h_pcc_vmd_1,h_pcc_vmd_3,h_pcc_vmd_5,h_pcc_vmd_7],
]
records_list=[]
preds_list=[]
for i in range(len(pacf_data)):
    for j in range(len(pacf_data[i])):
        pacf_records = pacf_data[i][j]['test_y'][0:120].values
        pacf_preds = pacf_data[i][j]['test_pred'][0:120].values
        pcc_records = pcc_data[i][j]['test_y'][0:120].values
        pcc_preds = pcc_data[i][j]['test_pred'][0:120].values
        records_list.append(pacf_records)
        records_list.append(pcc_records)
        preds_list.append(pacf_preds)
        preds_list.append(pcc_preds)    
xx,linear_list,xymin,xymax=compute_multi_linear_fit(
    records=records_list,
    predictions=preds_list,
)
models=[
    'DWT-SVR\n(1-month ahead)',
    'DWT-SVR\n(3-month ahead)',
    'DWT-SVR\n(5-month ahead)',
    'DWT-SVR\n(7-month ahead)',
    'EEMD-SVR\n(1-month ahead)',
    'EEMD-SVR\n(3-month ahead)',
    'EEMD-SVR\n(5-month ahead)',
    'EEMD-SVR\n(7-month ahead)',
    'SSA-SVR\n(1-month ahead)',
    'SSA-SVR\n(3-month ahead)',
    'SSA-SVR\n(5-month ahead)',
    'SSA-SVR\n(7-month ahead)',
    'VMD-SVR\n(1-month ahead)',
    'VMD-SVR\n(3-month ahead)',
    'VMD-SVR\n(5-month ahead)',
    'VMD-SVR\n(7-month ahead)',
]
print(len(linear_list))
plt.figure(figsize=(7.48,7.68))
for i in range(len(pacf_data)):
    for j in range(len(pacf_data[i])):
        #i=0,j=0,p=1;i=0;j=1;p=2;i=0,j=2,p=3;i=0,j=3,p=4;
        #i=1,j=0,p=5;i=1,j=1,p=6;i=1,j=2,p=7;i=1,j=0,p=8;p=4*i+j+1
        p=4*i+j+1
        ax=plt.subplot(len(pacf_data),len(pacf_data[i]),p,aspect='equal')
        print(p-1)
        if p==12:
            plt.text(2,30,models[p-1])
        else:
            plt.text(20,2,models[p-1])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if 4*i+j+1 in range(13,17):
            plt.xlabel('Predictions(' + r'$10^8m^3$' +')', )
        else:
            plt.xticks([])
        if 4*i+j+1 in [1,5,9,13]:
            plt.ylabel('Records(' + r'$10^8m^3$' + ')', )
        else:
            plt.yticks([])
        
        # p=1,k1=0,k2=1;k1=2*p-2,k2=2*p-1
        # p=2,k1=2,k2=3;
        # p=3,k1=4,k2=5;
        # p=4,k1=6,k2=7,
        k1 = 2*p-2
        k2 = 2*p-1
        plt.scatter(preds_list[k1], records_list[k1],marker='o',zorder=2,label='PACF')
        plt.scatter(preds_list[k2], records_list[k2],marker='v',zorder=1,label='PCC')
        plt.plot(xx, linear_list[k1], '--', label='PACF',zorder=2)
        plt.plot(xx, linear_list[k2], '-.', label='PCC',zorder=1)
        plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',zorder=0)
        plt.xlim([xymin,xymax])
        plt.ylim([xymin,xymax])
        if p==1:
            plt.legend(
                loc='upper left',
                # bbox_to_anchor=(0.08,1.01, 1,0.101),
                bbox_to_anchor=(1.16,1.15),
                ncol=5,
                shadow=False,
                frameon=True,
                )
# plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99,top=0.97, hspace=0.05, wspace=0.05)
plt.savefig(graphs_path+'Scatters of TSDP based on PACF and PCC at Huaxian.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'Scatters of TSDP based on PACF and PCC at Huaxian.tif',format='TIFF',dpi=500)
plt.savefig(graphs_path+'Scatters of TSDP based on PACF and PCC at Huaxian.pdf',format='PDF',dpi=1200)
plt.show()

# plt.figure(figsize=(7.48,7.48))
# for i in range(len(pacf_data)):
#     for j in range(len(pacf_data[i])):
#         #i=0,j=0,p=1;i=0;j=1;p=2;i=0,j=2,p=3;i=0,j=3,p=4;
#         #i=1,j=0,p=5;i=1,j=1,p=6;i=1,j=2,p=7;i=1,j=0,p=8;p=4*i+j+1
#         ax=plt.subplot(len(pacf_data),len(pacf_data[i]),4*i+j+1,aspect='equal')
#         pacf_records = pacf_data[i][j]['test_y'][0:120].values
#         pacf_preds = pacf_data[i][j]['test_pred'][0:120].values
#         pcc_records = pcc_data[i][j]['test_y'][0:120].values
#         pcc_preds = pcc_data[i][j]['test_pred'][0:120].values
#         records_list=[pacf_records,pcc_records]
#         preds_list=[pacf_preds,pcc_preds]
#         xx,linear_list,xymin,xymax=compute_multi_linear_fit(
#             records_list=records_list,
#             predictions_list=preds_list,
#         )
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#         if 4*i+j+1 in range(13,17):
#             plt.xlabel('Predictions(' + r'$10^8m^3$' +')', )
#         if 4*i+j+1 in [1,5,9,13]:
#             plt.ylabel('Records(' + r'$10^8m^3$' + ')', )
#         models=['PACF','PCC']
#         markers=['o','o',]
#         zorders=[2,1]
#         for k in range(len(preds_list)):
#             plt.scatter(preds_list[k], records_list[k],marker=markers[k],zorder=zorders[k])
#             plt.plot(xx, linear_list[k], '--', label=models[k],zorder=zorders[k])
#         plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',zorder=0)
#         plt.xlim([xymin,xymax])
#         plt.ylim([xymin,xymax])
#         plt.legend(
#                 loc=0,
#                 # bbox_to_anchor=(0.08,1.01, 1,0.101),
#                 # bbox_to_anchor=(1,1),
#                 # ncol=2,
#                 shadow=False,
#                 frameon=False,
#                 fontsize=6,
#                 )
# # plt.tight_layout()
# plt.subplots_adjust(left=0.06, bottom=0.08, right=0.99,top=0.99, hspace=0.1, wspace=0.15)
# plt.show()