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

h_dwt_1 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_dwta_1 = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
h_eemd_1 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_eemda_1 = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
h_ssa_1 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_ssaa_1 = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
h_vmd_1 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_vmda_1 = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')

x_dwt_1 = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_dwta_1 = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
x_eemd_1 = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_eemda_1 = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
x_ssa_1 = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_ssaa_1 = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
x_vmd_1 = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_vmda_1 = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')

z_dwt_1 = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_dwta_1 = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
z_eemd_1 = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_eemda_1 = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
z_ssa_1 = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_ssaa_1 = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
z_vmd_1 = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_vmda_1 = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')


h_records={
    'EEMD-SVR':h_eemd_1['test_y'][0:120].values,
    'SSA-SVR':h_ssa_1['test_y'][0:120].values,
    'DWT-SVR':h_dwt_1['test_y'][0:120].values,
    'VMD-SVR':h_vmd_1['test_y'][0:120].values,
    'EEMD-SVR-A':h_eemda_1['test_y'][0:120].values,
    'SSA-SVR-A':h_ssaa_1['test_y'][0:120].values,
    'DWT-SVR-A':h_dwta_1['test_y'][0:120].values,
    'VMD-SVR-A':h_vmda_1['test_y'][0:120].values,
}

h_preds={
    'EEMD-SVR':h_eemd_1['test_pred'][0:120].values,
    'SSA-SVR':h_ssa_1['test_pred'][0:120].values,
    'DWT-SVR':h_dwt_1['test_pred'][0:120].values,
    'VMD-SVR':h_vmd_1['test_pred'][0:120].values,
    'EEMD-SVR-A':h_eemda_1['test_pred'][0:120].values,
    'SSA-SVR-A':h_ssaa_1['test_pred'][0:120].values,
    'DWT-SVR-A':h_dwta_1['test_pred'][0:120].values,
    'VMD-SVR-A':h_vmda_1['test_pred'][0:120].values,
}


x_records={
    'EEMD-SVR':x_eemd_1['test_y'][0:120].values,
    'SSA-SVR':x_ssa_1['test_y'][0:120].values,
    'DWT-SVR':x_dwt_1['test_y'][0:120].values,
    'VMD-SVR':x_vmd_1['test_y'][0:120].values,
    'EEMD-SVR-A':x_eemda_1['test_y'][0:120].values,
    'SSA-SVR-A':x_ssaa_1['test_y'][0:120].values,
    'DWT-SVR-A':x_dwta_1['test_y'][0:120].values,
    'VMD-SVR-A':x_vmda_1['test_y'][0:120].values,
}

x_preds={
    'EEMD-SVR':x_eemd_1['test_pred'][0:120].values,
    'SSA-SVR':x_ssa_1['test_pred'][0:120].values,
    'DWT-SVR':x_dwt_1['test_pred'][0:120].values,
    'VMD-SVR':x_vmd_1['test_pred'][0:120].values,
    'EEMD-SVR-A':x_eemda_1['test_pred'][0:120].values,
    'SSA-SVR-A':x_ssaa_1['test_pred'][0:120].values,
    'DWT-SVR-A':x_dwta_1['test_pred'][0:120].values,
    'VMD-SVR-A':x_vmda_1['test_pred'][0:120].values,
}

z_records={
    'EEMD-SVR':z_eemd_1['test_y'][0:120].values,
    'SSA-SVR':z_ssa_1['test_y'][0:120].values,
    'DWT-SVR':z_dwt_1['test_y'][0:120].values,
    'VMD-SVR':z_vmd_1['test_y'][0:120].values,
    'EEMD-SVR-A':z_eemda_1['test_y'][0:120].values,
    'SSA-SVR-A':z_ssaa_1['test_y'][0:120].values,
    'DWT-SVR-A':z_dwta_1['test_y'][0:120].values,
    'VMD-SVR-A':z_vmda_1['test_y'][0:120].values,
}

z_preds={
    'EEMD-SVR':z_eemd_1['test_pred'][0:120].values,
    'SSA-SVR':z_ssa_1['test_pred'][0:120].values,
    'DWT-SVR':z_dwt_1['test_pred'][0:120].values,
    'VMD-SVR':z_vmd_1['test_pred'][0:120].values,
    'EEMD-SVR-A':z_eemda_1['test_pred'][0:120].values,
    'SSA-SVR-A':z_ssaa_1['test_pred'][0:120].values,
    'DWT-SVR-A':z_dwta_1['test_pred'][0:120].values,
    'VMD-SVR-A':z_vmda_1['test_pred'][0:120].values,
}



models=[
    [
        ['EEMD-SVR','EEMD-SVR-A'],
        ['SSA-SVR','SSA-SVR-A'],
        ['DWT-SVR','DWT-SVR-A'],
        ['VMD-SVR','VMD-SVR-A'],
    ],
    [
        ['EEMD-SVR','EEMD-SVR-A'],
        ['SSA-SVR','SSA-SVR-A'],
        ['DWT-SVR','DWT-SVR-A'],
        ['VMD-SVR','VMD-SVR-A'],
    ],
    [
        ['EEMD-SVR','EEMD-SVR-A'],
        ['SSA-SVR','SSA-SVR-A'],
        ['DWT-SVR','DWT-SVR-A'],
        ['VMD-SVR','VMD-SVR-A'],
    ],
]

records_lists=[h_records,x_records,z_records]
preds_lists=[h_preds,x_preds,z_preds]
markers=['o','v']
zorders1=[1,0]
zorders2=[0,1]
linestyles=['--','-.']
plt.figure(figsize=(7.48,6.48))
x=[29,19,4.6]
y=[1.3,0.8,0.2]
figid=[
    ['(a1)','(a2)','(a3)','(a4)',],
    ['(b1)','(b2)','(b3)','(b4)',],
    ['(c1)','(c2)','(c3)','(c4)',],
]
for i in range(len(models)):
    records_list=records_lists[i]
    preds_list=preds_lists[i]
    # print(records_list)
    xx,linear_list,xymin,xymax=compute_multi_linear_fit(
        records=records_list,
        predictions=preds_list,
    )
    # print(linear_list)
    for j in range(len(models[i])):
        #i=0,j=0,p=1;i=0;j=1;p=2;i=0,j=2,p=3;i=0,j=3,p=4;
        #i=1,j=0,p=5;i=1,j=1,p=6;i=1,j=2,p=7;i=1,j=0,p=8;p=4*i+j+1
        p=4*i+j+1
        ax=plt.subplot(len(models),len(models[i]),p,aspect='equal')
        plt.text(x[i],y[i],figid[i][j])
        # print(p-1)
        # if p==12:
        #     plt.text(2,30,models[p-1])
        # else:
        #     plt.text(20,2,models[p-1])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # if 4*i+j+1 in range(9,13):
        plt.xlabel('Predictions(' + r'$10^8m^3$' +')', )
        # else:
        #     plt.xticks([])
        if 4*i+j+1 in [1,5,9,]:
            plt.ylabel('Records(' + r'$10^8m^3$' + ')', )
        else:
            plt.yticks([])
        
        # p=1,k1=0,k2=1;k1=2*p-2,k2=2*p-1
        # p=2,k1=2,k2=3;
        # p=3,k1=4,k2=5;
        # p=4,k1=6,k2=7,
        k1 = 2*p-2
        k2 = 2*p-1
        for k in range(len(models[i][j])):
            print(models[i][j][k])
            if models[i][j][k]=='DWT-SVR' or models[i][j][k]=='DWT-SVR-A':
                plt.scatter(preds_list[models[i][j][k]], records_list[models[i][j][k]],marker=markers[k],zorder=zorders2[k],label='')
                plt.plot(xx, linear_list[models[i][j][k]], linestyles[k], label=models[i][j][k],zorder=zorders2[k])
            else:
                plt.scatter(preds_list[models[i][j][k]], records_list[models[i][j][k]],marker=markers[k],zorder=zorders1[k],label='')
                plt.plot(xx, linear_list[models[i][j][k]], linestyles[k], label=models[i][j][k],zorder=zorders1[k])
        plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',zorder=0)
        plt.xlim([xymin,xymax])
        plt.ylim([xymin,xymax])
        plt.legend()
        # if p==1:
        #     plt.legend(
        #         loc='upper left',
        #         # bbox_to_anchor=(0.08,1.01, 1,0.101),
        #         bbox_to_anchor=(1.16,1.15),
        #         ncol=5,
        #         shadow=False,
        #         frameon=True,
        #         )
# plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99,top=0.99, hspace=0.05, wspace=0.05)
plt.savefig(graphs_path+'Scatters of TSDP and TSDPE models.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'Scatters of TSDP and TSDPE models.tif',format='TIFF',dpi=500)
plt.savefig(graphs_path+'Scatters of TSDP and TSDPE models.pdf',format='PDF',dpi=1200)
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