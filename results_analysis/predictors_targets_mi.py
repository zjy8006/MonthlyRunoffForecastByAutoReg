import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
hfont = {'fontname':'Helvetica'}
# plt.rcParams['figure.figsize']=(10,8)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans','Lucida Grande', 'Verdana']
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['image.cmap']='Purples'
# plt.rcParams['axes.linewidth']=0.8
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
graphs_path = root_path+'/graphs/'
import sys
sys.path.append(root_path)

h_modwt_1 = pd.read_csv(root_path+'/Huaxian_modwt/data-wddff/db1-4/single_hybrid_1_ahead_lag12_mi_ts0.1/cal_samples.csv') 
h_modwt_3 = pd.read_csv(root_path+'/Huaxian_modwt/data-wddff/db1-4/single_hybrid_3_ahead_lag12_mi_ts0.1/cal_samples.csv') 
h_modwt_5 = pd.read_csv(root_path+'/Huaxian_modwt/data-wddff/db1-4/single_hybrid_5_ahead_lag12_mi_ts0.1/cal_samples.csv') 
h_modwt_7 = pd.read_csv(root_path+'/Huaxian_modwt/data-wddff/db1-4/single_hybrid_7_ahead_lag12_mi_ts0.1/cal_samples.csv') 

h_dwt_1 = pd.read_csv(root_path+'/Huaxian_dwt/data/db10-2/one_step_1_ahead_forecast_pacf/train_samples.csv') 
h_dwt_3 = pd.read_csv(root_path+'/Huaxian_dwt/data/db10-2/one_step_3_ahead_forecast_pacf/train_samples.csv') 
h_dwt_5 = pd.read_csv(root_path+'/Huaxian_dwt/data/db10-2/one_step_5_ahead_forecast_pacf/train_samples.csv') 
h_dwt_7 = pd.read_csv(root_path+'/Huaxian_dwt/data/db10-2/one_step_7_ahead_forecast_pacf/train_samples.csv') 

h_vmd_1 = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_forecast_pacf/train_samples.csv') 
h_vmd_3 = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_3_ahead_forecast_pacf/train_samples.csv') 
h_vmd_5 = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_5_ahead_forecast_pacf/train_samples.csv') 
h_vmd_7 = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_7_ahead_forecast_pacf/train_samples.csv') 

h_eemd_1 = pd.read_csv(root_path+'/Huaxian_eemd/data/one_step_1_ahead_forecast_pacf/train_samples.csv') 
h_eemd_3 = pd.read_csv(root_path+'/Huaxian_eemd/data/one_step_3_ahead_forecast_pacf/train_samples.csv') 
h_eemd_5 = pd.read_csv(root_path+'/Huaxian_eemd/data/one_step_5_ahead_forecast_pacf/train_samples.csv') 
h_eemd_7 = pd.read_csv(root_path+'/Huaxian_eemd/data/one_step_7_ahead_forecast_pacf/train_samples.csv') 

h_ssa_1 = pd.read_csv(root_path+'/Huaxian_ssa/data/one_step_1_ahead_forecast_pacf/train_samples.csv') 
h_ssa_3 = pd.read_csv(root_path+'/Huaxian_ssa/data/one_step_3_ahead_forecast_pacf/train_samples.csv') 
h_ssa_5 = pd.read_csv(root_path+'/Huaxian_ssa/data/one_step_5_ahead_forecast_pacf/train_samples.csv') 
h_ssa_7 = pd.read_csv(root_path+'/Huaxian_ssa/data/one_step_7_ahead_forecast_pacf/train_samples.csv') 

decom_samples=[
    [h_eemd_1,h_eemd_3,h_eemd_5,h_eemd_7],
    [h_ssa_1,h_ssa_3,h_ssa_5,h_ssa_7],
    [h_vmd_1,h_vmd_3,h_vmd_5,h_vmd_7],
    [h_dwt_1,h_dwt_3,h_dwt_5,h_dwt_7],
    [h_modwt_1,h_modwt_3,h_modwt_5,h_modwt_7],
]

labels=['1-month ahead','3-month ahead','5-month ahead','7-month ahead']
decomposers=['(a)EEMD','(b)SSA','(c)VMD','(d)DWT','(e)MODWT']
mis = []
linestyle=['-','--','-.',':']
plt.figure(figsize=(7.48,2.))
xx=[45,40,21,42,8.5]
yy=[0.2,0.3,0.264,0.42,0.35]

for i in range(len(decom_samples)):
    samples=decom_samples[i]
    ax=plt.subplot(1,5,i+1)
    # ax.axhline(yy[i])
    ax.text(xx[i],yy[i],decomposers[i])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.ylim(-0.01,0.46)
    # plt.ylim(0,6.5)
    if i==0:
        ax.set_ylabel('Mutual Information')
    # else:
    #     ax.set_yticks([])
    # if i==2 or i==3:
    ax.set_xlabel('Predictors')
    for j in range(len(samples)):
        samp=samples[j]
        labl=labels[j]
        # print(samp)
        y = samp['Y']
        X = samp.drop('Y',axis=1) 
        mi = mutual_info_regression(X,y) 
        mis.append(mi)
        mi_df = pd.DataFrame(mi,index=X.columns,columns=['MI'])
        mi_df_sort = mi_df.sort_values(by='MI',ascending=False)
        mi_df_sort = mi_df_sort.reset_index(drop=True)
        mi_df_sum = mi_df_sort['MI'].sum()
        mi_rate = mi_df_sort/mi_df_sum
        # xlabels=list(mi_df_sort.index.values)
        # print(xlabels)
        # mi_df_sort = mi_df_sort['MI'].cumsum()
        ax.plot(mi_df_sort.values,linestyle[j],label=labl)

        # print(mi_df)
    if i==2:    
        plt.legend(
            loc="upper center",bbox_to_anchor=(0.5,1.17),ncol=4,columnspacing=8,handlelength=5
            )
# plt.tight_layout()
plt.subplots_adjust(left=0.065, bottom=0.20, right=0.99,top=0.90, hspace=0.15, wspace=0.3)
plt.savefig(graphs_path+'Mutual information between predictors and predicted targets at Huaxian.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'Mutual information between predictors and predicted targets at Huaxian.tif',format='TIFF',dpi=500)
plt.savefig(graphs_path+'Mutual information between predictors and predicted targets at Huaxian.pdf',format='PDF',dpi=1200)
plt.show()

# width = 0.8
# action = [-1,0,1,2]
# alphas=[1,0.75,0.5,0.25]
# plt.figure(figsize=(7.48,7.48))
# for k in range(len(decom_samples)):
#     samples=decom_samples[k]
#     ax=plt.subplot(4,1,k+1)
#     # plt.ylim(-0.01,0.46)
#     # if i==0 or i==2:
#     #     plt.ylabel('Mutual Information')
#     # else:
#     #     plt.yticks([])
#     # if i==2 or i==3:
#     #     plt.xlabel('Iterations')
#     f_import = []
#     pos=[]
#     for j in range(samples[0].shape[1]-1):
#         pos.append(2*(j+1))
#     for i in range(len(samples)):
#         samp = samples[i]
#         y = samp['Y']
#         X = samp.drop('Y',axis=1) 
#         model = GradientBoostingRegressor(n_estimators=100)
#         model.fit(X,y)
#         feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#         feat_importances = feat_importances.sort_values(ascending=False)
#         feat_importances = feat_importances.reset_index(drop=True)
#         print(feat_importances.sum())
#         xx=[p+action[i]*width for p in pos]
#         print('len(xx)={}'.format(len(xx)))
#         print('feat_importances.shape[0]={}'.format(feat_importances.shape[0]))
#         assert len(xx)==feat_importances.shape[0]
#         # bars = ax.bar(xx,feat_importances, width, alpha=0.75, label=labels[i])
#         bars = ax.bar(range(X.shape[1]),feat_importances, width, alpha=alphas[i], label=labels[i])
#     plt.legend()
#     # if i==0:    
#     #     plt.legend(
#     #         loc="upper left",bbox_to_anchor=(0.35,1.28),ncol=2
#     #         )
# plt.tight_layout()
# # plt.subplots_adjust(left=0.13, bottom=0.11, right=0.99,top=0.90, hspace=0.15, wspace=0.05)
# plt.show()

# pos=[]
# for i in range(mis[0].shape[0]):
#     pos.append(2*(i+1))
# width = 0.2
# action = [ -1, 0,1,2]
# fig = plt.figure(figsize=(7.48, 2.48))
# ax=plt.subplot(1,1,1)
# for j in range(len(mis)):
#     x=[p+action[j]*width for p in pos]
#     print(x)
#     bars = ax.bar([p+action[j]*width for p in pos],mis[j], width, alpha=0.75, label=labels[j])
# ax.legend()
# plt.show()

