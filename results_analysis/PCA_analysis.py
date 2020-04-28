import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
# plt.rcParams['lines.markersize']=7
plt.rcParams['lines.linewidth'] = 0.8
from sklearn import decomposition
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)

def cumul_var_ratio(var_ratio):
    sum=0.0
    cumul_var_ratio=[]
    for i in range(len(var_ratio)):
        sum=sum+var_ratio[i]
        cumul_var_ratio.append(sum)
    return cumul_var_ratio

samples_setss=[
    [
        pd.read_csv(root_path+'/Huaxian_dwt/data/db10-2/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Huaxian_eemd/data/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Huaxian_modwt/data/db10-2/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Huaxian_ssa/data/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_forecast_pacf/train_samples.csv'),
    ],
    [
        pd.read_csv(root_path+'/Xianyang_dwt/data/db10-2/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Xianyang_eemd/data/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Xianyang_modwt/data/db10-2/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Xianyang_ssa/data/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Xianyang_vmd/data/one_step_1_ahead_forecast_pacf/train_samples.csv'),
    ],
    [
        pd.read_csv(root_path+'/Zhangjiashan_dwt/data/db10-2/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Zhangjiashan_eemd/data/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Zhangjiashan_modwt/data/db10-2/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Zhangjiashan_ssa/data/one_step_1_ahead_forecast_pacf/train_samples.csv'),
        pd.read_csv(root_path+'/Zhangjiashan_vmd/data/one_step_1_ahead_forecast_pacf/train_samples.csv'),
    ],
]
decomposers=['DWT','EEMD','MODWT','SSA','VMD']
stations=['Huaxian','Xianyang','Zhangjiashan']
zorders=[4,3,2,1,0]
ini_pcs_dict={}
fig_file=[
    'CVR of different decomposers at Huaxian',
    'CVR of different decomposers at Xianyang',
    'CVR of different decomposers at Zhangjiashan',
]
colors=['b','g','r','c','m']
fig_idx=['(a)','(b)','(c)']
plt.figure(figsize=(7.48,3.6))
for k in range(len(samples_setss)):
    samples_sets = samples_setss[k]
    n_components=[#remove dimension of Y
        samples_sets[0].shape[1]-1,
        samples_sets[1].shape[1]-1,
        samples_sets[2].shape[1]-1,
        samples_sets[3].shape[1]-1,
        samples_sets[4].shape[1]-1,
    ]
    ini_pc={}
    plt.subplot(1,3,k+1)
    # plt.title(decomposers[i])
    plt.xlabel('Number of principle components(PCs)\n'+fig_idx[k])
    if k==0:
        plt.ylabel('Cumulative variance ratio(CVR)')
    else:
        plt.yticks([])
    # plt.title(stations[k])
    for i in range(len(samples_sets)):
        samples = samples_sets[i]
        y = samples['Y']
        X = samples.drop('Y',axis=1)
        print(X)
        print(n_components[i])
        pca = decomposition.PCA(n_components=n_components[i])
        pca.fit(X)
        var_ratio = pca.explained_variance_ratio_
        cum_var_ratio = cumul_var_ratio(var_ratio)
        print(cum_var_ratio)
        xx = 0
        for j in range(len(cum_var_ratio)):
            if cum_var_ratio[j]>=0.99:
                xx = j+1
                break
        ini_pc[decomposers[i]]=xx
        print('xx={}'.format(xx))
        yy1 = cum_var_ratio[xx-1]
        yy2 = var_ratio[xx-1]
        # plt.plot(range(1,n_components[i]+1),var_ratio,'-o',label=decomposers[i]+': VR',zorder=0)
        plt.plot(range(1,n_components[i]+1),cum_var_ratio,'-o',label=decomposers[i],zorder=zorders[i])
        plt.plot([xx],[yy1],marker='+',zorder=10,
        label=decomposers[i]
        )
        # if i==len(samples_sets)-1:
        #     plt.plot(xx,yy1,c='black',marker='+',label='Initial number of PCs with CVR larger than 0.99',zorder=10)
        # else:
        #     plt.plot(xx,yy1,c='black',marker='+',label='',zorder=10)
        plt.xlim(0,30)
    if k==1:
        plt.legend(
                    loc='upper center',
                    # bbox_to_anchor=(0.08,1.01, 1,0.101),
                    bbox_to_anchor=(0.5,1.09),
                    ncol=10,
                    shadow=False,
                    frameon=True,
                    )
    ini_pcs_dict[stations[k]]=ini_pc
# plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.13, right=0.99,top=0.92, hspace=0.2, wspace=0.05)
plt.savefig(root_path+'/graphs/CVR of PCs.tif',format='TIFF',dpi=1200)
plt.savefig(root_path+'/graphs/CVR of PCs.pdf',format='PDF',dpi=1200)
plt.savefig(root_path+'/graphs/CVR of PCs.eps',format='eps',dpi=2000)

ini_pcs_df = pd.DataFrame(ini_pcs_dict)
print(ini_pcs_df)
ini_pcs_df.to_csv(root_path+'/results_analysis/results/ini_pcs.csv')

n_components_sets=[
    [28,23,27,18,14],
    [28,23,27,18,14],
    [28,22,27,18,12],
]
plt.figure(figsize=(7.48,3.6))
for k in range(len(samples_setss)):
    samples_sets = samples_setss[k]
    n_components=n_components_sets[k]
    ini_pc={}
    plt.subplot(1,3,k+1)
    # plt.title(decomposers[i])
    plt.xlabel('Number of principle components(PCs)\n'+fig_idx[k])
    if k==0:
        plt.ylabel('Cumulative variance ratio(CVR)')
    else:
        plt.yticks([])
    # plt.title(stations[k])
    for i in range(len(samples_sets)):
        samples = samples_sets[i]
        y = samples['Y']
        X = samples.drop('Y',axis=1)
        print(X)
        print(n_components[i])
        pca = decomposition.PCA(n_components=n_components[i])
        pca.fit(X)
        var_ratio = pca.explained_variance_ratio_
        cum_var_ratio = cumul_var_ratio(var_ratio)
        print(cum_var_ratio)
        xx = 0
        for j in range(len(cum_var_ratio)):
            if cum_var_ratio[j]>=0.99:
                xx = j+1
                break
        ini_pc[decomposers[i]]=xx
        print('xx={}'.format(xx))
        yy1 = cum_var_ratio[xx-1]
        yy2 = var_ratio[xx-1]
        # plt.plot(range(1,n_components[i]+1),var_ratio,'-o',label=decomposers[i]+': VR',zorder=0)
        plt.plot(range(1,n_components[i]+1),cum_var_ratio,'-o',label=decomposers[i],zorder=zorders[i])
        plt.plot([xx],[yy1],marker='+',zorder=10,
        label=decomposers[i]
        )
        # if i==len(samples_sets)-1:
        #     plt.plot(xx,yy1,c='black',marker='+',label='Initial number of PCs with CVR larger than 0.99',zorder=10)
        # else:
        #     plt.plot(xx,yy1,c='black',marker='+',label='',zorder=10)
        plt.xlim(0,30)
    if k==1:
        plt.legend(
                    loc='upper center',
                    # bbox_to_anchor=(0.08,1.01, 1,0.101),
                    bbox_to_anchor=(0.5,1.09),
                    ncol=10,
                    shadow=False,
                    frameon=True,
                    )
    ini_pcs_dict[stations[k]]=ini_pc
# plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.13, right=0.99,top=0.92, hspace=0.2, wspace=0.05)
plt.show()