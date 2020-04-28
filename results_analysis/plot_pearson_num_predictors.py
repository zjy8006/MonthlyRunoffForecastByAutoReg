import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [7.48, 5.61]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8

import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/graphs/'
print("root path:{}".format(root_path))
import sys
sys.path.append(root_path)
from tools.results_reader import read_samples_num

num_samples_eemd_hua = read_samples_num(station="Huaxian",decomposer="eemd",)
num_samples_ssa_hua = read_samples_num(station="Huaxian",decomposer="ssa",)
num_samples_vmd_hua = read_samples_num(station="Huaxian",decomposer="vmd",)
num_samples_dwt_hua = read_samples_num(station="Huaxian",decomposer="dwt",)
num_samples_modwt_hua = read_samples_num(station='Huaxian',decomposer='modwt')

num_samples_eemd_xian = read_samples_num(station="Xianyang",decomposer="eemd",)
num_samples_ssa_xian = read_samples_num(station="Xianyang",decomposer="ssa",)
num_samples_vmd_xian = read_samples_num(station="Xianyang",decomposer="vmd",)
num_samples_dwt_xian = read_samples_num(station="Xianyang",decomposer="dwt",)
num_samples_modwt_xian = read_samples_num(station="Xianyang",decomposer="modwt",)

num_samples_eemd_zhang = read_samples_num(station="Zhangjiashan",decomposer="eemd",)
num_samples_ssa_zhang = read_samples_num(station="Zhangjiashan",decomposer="ssa",)
num_samples_vmd_zhang = read_samples_num(station="Zhangjiashan",decomposer="vmd",)
num_samples_dwt_zhang = read_samples_num(station="Zhangjiashan",decomposer="dwt",)
num_samples_modwt_zhang = read_samples_num(station="Zhangjiashan",decomposer="modwt",)


one_month_hua_vmd = pd.read_csv(root_path+"/Huaxian_vmd/data/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_hua_vmd = one_month_hua_vmd.drop("Y",axis=1)
num_1_hua_vmd = one_month_hua_vmd.shape[1]

one_month_hua_eemd = pd.read_csv(root_path+"/Huaxian_eemd/data/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_hua_eemd = one_month_hua_eemd.drop("Y",axis=1)
num_1_hua_eemd = one_month_hua_eemd.shape[1]

one_month_hua_ssa = pd.read_csv(root_path+"/Huaxian_ssa/data/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_hua_ssa = one_month_hua_ssa.drop("Y",axis=1)
num_1_hua_ssa = one_month_hua_ssa.shape[1]

one_month_hua_dwt = pd.read_csv(root_path+"/Huaxian_dwt/data/db10-2/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_hua_dwt = one_month_hua_dwt.drop("Y",axis=1)
num_1_hua_dwt = one_month_hua_dwt.shape[1]

one_month_hua_modwt = pd.read_csv(root_path+"/Huaxian_modwt/data-wddff/db10-2/single_hybrid_1_ahead_mi_ts0.1/minmax_unsample_train.csv")
one_month_hua_modwt = one_month_hua_modwt.drop("Y",axis=1)
num_1_hua_modwt = one_month_hua_modwt.shape[1]

one_month_xian_vmd = pd.read_csv(root_path+"/Xianyang_vmd/data/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_xian_vmd = one_month_xian_vmd.drop("Y",axis=1)
num_1_xian_vmd = one_month_xian_vmd.shape[1]

one_month_xian_eemd = pd.read_csv(root_path+"/Xianyang_eemd/data/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_xian_eemd = one_month_xian_eemd.drop("Y",axis=1)
num_1_xian_eemd = one_month_xian_eemd.shape[1]

one_month_xian_ssa = pd.read_csv(root_path+"/Xianyang_ssa/data/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_xian_ssa = one_month_xian_ssa.drop("Y",axis=1)
num_1_xian_ssa = one_month_xian_ssa.shape[1]

one_month_xian_dwt = pd.read_csv(root_path+"/Xianyang_dwt/data/db10-2/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_xian_dwt = one_month_xian_dwt.drop("Y",axis=1)
num_1_xian_dwt = one_month_xian_dwt.shape[1]

one_month_xian_modwt = pd.read_csv(root_path+"/Xianyang_modwt/data-wddff/db10-2/single_hybrid_1_ahead_mi_ts0.1/minmax_unsample_train.csv")
one_month_xian_modwt = one_month_xian_modwt.drop("Y",axis=1)
num_1_xian_modwt = one_month_xian_modwt.shape[1]

one_month_zhang_vmd = pd.read_csv(root_path+"/Zhangjiashan_vmd/data/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_zhang_vmd = one_month_zhang_vmd.drop("Y",axis=1)
num_1_zhang_vmd = one_month_zhang_vmd.shape[1]

one_month_zhang_eemd = pd.read_csv(root_path+"/Zhangjiashan_eemd/data/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_zhang_eemd = one_month_zhang_eemd.drop("Y",axis=1)
num_1_zhang_eemd = one_month_zhang_eemd.shape[1]

one_month_zhang_ssa = pd.read_csv(root_path+"/Zhangjiashan_ssa/data/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_zhang_ssa = one_month_zhang_ssa.drop("Y",axis=1)
num_1_zhang_ssa = one_month_zhang_ssa.shape[1]

one_month_zhang_dwt = pd.read_csv(root_path+"/Zhangjiashan_dwt/data/db10-2/one_step_1_ahead_forecast_pacf/minmax_unsample_train.csv")
one_month_zhang_dwt = one_month_zhang_dwt.drop("Y",axis=1)
num_1_zhang_dwt = one_month_zhang_dwt.shape[1]

one_month_zhang_modwt = pd.read_csv(root_path+"/Zhangjiashan_modwt/data-wddff/db10-2/single_hybrid_1_ahead_mi_ts0.1/minmax_unsample_train.csv")
one_month_zhang_modwt = one_month_zhang_modwt.drop("Y",axis=1)
num_1_zhang_modwt = one_month_zhang_modwt.shape[1]

num_1_sets=[
    [
        num_1_hua_eemd,
        num_1_hua_ssa,
        num_1_hua_vmd,
        num_1_hua_dwt,
        num_1_hua_modwt,
    ],
    [
        num_1_xian_eemd,
        num_1_xian_ssa,
        num_1_xian_vmd,
        num_1_xian_dwt,
        num_1_xian_modwt,
    ],
    [
        num_1_zhang_eemd,
        num_1_zhang_ssa,
        num_1_zhang_vmd,
        num_1_zhang_dwt,
        num_1_zhang_modwt,
    ]
]

corrs_sets=[
    [
        num_samples_eemd_hua,
        num_samples_ssa_hua,
        num_samples_vmd_hua,
        num_samples_dwt_hua,
        num_samples_modwt_hua,
    ],
    [
        num_samples_eemd_xian,
        num_samples_ssa_xian,
        num_samples_vmd_xian,
        num_samples_dwt_xian,
        num_samples_modwt_xian,
    ],
    [
        num_samples_eemd_zhang,
        num_samples_ssa_zhang,
        num_samples_vmd_zhang,
        num_samples_dwt_zhang,
        num_samples_modwt_zhang,
    ],
]

lablels=[
    "3-month ahead",
    "5-month ahead",
    "7-month ahead",
    "9-month ahead",
]

titles=[
    "(a)EEMD",
    "(b)SSA",
    "(c)VMD",
    "(d)DWT",
    "(e)MODWT",
]
x_s=[
    [0.09,0.09,0.09,0.09,0.09],
    [0.09,0.09,0.09,0.09,0.09],
    [0.09,0.09,0.09,0.09,0.09],
]
y_s=[
    [5,5,5,5,1.2],
    [5,5,5,5,2],
    [5,5,5,5,2],
]
stations=['Huaxian','Xianyang','Zhangjiashan']
for k in range(len(corrs_sets)):
    num_1 = num_1_sets[k]
    corrs = corrs_sets[k]
    plt.figure(figsize=(3.54, 5.54))
    t= [0.1,0.2,0.3,0.4,0.5]
    x = x_s[k]
    y = y_s[k]
    plt.xticks(np.arange(start=0.1,stop=0.6,step=0.1))
    for i in range(len(corrs)):
        plt.subplot(len(corrs),1,i+1)
        if i==3:
            plt.ylim([0,70])
        plt.text(x[i],y[i],titles[i],fontsize=7)

        if i == len(corrs)-1:
            plt.xlabel("Threshold")
            plt.xticks([0.1,0.2,0.3,0.4,0.5])
        else:
            plt.xticks([])

        plt.ylabel("Number of predictors")
        plt.axhline(num_1[i],label='1-month ahead',color='black',linestyle='--')
        for j in range(len(corrs[i])):
            plt.plot(t,corrs[i][j],label=lablels[j])
        if i==0:
            plt.legend(
                loc='upper left',
                # loc=0,
                # bbox_to_anchor=(0.08,1.01, 1,0.101),
                bbox_to_anchor=(-0.02,1.39),
                # bbox_transform=plt.gcf().transFigure,
                ncol=3,
                shadow=False,
                frameon=True,)
    plt.subplots_adjust(left=0.125, bottom=0.07, right=0.98,top=0.94, hspace=0.05, wspace=0.2)
    plt.savefig(graphs_path+"Predictors num vs threshold at "+stations[k]+".eps",format="EPS",dpi=2000)
    plt.savefig(graphs_path+"Predictors num vs threshold at "+stations[k]+".tif",format="TIFF",dpi=1200)
    plt.savefig(graphs_path+"Predictors num vs threshold at "+stations[k]+".pdf",format="PDF",dpi=1200)
plt.show()
    
