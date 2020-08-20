import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
graphs_path = root_path+'\\graphs\\'
from tools.metrics_ import PPTS,mean_absolute_percentage_error
from tools.results_reader import read_pca_metrics

h_v=read_pca_metrics(station="Huaxian",decomposer="vmd",start_component=14,stop_component=30)
x_v=read_pca_metrics(station="Xianyang",decomposer="vmd",start_component=14,stop_component=30)
z_v=read_pca_metrics(station="Zhangjiashan",decomposer="vmd",start_component=10,stop_component=26)
h_e=read_pca_metrics(station="Huaxian",decomposer="eemd",start_component=53,stop_component=69)
x_e=read_pca_metrics(station="Xianyang",decomposer="eemd",start_component=52,stop_component=68)
z_e=read_pca_metrics(station="Zhangjiashan",decomposer="eemd",start_component=56,stop_component=72)
h_s=read_pca_metrics(station="Huaxian",decomposer="ssa",start_component=39,stop_component=55)
x_s=read_pca_metrics(station="Xianyang",decomposer="ssa",start_component=38,stop_component=54)
z_s=read_pca_metrics(station="Zhangjiashan",decomposer="ssa",start_component=34,stop_component=50)
h_w=read_pca_metrics(station="Huaxian",decomposer="dwt",start_component=44,stop_component=60)
x_w=read_pca_metrics(station="Xianyang",decomposer="dwt",start_component=44,stop_component=60)
z_w=read_pca_metrics(station="Zhangjiashan",decomposer="dwt",start_component=44,stop_component=60)


# data = [
#     [h_e,x_e,z_e],
#     [h_s,x_s,z_s],
#     [h_v,x_v,z_v],
#     [h_w,x_w,z_w],
# ]
# stations=['Huaxian','Xianyang','Zhangjiashan']
# markers = ['o','s','v']
# colors = ['tab:blue','tab:orange','tab:green']
# fig_idx=['(a)','(b)','(c)','(d)','(e)']
# x=[-0.6,-0.6,-0.6,-0.6,]
# y=[-1.15,-0.4,0.83,0.82,]
# plt.figure(figsize=(3.54,6.54))
# t=list(range(17))
# for i in range(len(data)):
#     ax=plt.subplot(5,1,i+1)
#     plt.text(x[i],y[i],fig_idx[i],fontweight='normal',fontsize=7)
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     if i==1:
#         plt.ylim(-0.5,1.1)
#     elif i==3:
#         plt.ylim(0.8,1.1)
#     plt.xticks(np.arange(0,17,1))
#     plt.ylabel(r'$NSE$')
#     if i==len(data)-1:
#         plt.xlabel('Number of reduced predictors')
#     for j in range(len(data[i])):
#         plt.plot(t,data[i][j]['nse'][1:],marker=markers[j],label=stations[j],markerfacecolor='w')
#         plt.axhline(data[i][j]['nse'][0],color=colors[j],label=stations[j]+':without PCA',linestyle='-',)
#         plt.axvline(data[i][j]['mle'],color=colors[j],label=stations[j]+':PCA MLE',linestyle='--',)
#     if i==0:
#         plt.legend(
#             loc='upper left',
#             # loc=0,
#             # bbox_to_anchor=(0.08,1.01, 1,0.101),
#             bbox_to_anchor=(0.015,1.9),
#             # bbox_transform=plt.gcf().transFigure,
#             ncol=2,
#             shadow=False,
#             frameon=True,
#         )
# plt.subplots_adjust(left=0.16, bottom=0.06, right=0.98,top=0.88, hspace=0.25, wspace=0.15)
# plt.savefig(graphs_path+"two_stage_pca_nse.eps",format="EPS",dpi=2000)
# plt.savefig(graphs_path+"two_stage_pca_nse.tif",format="TIFF",dpi=600)
# plt.show()


all_data = [
    [h_e, h_s,h_v,h_w,],
    [x_e, x_s,x_v,x_w,],
    [z_e, z_s,z_v,z_w,],
]
fig_idx=['EEMD-SVR','SSA-SVR','VMD-SVR','DWT-SVR',]
stations=['Huaxian','Xianyang','Zhangjiashan']
t=list(range(17))
# for k in range(len(all_data)):
#     plt.figure(figsize=(7.4861,5.48))
#     # ax1 = plt.subplot2grid((3,4), (0,0), colspan=2,)
#     # ax2 = plt.subplot2grid((3,4), (0,2), colspan=2,)
#     # ax3 = plt.subplot2grid((3,4), (1,0), colspan=2,)
#     # ax4 = plt.subplot2grid((3,4), (1,2), colspan=2,)
#     # ax5 = plt.subplot2grid((3,4), (2,1), colspan=2,)
#     # ax_list=[ax1,ax2,ax3,ax4,ax5]
#     station_data = all_data[k]
#     for i in range(len(station_data)):
#         # ax=ax_list[i]
#         ax=plt.subplot(2,2,i+1)
#         ax.set_title(fig_idx[i])
#         ax.set_ylim(-0.3,1)
#         # plt.text(x[i],y[i],fig_idx[i],fontweight='normal',fontsize=7)
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         # if i==1:
#         #     plt.ylim(-0.5,1.1)
#         # elif i==3:
#         #     plt.ylim(0.8,1.1)
#         ax.set_xticks(np.arange(0,17,1))
#         ax.set_ylabel(r'$NSE$')
#         if i==len(station_data)-1:
#             ax.set_xlabel('Number of reduced predictors')

#         ax.plot(t,station_data[i]['nse'][1:],color='b',marker='o',label='PCA',markerfacecolor='w')
#         ax.axhline(station_data[i]['nse'][0],color='r',label='Without PCA',linestyle='-',)
#         ax.axvline(station_data[i]['mle'],color='g',label='PCA MLE',linestyle='--',)
#         if i==4:
#             plt.legend(
#                 loc='upper left',
#                 # loc=0,
#                 # bbox_to_anchor=(0.08,1.01, 1,0.101),
#                 bbox_to_anchor=(1.05,0.9),
#                 # bbox_transform=plt.gcf().transFigure,
#                 ncol=1,
#                 shadow=False,
#                 frameon=True,
#             )
#     plt.subplots_adjust(left=0.08, bottom=0.06, right=0.98,top=0.96, hspace=0.25, wspace=0.4)
#     plt.savefig(graphs_path+"Two stage pca nse at "+stations[k]+".eps",format="EPS",dpi=2000)
#     plt.savefig(graphs_path+"Two stage pca nse at "+stations[k]+".tif",format="TIFF",dpi=1200)
#     plt.savefig(graphs_path+"Two stage pca nse at "+stations[k]+".pdf",format="PDF",dpi=1200)



nse_eemd_data = [
    [h_e['nse'][1],x_e['nse'][1],z_e['nse'][1]],
    [h_e['nse'][2],x_e['nse'][2],z_e['nse'][2]],
    [h_e['nse'][3],x_e['nse'][3],z_e['nse'][3]],
    [h_e['nse'][4],x_e['nse'][4],z_e['nse'][4]],
    [h_e['nse'][5],x_e['nse'][5],z_e['nse'][5]],
    [h_e['nse'][6],x_e['nse'][6],z_e['nse'][6]],
    [h_e['nse'][7],x_e['nse'][7],z_e['nse'][7]],
    [h_e['nse'][8],x_e['nse'][8],z_e['nse'][8]],
    [h_e['nse'][9],x_e['nse'][9],z_e['nse'][9]],
    [h_e['nse'][10],x_e['nse'][10],z_e['nse'][10]],
    [h_e['nse'][11],x_e['nse'][11],z_e['nse'][11]],
    [h_e['nse'][12],x_e['nse'][12],z_e['nse'][12]],
    [h_e['nse'][13],x_e['nse'][13],z_e['nse'][13]],
    [h_e['nse'][14],x_e['nse'][14],z_e['nse'][14]],
    [h_e['nse'][15],x_e['nse'][15],z_e['nse'][15]],
    [h_e['nse'][16],x_e['nse'][16],z_e['nse'][16]],
    [h_e['nse'][17],x_e['nse'][17],z_e['nse'][17]],
]

nse_ssa_data = [
    [h_s['nse'][1],x_s['nse'][1],z_s['nse'][1]],
    [h_s['nse'][2],x_s['nse'][2],z_s['nse'][2]],
    [h_s['nse'][3],x_s['nse'][3],z_s['nse'][3]],
    [h_s['nse'][4],x_s['nse'][4],z_s['nse'][4]],
    [h_s['nse'][5],x_s['nse'][5],z_s['nse'][5]],
    [h_s['nse'][6],x_s['nse'][6],z_s['nse'][6]],
    [h_s['nse'][7],x_s['nse'][7],z_s['nse'][7]],
    [h_s['nse'][8],x_s['nse'][8],z_s['nse'][8]],
    [h_s['nse'][9],x_s['nse'][9],z_s['nse'][9]],
    [h_s['nse'][10],x_s['nse'][10],z_s['nse'][10]],
    [h_s['nse'][11],x_s['nse'][11],z_s['nse'][11]],
    [h_s['nse'][12],x_s['nse'][12],z_s['nse'][12]],
    [h_s['nse'][13],x_s['nse'][13],z_s['nse'][13]],
    [h_s['nse'][14],x_s['nse'][14],z_s['nse'][14]],
    [h_s['nse'][15],x_s['nse'][15],z_s['nse'][15]],
    [h_s['nse'][16],x_s['nse'][16],z_s['nse'][16]],
    [h_s['nse'][17],x_s['nse'][17],z_s['nse'][17]],
]

nse_dwt_data = [
    [h_w['nse'][1],x_w['nse'][1],z_w['nse'][1]],
    [h_w['nse'][2],x_w['nse'][2],z_w['nse'][2]],
    [h_w['nse'][3],x_w['nse'][3],z_w['nse'][3]],
    [h_w['nse'][4],x_w['nse'][4],z_w['nse'][4]],
    [h_w['nse'][5],x_w['nse'][5],z_w['nse'][5]],
    [h_w['nse'][6],x_w['nse'][6],z_w['nse'][6]],
    [h_w['nse'][7],x_w['nse'][7],z_w['nse'][7]],
    [h_w['nse'][8],x_w['nse'][8],z_w['nse'][8]],
    [h_w['nse'][9],x_w['nse'][9],z_w['nse'][9]],
    [h_w['nse'][10],x_w['nse'][10],z_w['nse'][10]],
    [h_w['nse'][11],x_w['nse'][11],z_w['nse'][11]],
    [h_w['nse'][12],x_w['nse'][12],z_w['nse'][12]],
    [h_w['nse'][13],x_w['nse'][13],z_w['nse'][13]],
    [h_w['nse'][14],x_w['nse'][14],z_w['nse'][14]],
    [h_w['nse'][15],x_w['nse'][15],z_w['nse'][15]],
    [h_w['nse'][16],x_w['nse'][16],z_w['nse'][16]],
    [h_w['nse'][17],x_w['nse'][17],z_w['nse'][17]],
]

nse_vmd_data = [
    [h_v['nse'][1],x_v['nse'][1],z_v['nse'][1]],
    [h_v['nse'][2],x_v['nse'][2],z_v['nse'][2]],
    [h_v['nse'][3],x_v['nse'][3],z_v['nse'][3]],
    [h_v['nse'][4],x_v['nse'][4],z_v['nse'][4]],
    [h_v['nse'][5],x_v['nse'][5],z_v['nse'][5]],
    [h_v['nse'][6],x_v['nse'][6],z_v['nse'][6]],
    [h_v['nse'][7],x_v['nse'][7],z_v['nse'][7]],
    [h_v['nse'][8],x_v['nse'][8],z_v['nse'][8]],
    [h_v['nse'][9],x_v['nse'][9],z_v['nse'][9]],
    [h_v['nse'][10],x_v['nse'][10],z_v['nse'][10]],
    [h_v['nse'][11],x_v['nse'][11],z_v['nse'][11]],
    [h_v['nse'][12],x_v['nse'][12],z_v['nse'][12]],
    [h_v['nse'][13],x_v['nse'][13],z_v['nse'][13]],
    [h_v['nse'][14],x_v['nse'][14],z_v['nse'][14]],
    [h_v['nse'][15],x_v['nse'][15],z_v['nse'][15]],
    [h_v['nse'][16],x_v['nse'][16],z_v['nse'][16]],
    [h_v['nse'][17],x_v['nse'][17],z_v['nse'][17]],
]

nse_eemd_avg=(h_e['nse'][0]+x_e['nse'][0]+z_e['nse'][0])/3
nse_ssa_avg=(h_s['nse'][0]+x_s['nse'][0]+z_s['nse'][0])/3
nse_dwt_avg=(h_w['nse'][0]+x_w['nse'][0]+z_w['nse'][0])/3
nse_vmd_avg=(h_v['nse'][0]+x_v['nse'][0]+z_v['nse'][0])/3

nse_data=[nse_eemd_data,nse_ssa_data,nse_dwt_data,nse_vmd_data]
nse_avg_without_pca=[nse_eemd_avg,nse_ssa_avg,nse_dwt_avg,nse_vmd_avg]

mle=[
    [h_e['mle'],x_e['mle'],x_e['mle'],],
    [h_s['mle'],x_s['mle'],x_s['mle'],],
    [h_w['mle'],x_w['mle'],x_w['mle'],],
    [h_v['mle'],x_v['mle'],x_v['mle'],],
]
models=['EEMD-SVR','SSA-SVR','DWT-SVR','VMD-SVR']
fig_id=['(a)','(b)','(c)','(d)']
stations=['Huaxian','Xianyang','Zhangjiashan']
linestyles=['--','-.',':']
colors=['b','g','c']
# plt.figure(figsize=(7.48,4.48))
plt.figure(figsize=(7.48,2.0))
for i in range(len(nse_data)):
    plt.subplot(1,4,i+1)
    if i==0:
        plt.text(-0.5,0.8,fig_id[i]+' '+models[i])
    else:
        plt.text(11,-1.15,fig_id[i]+' '+models[i])
    if i in [0]:
        plt.ylabel(r'$NSE$')
    else:
        plt.yticks([])
    plt.ylim(-1.2,1.1)
    plt.xlabel('Number of excluded predictors')
    # if i==len(nse_data)-1 or i==len(nse_data)-2:
    #     plt.xticks(np.arange(0,17,1))
    #     plt.xlabel('Number of reduced predictors')
    # else:
    #     plt.xticks([])
    plt.violinplot(
            dataset=nse_data[i],
            positions=list(range(len(nse_data[i]))),
            showmeans=True,
        )
    plt.axhline(nse_avg_without_pca[i],color='r',label='Mean NSE without PCA',linestyle='-',)
    for j in range(len(stations)):
        plt.axvline(mle[i][j],color=colors[j],label='PCA MLE of '+stations[j],linestyle=linestyles[j],lw=0.8)
    if i==0:
        plt.legend(loc="upper left",bbox_to_anchor=(0.0,1.17),ncol=4,
        columnspacing=4.3,
        handlelength=5,
        )
# plt.tight_layout()
plt.subplots_adjust(left=0.065, bottom=0.20, right=0.99,top=0.90, hspace=0.15, wspace=0.03)
plt.savefig(graphs_path+"NSE of PCA-based TSDP models.eps",format="EPS",dpi=2000)
plt.savefig(graphs_path+"NSE of PCA-based TSDP models.tif",format="TIFF",dpi=500)
plt.savefig(graphs_path+"NSE of PCA-based TSDP models.pdf",format="PDF",dpi=1200)
plt.show()
