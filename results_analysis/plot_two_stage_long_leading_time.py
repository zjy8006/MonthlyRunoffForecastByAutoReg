import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
import math
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
graphs_path = root_path+'\\graphs\\'

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import sys
sys.path.append(root_path)
from tools.metrics_ import PPTS,mean_absolute_percentage_error
from tools.results_reader import read_long_leading_time
from tools.fit_line import compute_linear_fit,compute_multi_linear_fit



h_v=read_long_leading_time(station="Huaxian",decomposer="vmd")
x_v=read_long_leading_time(station="Xianyang",decomposer="vmd")
z_v=read_long_leading_time(station="Zhangjiashan",decomposer="vmd")
h_e=read_long_leading_time(station="Huaxian",decomposer="eemd")
x_e=read_long_leading_time(station="Xianyang",decomposer="eemd")
z_e=read_long_leading_time(station="Zhangjiashan",decomposer="eemd")
h_s=read_long_leading_time(station="Huaxian",decomposer="ssa")
x_s=read_long_leading_time(station="Xianyang",decomposer="ssa")
z_s=read_long_leading_time(station="Zhangjiashan",decomposer="ssa")
h_w=read_long_leading_time(station="Huaxian",decomposer="dwt")
x_w=read_long_leading_time(station="Xianyang",decomposer="dwt")
z_w=read_long_leading_time(station="Zhangjiashan",decomposer="dwt")
h_m=read_long_leading_time(station="Huaxian",decomposer="modwt",mode='mi')
x_m=read_long_leading_time(station="Xianyang",decomposer="modwt",mode='mi')
z_m=read_long_leading_time(station="Zhangjiashan",decomposer="modwt",mode='mi')
def autolabels(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height,2)
        ax.text(
            x=rect.get_x() + rect.get_width() / 2,
            y=height,
            s='{}'.format(height),
            fontsize=6,
            rotation=90,
            ha='center', va='bottom',
                    )

records_sets=[
    [h_e['records'],h_s['records'],h_v['records'],h_w['records'],h_m['records'],],
    [x_e['records'],x_s['records'],x_v['records'],x_w['records'],x_m['records'],],
    [z_e['records'],z_s['records'],z_v['records'],z_w['records'],z_m['records'],],
]
predictions_sets=[
    [h_e['predictions'],h_s['predictions'],h_v['predictions'],h_w['predictions'],h_m['predictions'],],
    [x_e['predictions'],x_s['predictions'],x_v['predictions'],x_w['predictions'],x_m['predictions'],],
    [z_e['predictions'],z_s['predictions'],z_v['predictions'],z_w['predictions'],z_m['predictions'],],
]

text=[
    'TSDP(EEMD-SVR)','TSDP(SSA-SVR)','TSDP(VMD-SVR)','TSDP(DWT-SVR)','WDDFF(MODWT-SVR)',
]
fig_id=[
    '(a)','(b)','(c)','(d)','(e)',
]
stations=['Huaxian','Xianyang','Zhangjiashan']
for k in range(len(records_sets)):
    records_list = records_sets[k]
    predictions_list = predictions_sets[k]
    plt.figure(figsize=(7.48,6))
    ax1 = plt.subplot2grid((2,6), (0,0), colspan=2,aspect='equal')
    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2,aspect='equal')
    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2,aspect='equal')
    ax4 = plt.subplot2grid((2,6), (1,1), colspan=2,aspect='equal')
    ax5 = plt.subplot2grid((2,6), (1,3), colspan=2,aspect='equal')
    axs = [ax1,ax2,ax3,ax4,ax5]
    for j in range(len(records_list)):
        ax=axs[j]
        ax.set_title(text[j],fontsize=7)
        xx,linear_list,xymin,xymax=compute_multi_linear_fit(
            records=records_list[j],
            predictions=predictions_list[j],
        )
        if j in [4,5,6,7]:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        ax.set_xlabel('Predictions(' + r'$10^8m^3$' +')', )
        if j in [0,3]:
            ax.set_ylabel('Records(' + r'$10^8m^3$' + ')', )
        models=['1-month','3-month','5-month','7-month','9-month']
        markers=['o','v','*','s','+',]
        zorders=[4,3,2,1,0]
        for i in range(len(predictions_list[j])):
            print("length of predictions list:{}".format(len(predictions_list[j])))
            print("length of records list:{}".format(len(records_list[j])))
            # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
            ax.scatter(predictions_list[j][i], records_list[j][i],marker=markers[i],zorder=zorders[i], label=models[i])
            ax.plot(xx, linear_list[i], '--', label=models[i],linewidth=1.0,zorder=zorders[i])
        ax.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
        ax.set_xlim([xymin,xymax])
        ax.set_ylim([xymin,xymax])
        if j==4:
            plt.legend(
                        loc='upper left',
                        # bbox_to_anchor=(0.08,1.01, 1,0.101),
                        bbox_to_anchor=(1.1,1),
                        ncol=1,
                        shadow=False,
                        frameon=True,
                        # fontsize=6,
                        )
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99,top=0.97, hspace=0.25, wspace=0.25)
    plt.savefig(graphs_path+".Long leading time scatters at "+stations[k]+".eps",format="EPS",dpi=2000)
    plt.savefig(graphs_path+".Long leading time scatters at "+stations[k]+".tif",format="TIFF",dpi=1200)
    plt.savefig(graphs_path+".Long leading time scatters at "+stations[k]+".pdf",format="PDF",dpi=1200)


nse_eemd_data1=[h_e['nse'][0],x_e['nse'][0],z_e['nse'][0]]
nse_eemd_data3=[h_e['nse'][1],x_e['nse'][1],z_e['nse'][1]]
nse_eemd_data5=[h_e['nse'][2],x_e['nse'][2],z_e['nse'][2]]
nse_eemd_data7=[h_e['nse'][3],x_e['nse'][3],z_e['nse'][3]]
nse_eemd_data9=[h_e['nse'][4],x_e['nse'][4],z_e['nse'][4]]
nse_ssa_data1=[h_s['nse'][0],x_s['nse'][0],z_s['nse'][0]]
nse_ssa_data3=[h_s['nse'][1],x_s['nse'][1],z_s['nse'][1]]
nse_ssa_data5=[h_s['nse'][2],x_s['nse'][2],z_s['nse'][2]]
nse_ssa_data7=[h_s['nse'][3],x_s['nse'][3],z_s['nse'][3]]
nse_ssa_data9=[h_s['nse'][4],x_s['nse'][4],z_s['nse'][4]]
nse_vmd_data1=[h_v['nse'][0],x_v['nse'][0],z_v['nse'][0]]
nse_vmd_data3=[h_v['nse'][1],x_v['nse'][1],z_v['nse'][1]]
nse_vmd_data5=[h_v['nse'][2],x_v['nse'][2],z_v['nse'][2]]
nse_vmd_data7=[h_v['nse'][3],x_v['nse'][3],z_v['nse'][3]]
nse_vmd_data9=[h_v['nse'][4],x_v['nse'][4],z_v['nse'][4]]
nse_dwt_data1=[h_w['nse'][0],x_w['nse'][0],z_w['nse'][0]]
nse_dwt_data3=[h_w['nse'][1],x_w['nse'][1],z_w['nse'][1]]
nse_dwt_data5=[h_w['nse'][2],x_w['nse'][2],z_w['nse'][2]]
nse_dwt_data7=[h_w['nse'][3],x_w['nse'][3],z_w['nse'][3]]
nse_dwt_data9=[h_w['nse'][4],x_w['nse'][4],z_w['nse'][4]]
nse_modwt_data1=[h_m['nse'][0],x_m['nse'][0],z_m['nse'][0]]
nse_modwt_data3=[h_m['nse'][1],x_m['nse'][1],z_m['nse'][1]]
nse_modwt_data5=[h_m['nse'][2],x_m['nse'][2],z_m['nse'][2]]
nse_modwt_data7=[h_m['nse'][3],x_m['nse'][3],z_m['nse'][3]]
nse_modwt_data9=[h_m['nse'][4],x_m['nse'][4],z_m['nse'][4]]
nse_data=[
    nse_eemd_data1,
    nse_eemd_data3,
    nse_eemd_data5,
    nse_eemd_data7,
    nse_eemd_data9,
    nse_ssa_data1,
    nse_ssa_data3,
    nse_ssa_data5,
    nse_ssa_data7,
    nse_ssa_data9,
    nse_vmd_data1,
    nse_vmd_data3,
    nse_vmd_data5,
    nse_vmd_data7,
    nse_vmd_data9,
    nse_dwt_data1,
    nse_dwt_data3,
    nse_dwt_data5,
    nse_dwt_data7,
    nse_dwt_data9,
    nse_modwt_data1,
    nse_modwt_data3,
    nse_modwt_data5,
    nse_modwt_data7,
    nse_modwt_data9,
]
eemd_mean_nse=[
    sum(nse_eemd_data1)/len(nse_eemd_data1),
    sum(nse_eemd_data3)/len(nse_eemd_data3),
    sum(nse_eemd_data5)/len(nse_eemd_data5),
    sum(nse_eemd_data7)/len(nse_eemd_data7),
    sum(nse_eemd_data9)/len(nse_eemd_data9),
]
ssa_mean_nse=[
    sum(nse_ssa_data1)/len(nse_ssa_data1),
    sum(nse_ssa_data3)/len(nse_ssa_data3),
    sum(nse_ssa_data5)/len(nse_ssa_data5),
    sum(nse_ssa_data7)/len(nse_ssa_data7),
    sum(nse_ssa_data9)/len(nse_ssa_data9),
]
vmd_mean_nse=[
    sum(nse_vmd_data1)/len(nse_vmd_data1),
    sum(nse_vmd_data3)/len(nse_vmd_data3),
    sum(nse_vmd_data5)/len(nse_vmd_data5),
    sum(nse_vmd_data7)/len(nse_vmd_data7),
    sum(nse_vmd_data9)/len(nse_vmd_data9),
]
dwt_mean_nse=[
    sum(nse_dwt_data1)/len(nse_dwt_data1),
    sum(nse_dwt_data3)/len(nse_dwt_data3),
    sum(nse_dwt_data5)/len(nse_dwt_data5),
    sum(nse_dwt_data7)/len(nse_dwt_data7),
    sum(nse_dwt_data9)/len(nse_dwt_data9),
]

modwt_mean_nse=[
    sum(nse_modwt_data1)/len(nse_modwt_data1),
    sum(nse_modwt_data3)/len(nse_modwt_data3),
    sum(nse_modwt_data5)/len(nse_modwt_data5),
    sum(nse_modwt_data7)/len(nse_modwt_data7),
    sum(nse_modwt_data9)/len(nse_modwt_data9),
]

nrmse_eemd_data1=[h_e['nrmse'][0],x_e['nrmse'][0],z_e['nrmse'][0]]
nrmse_eemd_data3=[h_e['nrmse'][1],x_e['nrmse'][1],z_e['nrmse'][1]]
nrmse_eemd_data5=[h_e['nrmse'][2],x_e['nrmse'][2],z_e['nrmse'][2]]
nrmse_eemd_data7=[h_e['nrmse'][3],x_e['nrmse'][3],z_e['nrmse'][3]]
nrmse_eemd_data9=[h_e['nrmse'][4],x_e['nrmse'][4],z_e['nrmse'][4]]
nrmse_ssa_data1=[h_s['nrmse'][0],x_s['nrmse'][0],z_s['nrmse'][0]]
nrmse_ssa_data3=[h_s['nrmse'][1],x_s['nrmse'][1],z_s['nrmse'][1]]
nrmse_ssa_data5=[h_s['nrmse'][2],x_s['nrmse'][2],z_s['nrmse'][2]]
nrmse_ssa_data7=[h_s['nrmse'][3],x_s['nrmse'][3],z_s['nrmse'][3]]
nrmse_ssa_data9=[h_s['nrmse'][4],x_s['nrmse'][4],z_s['nrmse'][4]]
nrmse_vmd_data1=[h_v['nrmse'][0],x_v['nrmse'][0],z_v['nrmse'][0]]
nrmse_vmd_data3=[h_v['nrmse'][1],x_v['nrmse'][1],z_v['nrmse'][1]]
nrmse_vmd_data5=[h_v['nrmse'][2],x_v['nrmse'][2],z_v['nrmse'][2]]
nrmse_vmd_data7=[h_v['nrmse'][3],x_v['nrmse'][3],z_v['nrmse'][3]]
nrmse_vmd_data9=[h_v['nrmse'][4],x_v['nrmse'][4],z_v['nrmse'][4]]
nrmse_dwt_data1=[h_w['nrmse'][0],x_w['nrmse'][0],z_w['nrmse'][0]]
nrmse_dwt_data3=[h_w['nrmse'][1],x_w['nrmse'][1],z_w['nrmse'][1]]
nrmse_dwt_data5=[h_w['nrmse'][2],x_w['nrmse'][2],z_w['nrmse'][2]]
nrmse_dwt_data7=[h_w['nrmse'][3],x_w['nrmse'][3],z_w['nrmse'][3]]
nrmse_dwt_data9=[h_w['nrmse'][4],x_w['nrmse'][4],z_w['nrmse'][4]]
nrmse_modwt_data1=[h_m['nrmse'][0],x_m['nrmse'][0],z_m['nrmse'][0]]
nrmse_modwt_data3=[h_m['nrmse'][1],x_m['nrmse'][1],z_m['nrmse'][1]]
nrmse_modwt_data5=[h_m['nrmse'][2],x_m['nrmse'][2],z_m['nrmse'][2]]
nrmse_modwt_data7=[h_m['nrmse'][3],x_m['nrmse'][3],z_m['nrmse'][3]]
nrmse_modwt_data9=[h_m['nrmse'][4],x_m['nrmse'][4],z_m['nrmse'][4]]
nrmse_data=[
    nrmse_eemd_data1,
    nrmse_eemd_data3,
    nrmse_eemd_data5,
    nrmse_eemd_data7,
    nrmse_eemd_data9,
    nrmse_ssa_data1,
    nrmse_ssa_data3,
    nrmse_ssa_data5,
    nrmse_ssa_data7,
    nrmse_ssa_data9,
    nrmse_vmd_data1,
    nrmse_vmd_data3,
    nrmse_vmd_data5,
    nrmse_vmd_data7,
    nrmse_vmd_data9,
    nrmse_dwt_data1,
    nrmse_dwt_data3,
    nrmse_dwt_data5,
    nrmse_dwt_data7,
    nrmse_dwt_data9,
    nrmse_modwt_data1,
    nrmse_modwt_data3,
    nrmse_modwt_data5,
    nrmse_modwt_data7,
    nrmse_modwt_data9,
]
eemd_mean_nrmse=[
    sum(nrmse_eemd_data1)/len(nrmse_eemd_data1),
    sum(nrmse_eemd_data3)/len(nrmse_eemd_data3),
    sum(nrmse_eemd_data5)/len(nrmse_eemd_data5),
    sum(nrmse_eemd_data7)/len(nrmse_eemd_data7),
    sum(nrmse_eemd_data9)/len(nrmse_eemd_data9),
]
ssa_mean_nrmse=[
    sum(nrmse_ssa_data1)/len(nrmse_ssa_data1),
    sum(nrmse_ssa_data3)/len(nrmse_ssa_data3),
    sum(nrmse_ssa_data5)/len(nrmse_ssa_data5),
    sum(nrmse_ssa_data7)/len(nrmse_ssa_data7),
    sum(nrmse_ssa_data9)/len(nrmse_ssa_data9),
]
vmd_mean_nrmse=[
    sum(nrmse_vmd_data1)/len(nrmse_vmd_data1),
    sum(nrmse_vmd_data3)/len(nrmse_vmd_data3),
    sum(nrmse_vmd_data5)/len(nrmse_vmd_data5),
    sum(nrmse_vmd_data7)/len(nrmse_vmd_data7),
    sum(nrmse_vmd_data9)/len(nrmse_vmd_data9),
]
dwt_mean_nrmse=[
    sum(nrmse_dwt_data1)/len(nrmse_dwt_data1),
    sum(nrmse_dwt_data3)/len(nrmse_dwt_data3),
    sum(nrmse_dwt_data5)/len(nrmse_dwt_data5),
    sum(nrmse_dwt_data7)/len(nrmse_dwt_data7),
    sum(nrmse_dwt_data9)/len(nrmse_dwt_data9),
]
modwt_mean_nrmse=[
    sum(nrmse_modwt_data1)/len(nrmse_modwt_data1),
    sum(nrmse_modwt_data3)/len(nrmse_modwt_data3),
    sum(nrmse_modwt_data5)/len(nrmse_modwt_data5),
    sum(nrmse_modwt_data7)/len(nrmse_modwt_data7),
    sum(nrmse_modwt_data9)/len(nrmse_modwt_data9),
]
ppts_eemd_data1=[h_e['ppts'][0],x_e['ppts'][0],z_e['ppts'][0]]
ppts_eemd_data3=[h_e['ppts'][1],x_e['ppts'][1],z_e['ppts'][1]]
ppts_eemd_data5=[h_e['ppts'][2],x_e['ppts'][2],z_e['ppts'][2]]
ppts_eemd_data7=[h_e['ppts'][3],x_e['ppts'][3],z_e['ppts'][3]]
ppts_eemd_data9=[h_e['ppts'][4],x_e['ppts'][4],z_e['ppts'][4]]
ppts_ssa_data1=[h_s['ppts'][0],x_s['ppts'][0],z_s['ppts'][0]]
ppts_ssa_data3=[h_s['ppts'][1],x_s['ppts'][1],z_s['ppts'][1]]
ppts_ssa_data5=[h_s['ppts'][2],x_s['ppts'][2],z_s['ppts'][2]]
ppts_ssa_data7=[h_s['ppts'][3],x_s['ppts'][3],z_s['ppts'][3]]
ppts_ssa_data9=[h_s['ppts'][4],x_s['ppts'][4],z_s['ppts'][4]]
ppts_vmd_data1=[h_v['ppts'][0],x_v['ppts'][0],z_v['ppts'][0]]
ppts_vmd_data3=[h_v['ppts'][1],x_v['ppts'][1],z_v['ppts'][1]]
ppts_vmd_data5=[h_v['ppts'][2],x_v['ppts'][2],z_v['ppts'][2]]
ppts_vmd_data7=[h_v['ppts'][3],x_v['ppts'][3],z_v['ppts'][3]]
ppts_vmd_data9=[h_v['ppts'][4],x_v['ppts'][4],z_v['ppts'][4]]
ppts_dwt_data1=[h_w['ppts'][0],x_w['ppts'][0],z_w['ppts'][0]]
ppts_dwt_data3=[h_w['ppts'][1],x_w['ppts'][1],z_w['ppts'][1]]
ppts_dwt_data5=[h_w['ppts'][2],x_w['ppts'][2],z_w['ppts'][2]]
ppts_dwt_data7=[h_w['ppts'][3],x_w['ppts'][3],z_w['ppts'][3]]
ppts_dwt_data9=[h_w['ppts'][4],x_w['ppts'][4],z_w['ppts'][4]]
ppts_modwt_data1=[h_m['ppts'][0],x_m['ppts'][0],z_m['ppts'][0]]
ppts_modwt_data3=[h_m['ppts'][1],x_m['ppts'][1],z_m['ppts'][1]]
ppts_modwt_data5=[h_m['ppts'][2],x_m['ppts'][2],z_m['ppts'][2]]
ppts_modwt_data7=[h_m['ppts'][3],x_m['ppts'][3],z_m['ppts'][3]]
ppts_modwt_data9=[h_m['ppts'][4],x_m['ppts'][4],z_m['ppts'][4]]
ppts_data=[
    ppts_eemd_data1,
    ppts_eemd_data3,
    ppts_eemd_data5,
    ppts_eemd_data7,
    ppts_eemd_data9,
    ppts_ssa_data1,
    ppts_ssa_data3,
    ppts_ssa_data5,
    ppts_ssa_data7,
    ppts_ssa_data9,
    ppts_vmd_data1,
    ppts_vmd_data3,
    ppts_vmd_data5,
    ppts_vmd_data7,
    ppts_vmd_data9,
    ppts_dwt_data1,
    ppts_dwt_data3,
    ppts_dwt_data5,
    ppts_dwt_data7,
    ppts_dwt_data9,
    ppts_modwt_data1,
    ppts_modwt_data3,
    ppts_modwt_data5,
    ppts_modwt_data7,
    ppts_modwt_data9,
]
eemd_mean_ppts=[
    sum(ppts_eemd_data1)/len(ppts_eemd_data1),
    sum(ppts_eemd_data3)/len(ppts_eemd_data3),
    sum(ppts_eemd_data5)/len(ppts_eemd_data5),
    sum(ppts_eemd_data7)/len(ppts_eemd_data7),
    sum(ppts_eemd_data9)/len(ppts_eemd_data9),
]
ssa_mean_ppts=[
    sum(ppts_ssa_data1)/len(ppts_ssa_data1),
    sum(ppts_ssa_data3)/len(ppts_ssa_data3),
    sum(ppts_ssa_data5)/len(ppts_ssa_data5),
    sum(ppts_ssa_data7)/len(ppts_ssa_data7),
    sum(ppts_ssa_data9)/len(ppts_ssa_data9),
]
vmd_mean_ppts=[
    sum(ppts_vmd_data1)/len(ppts_vmd_data1),
    sum(ppts_vmd_data3)/len(ppts_vmd_data3),
    sum(ppts_vmd_data5)/len(ppts_vmd_data5),
    sum(ppts_vmd_data7)/len(ppts_vmd_data7),
    sum(ppts_vmd_data9)/len(ppts_vmd_data9),
]
dwt_mean_ppts=[
    sum(ppts_dwt_data1)/len(ppts_dwt_data1),
    sum(ppts_dwt_data3)/len(ppts_dwt_data3),
    sum(ppts_dwt_data5)/len(ppts_dwt_data5),
    sum(ppts_dwt_data7)/len(ppts_dwt_data7),
    sum(ppts_dwt_data9)/len(ppts_dwt_data9),
]
modwt_mean_ppts=[
    sum(ppts_modwt_data1)/len(ppts_modwt_data1),
    sum(ppts_modwt_data3)/len(ppts_modwt_data3),
    sum(ppts_modwt_data5)/len(ppts_modwt_data5),
    sum(ppts_modwt_data7)/len(ppts_modwt_data7),
    sum(ppts_modwt_data9)/len(ppts_modwt_data9),
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


all_datas = [nse_data,nrmse_data,ppts_data]
fig_index=["(a)","(b)","(c)"]
labels=[
    "TSDP(EEMD-SVR)\n(1-month ahead)",
    "TSDP(EEMD-SVR)\n(3-month ahead)",
    "TSDP(EEMD-SVR)\n(5-month ahead)",
    "TSDP(EEMD-SVR)\n(7-month ahead)",
    "TSDP(EEMD-SVR)\n(9-month ahead)",
    "TSDP(SSA-SVR)\n(1-month ahead)",
    "TSDP(SSA-SVR)\n(3-month ahead)",
    "TSDP(SSA-SVR)\n(5-month ahead)",
    "TSDP(SSA-SVR)\n(7-month ahead)",
    "TSDP(SSA-SVR)\n(9-month ahead)",
    "TSDP(VMD-SVR)\n(1-month ahead)",
    "TSDP(VMD-SVR)\n(3-month ahead)",
    "TSDP(VMD-SVR)\n(5-month ahead)",
    "TSDP(VMD-SVR)\n(7-month ahead)",
    "TSDP(VMD-SVR)\n(9-month ahead)",
    "TSDP(DWT-SVR)\n(1-month ahead)",
    "TSDP(DWT-SVR)\n(3-month ahead)",
    "TSDP(DWT-SVR)\n(5-month ahead)",
    "TSDP(DWT-SVR)\n(7-month ahead)",
    "TSDP(DWT-SVR)\n(9-month ahead)",
    "WDDFF(MODWT-SVR)\n(1-month ahead)",
    "WDDFF(MODWT-SVR)\n(3-month ahead)",
    "WDDFF(MODWT-SVR)\n(5-month ahead)",
    "WDDFF(MODWT-SVR)\n(7-month ahead)",
    "WDDFF(MODWT-SVR)\n(9-month ahead)",
    ]
x = list(range(25))
ylabels=[
    r"$NSE$",r"$NRMSE$",r"$PPTS(5)(\%)$",
]
x_s=[-1.1,-1.1,-1.1]
y_s=[0.93,1.8,78]
plt.figure(figsize=(7.48, 5.54))
for i in range(len(all_datas)):
    ax1 = plt.subplot(3, 1, i+1)
    ax1.yaxis.grid(True)
    ax1.text(x_s[i],y_s[i],fig_index[i],fontsize=7)
    vplot1 = plt.violinplot(
        dataset=all_datas[i],
        positions=x,
        showmeans=True,
    )
    ax1.plot(list(range(0,5)),lines[i][0],'--',lw=0.5,color='blue')
    ax1.plot(list(range(5,10)),lines[i][1],'--',lw=0.5,color='blue')
    ax1.plot(list(range(10,15)),lines[i][2],'--',lw=0.5,color='blue')
    ax1.plot(list(range(15,20)),lines[i][3],'--',lw=0.5,color='blue')
    ax1.plot(list(range(20,25)),lines[i][4],'--',lw=0.5,color='blue')
    print(type(vplot1["cmeans"]))
    plt.ylabel(ylabels[i])
    if i==len(all_datas)-1:
        plt.xticks(x, labels, rotation=45)
    else:
        plt.xticks([])
    for pc in vplot1['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

plt.tight_layout()
plt.savefig(graphs_path+'/Long leading time metrics.eps', format='EPS', dpi=2000)
plt.savefig(graphs_path+'/Long leading time metrics.tif',format='TIFF', dpi=1200)
plt.savefig(graphs_path+'/Long leading time metrics.pdf',format='PDF', dpi=1200)
plt.show()
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