import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 6
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/graphs/'
print("root path:{}".format(root_path))
sys.path.append(root_path)
from tools.results_reader import read_two_stage_train_devtest,read_pure_esvr

h_eemd_td_t = pd.read_csv(root_path+"/Huaxian_eemd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")
h_ssa_td_t = pd.read_csv(root_path+"/Huaxian_ssa/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")
h_vmd_td_t = pd.read_csv(root_path+"/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")
h_dwt_td_t = pd.read_csv(root_path+"/Huaxian_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")

h_eemd_td_ap = pd.read_csv(root_path+"/Huaxian_eemd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")
h_ssa_td_ap = pd.read_csv(root_path+"/Huaxian_ssa/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")
h_vmd_td_ap = pd.read_csv(root_path+"/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")
h_dwt_td_ap = pd.read_csv(root_path+"/Huaxian_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")

h_eemd_t_v = pd.read_csv(root_path+"/Huaxian_eemd/projects/esvr/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")
h_ssa_t_v = pd.read_csv(root_path+"/Huaxian_ssa/projects/esvr/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")
h_vmd_t_v = pd.read_csv(root_path+"/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")
h_dwt_t_v = pd.read_csv(root_path+"/Huaxian_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")

h_eemd_t_ap = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_ssa_t_ap = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_vmd_t_ap = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
h_dwt_t_ap = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')

x_eemd_td_t = pd.read_csv(root_path+"/Xianyang_eemd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")
x_ssa_td_t = pd.read_csv(root_path+"/Xianyang_ssa/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")
x_vmd_td_t = pd.read_csv(root_path+"/Xianyang_vmd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")
x_dwt_td_t = pd.read_csv(root_path+"/Xianyang_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")

x_eemd_td_ap = pd.read_csv(root_path+"/Xianyang_eemd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")
x_ssa_td_ap = pd.read_csv(root_path+"/Xianyang_ssa/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")
x_vmd_td_ap = pd.read_csv(root_path+"/Xianyang_vmd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")
x_dwt_td_ap = pd.read_csv(root_path+"/Xianyang_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")

x_eemd_t_v = pd.read_csv(root_path+"/Xianyang_eemd/projects/esvr/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")
x_ssa_t_v = pd.read_csv(root_path+"/Xianyang_ssa/projects/esvr/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")
x_vmd_t_v = pd.read_csv(root_path+"/Xianyang_vmd/projects/esvr/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")
x_dwt_t_v = pd.read_csv(root_path+"/Xianyang_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")

x_eemd_t_ap = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_ssa_t_ap = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_vmd_t_ap = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
x_dwt_t_ap = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')

z_eemd_td_t = pd.read_csv(root_path+"/Zhangjiashan_eemd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")
z_ssa_td_t = pd.read_csv(root_path+"/Zhangjiashan_ssa/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")
z_vmd_td_t = pd.read_csv(root_path+"/Zhangjiashan_vmd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")
z_dwt_td_t = pd.read_csv(root_path+"/Zhangjiashan_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf_traindev_test/optimal_model_results.csv")

z_eemd_td_ap = pd.read_csv(root_path+"/Zhangjiashan_eemd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")
z_ssa_td_ap = pd.read_csv(root_path+"/Zhangjiashan_ssa/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")
z_vmd_td_ap = pd.read_csv(root_path+"/Zhangjiashan_vmd/projects/esvr/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")
z_dwt_td_ap = pd.read_csv(root_path+"/Zhangjiashan_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf_traindev_append/optimal_model_results.csv")

z_eemd_t_v = pd.read_csv(root_path+"/Zhangjiashan_eemd/projects/esvr/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")
z_ssa_t_v = pd.read_csv(root_path+"/Zhangjiashan_ssa/projects/esvr/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")
z_vmd_t_v = pd.read_csv(root_path+"/Zhangjiashan_vmd/projects/esvr/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")
z_dwt_t_v = pd.read_csv(root_path+"/Zhangjiashan_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf_train_val/optimal_model_results.csv")

z_eemd_t_ap = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_ssa_t_ap = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_vmd_t_ap = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')
z_dwt_t_ap = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/optimal_model_results.csv')

print(h_eemd_t_v['train_nse'])
print(h_eemd_t_v['train_nse'])
print(h_eemd_t_v['train_nse'])

td_t = [#without mix and shuffle and generate testing samples from testing decompositions
    [h_eemd_td_t["train_nse"][0], x_eemd_td_t["train_nse"][0], z_eemd_td_t["train_nse"][0]],
    [h_eemd_td_t["dev_nse"][0], x_eemd_td_t["dev_nse"][0], z_eemd_td_t["dev_nse"][0]],
    [h_eemd_td_t["test_nse"][0], x_eemd_td_t["test_nse"][0], z_eemd_td_t["test_nse"][0]],
    [h_ssa_td_t["train_nse"][0], x_ssa_td_t["train_nse"][0], z_ssa_td_t["train_nse"][0]],
    [h_ssa_td_t["dev_nse"][0], x_ssa_td_t["dev_nse"][0], z_ssa_td_t["dev_nse"][0]],
    [h_ssa_td_t["test_nse"][0], x_ssa_td_t["test_nse"][0], z_ssa_td_t["test_nse"][0]],
    [h_vmd_td_t["train_nse"][0], x_vmd_td_t["train_nse"][0], z_vmd_td_t["train_nse"][0]],
    [h_vmd_td_t["dev_nse"][0], x_vmd_td_t["dev_nse"][0], z_vmd_td_t["dev_nse"][0]],
    [h_vmd_td_t["test_nse"][0], x_vmd_td_t["test_nse"][0], z_vmd_td_t["test_nse"][0]],
    [h_dwt_td_t["train_nse"][0], x_dwt_td_t["train_nse"][0], z_dwt_td_t["train_nse"][0]],
    [h_dwt_td_t["dev_nse"][0], x_dwt_td_t["dev_nse"][0], z_dwt_td_t["dev_nse"][0]],
    [h_dwt_td_t["test_nse"][0], x_dwt_td_t["test_nse"][0], z_dwt_td_t["test_nse"][0]],
]

td_ap = [#without mix and shuffle and generate testing samples from appended decompositions
    [h_eemd_td_ap["train_nse"][0], x_eemd_td_ap["train_nse"][0], z_eemd_td_ap["train_nse"][0]],
    [h_eemd_td_ap["dev_nse"][0], x_eemd_td_ap["dev_nse"][0], z_eemd_td_ap["dev_nse"][0]],
    [h_eemd_td_ap["test_nse"][0], x_eemd_td_ap["test_nse"][0], z_eemd_td_ap["test_nse"][0]],
    [h_ssa_td_ap["train_nse"][0], x_ssa_td_ap["train_nse"][0], z_ssa_td_ap["train_nse"][0]],
    [h_ssa_td_ap["dev_nse"][0], x_ssa_td_ap["dev_nse"][0], z_ssa_td_ap["dev_nse"][0]],
    [h_ssa_td_ap["test_nse"][0], x_ssa_td_ap["test_nse"][0], z_ssa_td_ap["test_nse"][0]],
    [h_vmd_td_ap["train_nse"][0], x_vmd_td_ap["train_nse"][0], z_vmd_td_ap["train_nse"][0]],
    [h_vmd_td_ap["dev_nse"][0], x_vmd_td_ap["dev_nse"][0], z_vmd_td_ap["dev_nse"][0]],
    [h_vmd_td_ap["test_nse"][0], x_vmd_td_ap["test_nse"][0], z_vmd_td_ap["test_nse"][0]],
    [h_dwt_td_ap["train_nse"][0], x_dwt_td_ap["train_nse"][0], z_dwt_td_ap["train_nse"][0]],
    [h_dwt_td_ap["dev_nse"][0], x_dwt_td_ap["dev_nse"][0], z_dwt_td_ap["dev_nse"][0]],
    [h_dwt_td_ap["test_nse"][0], x_dwt_td_ap["test_nse"][0], z_dwt_td_ap["test_nse"][0]],
]
t_v = [#with mix and shuffle and generate validation samples from validation decompositions
    [h_eemd_t_v["train_nse"][0], x_eemd_t_v["train_nse"][0], z_eemd_t_v["train_nse"][0]],
    [h_eemd_t_v["dev_nse"][0], x_eemd_t_v["dev_nse"][0], z_eemd_t_v["dev_nse"][0]],
    [h_eemd_t_v["test_nse"][0], x_eemd_t_v["test_nse"][0], z_eemd_t_v["test_nse"][0]],
    [h_ssa_t_v["train_nse"][0], x_ssa_t_v["train_nse"][0], z_ssa_t_v["train_nse"][0]],
    [h_ssa_t_v["dev_nse"][0], x_ssa_t_v["dev_nse"][0], z_ssa_t_v["dev_nse"][0]],
    [h_ssa_t_v["test_nse"][0], x_ssa_t_v["test_nse"][0], z_ssa_t_v["test_nse"][0]],
    [h_vmd_t_v["train_nse"][0], x_vmd_t_v["train_nse"][0], z_vmd_t_v["train_nse"][0]],
    [h_vmd_t_v["dev_nse"][0], x_vmd_t_v["dev_nse"][0], z_vmd_t_v["dev_nse"][0]],
    [h_vmd_t_v["test_nse"][0], x_vmd_t_v["test_nse"][0], z_vmd_t_v["test_nse"][0]],
    [h_dwt_t_v["train_nse"][0], x_dwt_t_v["train_nse"][0], z_dwt_t_v["train_nse"][0]],
    [h_dwt_t_v["dev_nse"][0], x_dwt_t_v["dev_nse"][0], z_dwt_t_v["dev_nse"][0]],
    [h_dwt_t_v["test_nse"][0], x_dwt_t_v["test_nse"][0], z_dwt_t_v["test_nse"][0]],
]

t_ap = [#with mix and shuffle and generate validation samples from appended decompositions
    [h_eemd_t_ap["train_nse"][0], x_eemd_t_ap["train_nse"][0], z_eemd_t_ap["train_nse"][0]],
    [h_eemd_t_ap["dev_nse"][0], x_eemd_t_ap["dev_nse"][0], z_eemd_t_ap["dev_nse"][0]],
    [h_eemd_t_ap["test_nse"][0], x_eemd_t_ap["test_nse"][0], z_eemd_t_ap["test_nse"][0]],
    [h_ssa_t_ap["train_nse"][0], x_ssa_t_ap["train_nse"][0], z_ssa_t_ap["train_nse"][0]],
    [h_ssa_t_ap["dev_nse"][0], x_ssa_t_ap["dev_nse"][0], z_ssa_t_ap["dev_nse"][0]],
    [h_ssa_t_ap["test_nse"][0], x_ssa_t_ap["test_nse"][0], z_ssa_t_ap["test_nse"][0]],
    [h_vmd_t_ap["train_nse"][0], x_vmd_t_ap["train_nse"][0], z_vmd_t_ap["train_nse"][0]],
    [h_vmd_t_ap["dev_nse"][0], x_vmd_t_ap["dev_nse"][0], z_vmd_t_ap["dev_nse"][0]],
    [h_vmd_t_ap["test_nse"][0], x_vmd_t_ap["test_nse"][0], z_vmd_t_ap["test_nse"][0]],
    [h_dwt_t_ap["train_nse"][0], x_dwt_t_ap["train_nse"][0], z_dwt_t_ap["train_nse"][0]],
    [h_dwt_t_ap["dev_nse"][0], x_dwt_t_ap["dev_nse"][0], z_dwt_t_ap["dev_nse"][0]],
    [h_dwt_t_ap["test_nse"][0], x_dwt_t_ap["test_nse"][0], z_dwt_t_ap["test_nse"][0]],
]

labels = [
    "EEMD-SVR (cal)",
    "EEMD-SVR (dev)",
    "EEMD-SVR (test)",
    "SSA-SVR (cal)",
    "SSA-SVR (dev)",
    "SSA-SVR (test)",
    "VMD-SVR (cal)",
    "VMD-SVR (dev)",
    "VMD-SVR (test)",
    "DWT-SVR (cal)",
    "DWT-SVR (dev)",
    "DWT-SVR (test)",
]

all_datas = [
    td_t,
    td_ap,
    t_v,
    t_ap,
]

fig_index=[
    "(a) Scheme 1",
    "(b) Scheme 2",
    "(c) Scheme 3",
    "(d) Scheme 4",
    ]

face_colors={
    0:'#00FF9F',#eemd-svr
    1:'#00FF9F',#eemd-svr
    2:'#00FF9F',#eemd-svr
    3:'#FFE4E1',#ssa-svr
    4:'#FFE4E1',#ssa-svr
    5:'#FFE4E1',#ssa-svr
    6:'#FF4500',#vmd-svr
    7:'#FF4500',#vmd-svr
    8:'#FF4500',#vmd-svr
    9:'#FAA460',#dwt-svr
    10:'#FAA460',#dwt-svr
    11:'#FAA460',#dwt-svr
}
x = list(range(len(all_datas[0])))
# print('x={}'.format(x))
plt.figure(figsize=(7.48, 3.48))
x_s = [11,11,11,11]
y_s = [-0.35,-1.65,-0.95,-0.3]

for i in range(len(all_datas)):
    # print(all_datas[i])
    ax=plt.subplot(2, 2, i+1)
    plt.ylim(-1.8,1.1)
    vplot1 = plt.violinplot(
        dataset=all_datas[i],
        positions=x,
        showmeans=True,
    )
    plt.text(9,-1.7,fig_index[i],fontweight='normal',fontsize=7)
    if i==len(all_datas)-1 or i==len(all_datas)-2:
        plt.xticks(x, labels, rotation=45)
    else:
        plt.xticks([])
    if i in [0,2]:
        plt.ylabel(r"$NSE$")
    # else:
    #     plt.yticks([])
    ax.yaxis.grid(True)
    for pc,k in zip(vplot1['bodies'],range(len(face_colors))):
        print(pc)
        # pc.set_facecolor('#D43F3A')
        pc.set_facecolor(face_colors[k])
        pc.set_edgecolor('black')
        pc.set_alpha(1)

plt.subplots_adjust(left=0.066, bottom=0.2, right=0.99,top=0.99, hspace=0.05, wspace=0.12)
plt.savefig(graphs_path+'/Nse violin of evaluations of boundary reduction.eps',format='EPS', dpi=2000)
plt.savefig(graphs_path+'/Nse violin of evaluations of boundary reduction.tif',format='TIFF', dpi=1200)
plt.savefig(graphs_path+'/Nse violin of evaluations of boundary reduction.pdf',format='PDF', dpi=1200)
plt.show()


