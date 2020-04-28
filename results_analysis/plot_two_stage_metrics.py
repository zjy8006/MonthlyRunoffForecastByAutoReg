import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 6
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/graphs/'
results_path = root_path+'/results_analysis/results/'
print("root path:{}".format(root_path))
sys.path.append(root_path)
from tools.results_reader import read_two_stage, read_pure_esvr,read_pure_arma
h_arma = read_pure_arma("Huaxian")
x_arma = read_pure_arma("Xianyang")
z_arma = read_pure_arma("Zhangjiashan")

h_esvr = read_pure_esvr("Huaxian")
x_esvr = read_pure_esvr("Xianyang")
z_esvr = read_pure_esvr("Zhangjiashan")

h_vmd_esvr = read_two_stage(station="Huaxian", decomposer="vmd", predict_pattern="one_step_1_ahead_forecast_pacf")
x_vmd_esvr = read_two_stage(station="Xianyang", decomposer="vmd", predict_pattern="one_step_1_ahead_forecast_pacf")
z_vmd_esvr = read_two_stage(station="Zhangjiashan", decomposer="vmd", predict_pattern="one_step_1_ahead_forecast_pacf")

h_eemd_esvr = read_two_stage(station="Huaxian", decomposer="eemd", predict_pattern="one_step_1_ahead_forecast_pacf")
x_eemd_esvr = read_two_stage(station="Xianyang", decomposer="eemd", predict_pattern="one_step_1_ahead_forecast_pacf")
z_eemd_esvr = read_two_stage(station="Zhangjiashan", decomposer="eemd", predict_pattern="one_step_1_ahead_forecast_pacf")

h_ssa_esvr = read_two_stage(station="Huaxian", decomposer="ssa", predict_pattern="one_step_1_ahead_forecast_pacf")
x_ssa_esvr = read_two_stage(station="Xianyang", decomposer="ssa", predict_pattern="one_step_1_ahead_forecast_pacf")
z_ssa_esvr = read_two_stage(station="Zhangjiashan", decomposer="ssa", predict_pattern="one_step_1_ahead_forecast_pacf")

h_dwt_esvr = read_two_stage(station="Huaxian", decomposer="dwt", predict_pattern="one_step_1_ahead_forecast_pacf")
x_dwt_esvr = read_two_stage(station="Xianyang", decomposer="dwt", predict_pattern="one_step_1_ahead_forecast_pacf")
z_dwt_esvr = read_two_stage(station="Zhangjiashan", decomposer="dwt", predict_pattern="one_step_1_ahead_forecast_pacf")

h_modwt_esvr = read_two_stage(station="Huaxian", decomposer="modwt", predict_pattern="single_hybrid_1_ahead")
x_modwt_esvr = read_two_stage(station="Xianyang", decomposer="modwt", predict_pattern="single_hybrid_1_ahead")
z_modwt_esvr = read_two_stage(station="Zhangjiashan", decomposer="modwt", predict_pattern="single_hybrid_1_ahead")

index = [
    "Huaxian", "Huaxian-vmd", "Huaxian-eemd", "Huaxian-ssa", "Huaxian-dwt",
    "Xianyang", "Xianyang-vmd", "Xianyang-eemd", "Xianyang-ssa", "Xianyang-dwt",
    "Zhangjiashan", "Zhangjiashan-vmd", "Zhangjiashan-eemd", "Zhangjiashan-ssa", "Zhangjiashan-dwt"]

metrics_dict = {
    "nse": [h_esvr['test_nse'], h_vmd_esvr['test_nse'], h_eemd_esvr['test_nse'], h_ssa_esvr['test_nse'], h_dwt_esvr['test_nse'],
           x_esvr['test_nse'], x_vmd_esvr['test_nse'], x_eemd_esvr['test_nse'], x_ssa_esvr['test_nse'], x_dwt_esvr['test_nse'],
           z_esvr['test_nse'], z_vmd_esvr['test_nse'], z_eemd_esvr['test_nse'], z_ssa_esvr['test_nse'], z_dwt_esvr['test_nse'], ],
    "rmse": [h_esvr['test_nrmse'], h_vmd_esvr['test_nrmse'], h_eemd_esvr['test_nrmse'], h_ssa_esvr['test_nrmse'], h_dwt_esvr['test_nrmse'],
             x_esvr['test_nrmse'], x_vmd_esvr['test_nrmse'], x_eemd_esvr['test_nrmse'], x_ssa_esvr['test_nrmse'], x_dwt_esvr['test_nrmse'],
             z_esvr['test_nrmse'], z_vmd_esvr['test_nrmse'], z_eemd_esvr['test_nrmse'], z_ssa_esvr['test_nrmse'], z_dwt_esvr['test_nrmse'], ],

    "ppts": [h_esvr['test_ppts'], h_vmd_esvr['test_ppts'], h_eemd_esvr['test_ppts'], h_ssa_esvr['test_ppts'], h_dwt_esvr['test_ppts'],
             x_esvr['test_ppts'], x_vmd_esvr['test_ppts'], x_eemd_esvr['test_ppts'], x_ssa_esvr['test_ppts'], x_dwt_esvr['test_ppts'],
             z_esvr['test_ppts'], z_vmd_esvr['test_ppts'], z_eemd_esvr['test_ppts'], z_ssa_esvr['test_ppts'], z_dwt_esvr['test_ppts'], ],
    "time_cost": [h_esvr['time_cost'], h_vmd_esvr['time_cost'], h_eemd_esvr['time_cost'], h_ssa_esvr['time_cost'], h_dwt_esvr['time_cost'],
                  x_esvr['time_cost'], x_vmd_esvr['time_cost'], x_eemd_esvr['time_cost'], x_ssa_esvr['time_cost'], x_dwt_esvr['time_cost'],
                  z_esvr['time_cost'], z_vmd_esvr['time_cost'], z_eemd_esvr['time_cost'], z_ssa_esvr['time_cost'], z_dwt_esvr['time_cost'], ],
}

metrics_df = pd.DataFrame(metrics_dict, index=index)
print(metrics_df)
metrics_df.to_csv(results_path+"two_stage_decomposer_esvr_metrics.csv")

huaxian_nse = [h_esvr['test_nse'], h_eemd_esvr['test_nse'], h_ssa_esvr['test_nse'], h_ssa_esvr['test_nse'], h_dwt_esvr['test_nse'], ]
huaxian_nrmse = [h_esvr['test_nrmse'], h_eemd_esvr['test_nrmse'], h_ssa_esvr['test_nrmse'], h_vmd_esvr['test_nrmse'], h_dwt_esvr['test_nrmse'], ]
huaxian_ppts = [h_esvr['test_ppts'], h_eemd_esvr['test_ppts'], h_ssa_esvr['test_ppts'], h_vmd_esvr['test_ppts'], h_dwt_esvr['test_ppts'], ]
huaxian_time = [h_esvr['time_cost'], h_eemd_esvr['time_cost'],h_ssa_esvr['time_cost'], h_vmd_esvr['time_cost'], h_dwt_esvr['time_cost'], ]

xianyang_nse = [x_esvr['test_nse'], x_eemd_esvr['test_nse'], x_ssa_esvr['test_nse'], x_vmd_esvr['test_nse'], x_dwt_esvr['test_nse'], ]
xianyang_nrmse = [x_esvr['test_nrmse'], x_eemd_esvr['test_nrmse'],x_ssa_esvr['test_nrmse'], x_vmd_esvr['test_nrmse'], x_dwt_esvr['test_nrmse'], ]
xianyang_ppts = [x_esvr['test_ppts'], x_eemd_esvr['test_ppts'], x_ssa_esvr['test_ppts'], x_vmd_esvr['test_ppts'], x_dwt_esvr['test_ppts'], ]
xianyang_time = [x_esvr['time_cost'], x_eemd_esvr['time_cost'],x_ssa_esvr['time_cost'], x_vmd_esvr['time_cost'], x_dwt_esvr['time_cost'], ]

zhangjiashan_nse = [z_esvr['test_nse'], z_eemd_esvr['test_nse'], z_ssa_esvr['test_nse'], z_vmd_esvr['test_nse'], z_dwt_esvr['test_nse'], ]
zhangjiashan_nrmse = [z_esvr['test_nrmse'], z_eemd_esvr['test_nrmse'],z_ssa_esvr['test_nrmse'], z_vmd_esvr['test_nrmse'], z_dwt_esvr['test_nrmse'], ]
zhangjiashan_ppts = [z_esvr['test_ppts'], z_eemd_esvr['test_ppts'], z_ssa_esvr['test_ppts'], z_vmd_esvr['test_ppts'], z_dwt_esvr['test_ppts'], ]
zhangjiashan_time = [z_esvr['time_cost'], z_eemd_esvr['time_cost'],z_ssa_esvr['time_cost'], z_vmd_esvr['time_cost'], z_dwt_esvr['time_cost'], ]


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height, 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height/2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def autolabels(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height, 2)
        ax.text(
            x=rect.get_x() + rect.get_width() / 2,
            y=height,
            s='{}'.format(height),
            rotation=90,
            ha='center', va='bottom',
        )


#########################################################################################
# metrics_lists = [
#     [huaxian_nse, xianyang_nse, zhangjiashan_nse],
#     [huaxian_nrmse, xianyang_nrmse, zhangjiashan_nrmse],
#     [huaxian_ppts, xianyang_ppts, zhangjiashan_ppts],
#     [huaxian_time, xianyang_time, zhangjiashan_time],
# ]
# stations = ['Huaxian', 'Xianyang', 'Zhangjiashan']
# pos = [2, 4, 6, 8, 10]
# print(pos)
# width = 0.5
# action = [-1, 0, 1]
# ylims = [
#     [0, 1.2],
#     [0, 1.7],
#     [0, 3.3],
#     [0, 570],
#     [0, 90],
#     [0, 360],
# ]
# labels = ['SVR', 'EEMD-SVR', 'SSA-SVR', 'VMD-SVR', 'DWT-SVR','MODWT-SVR']
# y_labels = [
#     r"$NSE$", r"$NRMSE(10^8m^3)$", r"$PPTS(5)(\%)$", r"$Time(s)$"
# ]
# density = 5
# hatch_str = ['/'*density, 'x'*density, '|'*density]
# fig = plt.figure(figsize=(7.48, 7.48))
# for i in range(len(metrics_lists)):
#     ax = fig.add_subplot(3, 2, i+1)
#     for j in range(len(metrics_lists[i])):
#         bars = ax.bar([p+action[j]*width for p in pos],
#                       metrics_lists[i][j], width, alpha=0.5, label=stations[j])
#         for bar in bars:
#             bar.set_hatch(hatch_str[j])
#         # autolabels(bars,ax)
#     # ax.set_ylim(ylims[i])
#     ax.set_ylabel(y_labels[i])
#     ax.set_xticks(pos)
#     ax.set_xticklabels(labels, rotation=45)
#     if i == 0:
#         ax.legend(
#             loc='upper left',
#             # bbox_to_anchor=(0.08,1.01, 1,0.101),
#             bbox_to_anchor=(0.6, 1.25),
#             # bbox_transform=plt.gcf().transFigure,
#             ncol=3,
#             shadow=False,
#             frameon=True,
#         )
# plt.subplots_adjust(left=0.08, bottom=0.1, right=0.98,
#                     top=0.95, hspace=0.5, wspace=0.25)
# plt.savefig(graphs_path+'two_stage_metrics.eps', format='EPS', dpi=2000)
# plt.savefig(graphs_path+'two_stage_metrics.tif', format='TIFF', dpi=600)


# ###########################################################################################
# metrics_lists = [
#     [huaxian_nse, xianyang_nse, zhangjiashan_nse],
#     [huaxian_nrmse, xianyang_nrmse, zhangjiashan_nrmse],
#     [huaxian_ppts, xianyang_ppts, zhangjiashan_ppts],
#     [huaxian_time, xianyang_time, zhangjiashan_time],
# ]
# stations = ['Huaxian', 'Xianyang', 'Zhangjiashan']
# pos = [2, 4, 6, 8, 10]
# print(pos)
# width = 0.5
# action = [-1, 0, 1]
# ylims = [
#     [0, 1.2],
#     [0, 3.3],
#     [0, 90],
#     [0, 360],
# ]
# labels = ['SVR', 'EEMD-SVR', 'SSA-SVR', 'VMD-SVR', 'DWT-SVR']
# y_labels = [
#     r"$NSE$", r"$NRMSE(10^8m^3)$", r"$PPTS(5)(\%)$", r"$Time(s)$"
# ]
# fig = plt.figure(figsize=(7.48, 5))
# for i in range(len(metrics_lists)):
#     ax = fig.add_subplot(2, 2, i+1)
#     for j in range(len(metrics_lists[i])):
#         bars = ax.bar([p+action[j]*width for p in pos],
#                       metrics_lists[i][j], width, alpha=0.5, label=stations[j])
#         for bar in bars:
#             bar.set_hatch(hatch_str[j])
#         # autolabels(bars,ax)
#     # ax.set_ylim(ylims[i])
#     ax.set_ylabel(y_labels[i])
#     ax.set_xticks(pos)
#     ax.set_xticklabels(labels, rotation=45)
#     if i == 0:
#         ax.legend(
#             loc='upper left',
#             # bbox_to_anchor=(0.08,1.01, 1,0.101),
#             bbox_to_anchor=(0.6, 1.17),
#             # bbox_transform=plt.gcf().transFigure,
#             ncol=3,
#             shadow=False,
#             frameon=True,
#         )
# plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98,
#                     top=0.95, hspace=0.5, wspace=0.25)
# plt.savefig(graphs_path+'two_stage_NSE_NRMSE_PPTS_TIMECOST.eps',
#             format='EPS', dpi=2000)
# plt.savefig(graphs_path+'two_stage_NSE_NRMSE_PPTS_TIMECOST.tif',
#             format='TIFF', dpi=600)


###########################################################################################
nse_data = [
    [h_arma['test_nse'], x_arma['test_nse'], z_arma['test_nse']],
    [h_esvr['test_nse'], x_esvr['test_nse'], z_esvr['test_nse']],
    [h_modwt_esvr['test_nse'], x_modwt_esvr['test_nse'], z_modwt_esvr['test_nse']],
    [h_eemd_esvr['test_nse'], x_eemd_esvr['test_nse'], z_eemd_esvr['test_nse']],
    [h_ssa_esvr['test_nse'], x_ssa_esvr['test_nse'], z_ssa_esvr['test_nse']],
    [h_dwt_esvr['test_nse'], x_dwt_esvr['test_nse'], z_dwt_esvr['test_nse']],
    [h_vmd_esvr['test_nse'], x_vmd_esvr['test_nse'], z_vmd_esvr['test_nse']],
]

mean_nse = [
    sum([h_arma['test_nse'], x_arma['test_nse'], z_arma['test_nse']])/len([h_arma['test_nse'], x_arma['test_nse'], z_arma['test_nse']]),
    sum([h_esvr['test_nse'], x_esvr['test_nse'], z_esvr['test_nse']])/len([h_esvr['test_nse'], x_esvr['test_nse'], z_esvr['test_nse']]),
    sum([h_modwt_esvr['test_nse'], x_modwt_esvr['test_nse'], z_modwt_esvr['test_nse']])/len([h_modwt_esvr['test_nse'], x_modwt_esvr['test_nse'], z_modwt_esvr['test_nse']]),
    sum([h_eemd_esvr['test_nse'], x_eemd_esvr['test_nse'], z_eemd_esvr['test_nse']])/len([h_eemd_esvr['test_nse'], x_eemd_esvr['test_nse'], z_eemd_esvr['test_nse']]),
    sum([h_ssa_esvr['test_nse'], x_ssa_esvr['test_nse'], z_ssa_esvr['test_nse']])/len([h_ssa_esvr['test_nse'], x_ssa_esvr['test_nse'], z_ssa_esvr['test_nse']]),
    sum([h_dwt_esvr['test_nse'], x_dwt_esvr['test_nse'], z_dwt_esvr['test_nse']])/len([h_dwt_esvr['test_nse'], x_dwt_esvr['test_nse'], z_dwt_esvr['test_nse']]),
    sum([h_vmd_esvr['test_nse'], x_vmd_esvr['test_nse'], z_vmd_esvr['test_nse']])/len([h_vmd_esvr['test_nse'], x_vmd_esvr['test_nse'], z_vmd_esvr['test_nse']]),
]

for i in range(1, len(mean_nse)):
    print("Compared with mean NSE of SVR\nEEMD-SVR, SSA-SVR, VMD-SVR and DWT-SVR reduced by {}%".format(
        (mean_nse[i]-mean_nse[0])/mean_nse[0]*100))

nrmse_data = [
    [h_arma['test_nrmse'], x_arma['test_nrmse'], z_arma['test_nrmse']],
    [h_esvr['test_nrmse'], x_esvr['test_nrmse'], z_esvr['test_nrmse']],
    [h_modwt_esvr['test_nrmse'], x_modwt_esvr['test_nrmse'], z_modwt_esvr['test_nrmse']],
    [h_eemd_esvr['test_nrmse'], x_eemd_esvr['test_nrmse'], z_eemd_esvr['test_nrmse']],
    [h_ssa_esvr['test_nrmse'], x_ssa_esvr['test_nrmse'], z_ssa_esvr['test_nrmse']],
    [h_dwt_esvr['test_nrmse'], x_dwt_esvr['test_nrmse'], z_dwt_esvr['test_nrmse']],
    [h_vmd_esvr['test_nrmse'], x_vmd_esvr['test_nrmse'], z_vmd_esvr['test_nrmse']],
]

mean_nrmse = [
    sum([h_arma['test_nrmse'], x_arma['test_nrmse'], z_arma['test_nrmse']])/len([h_arma['test_nrmse'], x_arma['test_nrmse'], z_arma['test_nrmse']]),
    sum([h_esvr['test_nrmse'], x_esvr['test_nrmse'], z_esvr['test_nrmse']])/len([h_esvr['test_nrmse'], x_esvr['test_nrmse'], z_esvr['test_nrmse']]),
    sum([h_modwt_esvr['test_nrmse'], x_modwt_esvr['test_nrmse'], z_modwt_esvr['test_nrmse']]) /len([h_modwt_esvr['test_nrmse'], x_modwt_esvr['test_nrmse'], z_modwt_esvr['test_nrmse']]),
    sum([h_eemd_esvr['test_nrmse'], x_eemd_esvr['test_nrmse'], z_eemd_esvr['test_nrmse']]) /len([h_eemd_esvr['test_nrmse'], x_eemd_esvr['test_nrmse'], z_eemd_esvr['test_nrmse']]),
    sum([h_ssa_esvr['test_nrmse'], x_ssa_esvr['test_nrmse'], z_ssa_esvr['test_nrmse']]) / len([h_ssa_esvr['test_nrmse'], x_ssa_esvr['test_nrmse'], z_ssa_esvr['test_nrmse']]),
    sum([h_dwt_esvr['test_nrmse'], x_dwt_esvr['test_nrmse'], z_dwt_esvr['test_nrmse']]) /len([h_dwt_esvr['test_nrmse'], x_dwt_esvr['test_nrmse'], z_dwt_esvr['test_nrmse']]),
    sum([h_vmd_esvr['test_nrmse'], x_vmd_esvr['test_nrmse'], z_vmd_esvr['test_nrmse']]) /len([h_vmd_esvr['test_nrmse'], x_vmd_esvr['test_nrmse'], z_vmd_esvr['test_nrmse']]),
]

for i in range(1, len(mean_nrmse)):
    print("Compared with mean NRMSE of SVR\nEEMD-SVR, SSA-SVR, VMD-SVR and DWT-SVR reduced by {}%".format(
        (mean_nrmse[i]-mean_nrmse[0])/mean_nrmse[0]*100))

ppts_data = [
    [h_arma['test_ppts'], x_arma['test_ppts'], z_arma['test_ppts']],
    [h_esvr['test_ppts'], x_esvr['test_ppts'], z_esvr['test_ppts']],
    [h_modwt_esvr['test_ppts'], x_modwt_esvr['test_ppts'], z_modwt_esvr['test_ppts']],
    [h_eemd_esvr['test_ppts'], x_eemd_esvr['test_ppts'], z_eemd_esvr['test_ppts']],
    [h_ssa_esvr['test_ppts'], x_ssa_esvr['test_ppts'], z_ssa_esvr['test_ppts']],
    [h_dwt_esvr['test_ppts'], x_dwt_esvr['test_ppts'], z_dwt_esvr['test_ppts']],
    [h_vmd_esvr['test_ppts'], x_vmd_esvr['test_ppts'], z_vmd_esvr['test_ppts']],
]

mean_ppts=[
    sum([h_arma['test_ppts'], x_arma['test_ppts'], z_arma['test_ppts']])/len([h_arma['test_ppts'], x_arma['test_ppts'], z_arma['test_ppts']]),
    sum([h_esvr['test_ppts'], x_esvr['test_ppts'], z_esvr['test_ppts']])/len([h_esvr['test_ppts'], x_esvr['test_ppts'], z_esvr['test_ppts']]),
    sum([h_modwt_esvr['test_ppts'], x_modwt_esvr['test_ppts'], z_modwt_esvr['test_ppts']])/len([h_modwt_esvr['test_ppts'], x_modwt_esvr['test_ppts'], z_modwt_esvr['test_ppts']]),
    sum([h_eemd_esvr['test_ppts'], x_eemd_esvr['test_ppts'], z_eemd_esvr['test_ppts']])/len([h_eemd_esvr['test_ppts'], x_eemd_esvr['test_ppts'], z_eemd_esvr['test_ppts']]),
    sum([h_ssa_esvr['test_ppts'], x_ssa_esvr['test_ppts'], z_ssa_esvr['test_ppts']])/len([h_ssa_esvr['test_ppts'], x_ssa_esvr['test_ppts'], z_ssa_esvr['test_ppts']]),
    sum([h_dwt_esvr['test_ppts'], x_dwt_esvr['test_ppts'], z_dwt_esvr['test_ppts']])/len([h_dwt_esvr['test_ppts'], x_dwt_esvr['test_ppts'], z_dwt_esvr['test_ppts']]),
    sum([h_vmd_esvr['test_ppts'], x_vmd_esvr['test_ppts'], z_vmd_esvr['test_ppts']])/len([h_vmd_esvr['test_ppts'], x_vmd_esvr['test_ppts'], z_vmd_esvr['test_ppts']]),
]

for i in range(1, len(mean_ppts)):
    print("Compared with mean PPTS of SVR\nEEMD-SVR, SSA-SVR, VMD-SVR and DWT-SVR reduced by {}%".format(
        (mean_ppts[i]-mean_ppts[0])/mean_ppts[0]*100))

timecost_data = [
    [h_arma['time_cost'], x_arma['time_cost'], z_arma['time_cost']],
    [h_esvr['time_cost'], x_esvr['time_cost'], z_esvr['time_cost']],
    [h_modwt_esvr['time_cost'], x_modwt_esvr['time_cost'], z_modwt_esvr['time_cost']],
    [h_eemd_esvr['time_cost'], x_eemd_esvr['time_cost'], z_eemd_esvr['time_cost']],
    [h_ssa_esvr['time_cost'], x_ssa_esvr['time_cost'], z_ssa_esvr['time_cost']],
    [h_dwt_esvr['time_cost'], x_dwt_esvr['time_cost'], z_dwt_esvr['time_cost']],
    [h_vmd_esvr['time_cost'], x_vmd_esvr['time_cost'], z_vmd_esvr['time_cost']],
]
all_datas = [
    nse_data, nrmse_data, ppts_data, timecost_data
]

labels = [ 'ARMA','SVR','WDDFF(MODWT-SVR)','TSDP(EEMD-SVR)', 'TSDP(SSA-SVR)', 'TSDP(DWT-SVR)', 'TSDP(VMD-SVR)']
y_labels = [
    r"$NSE$", r"$NRMSE(10^8m^3)$", r"$PPTS(5)(\%)$", r"$Time(s)$"
]
x = list(range(7))
plt.figure(figsize=(3.54, 5.54))
x_s = [-0.38, 3.8, -0.38, 3.8]
y_s = [0.92, 1.28, 3, 212]
for i in range(len(all_datas)):
    ax1 = plt.subplot(4, 1, i+1)
    ax1.yaxis.grid(True)
    vplot1 = plt.violinplot(
        dataset=all_datas[i],
        positions=x,
        showmeans=True,
    )
    plt.ylabel(y_labels[i])
    if i==3:
        plt.xticks(x, labels, rotation=45)
    else:
        plt.xticks([])
    for pc in vplot1['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
plt.tight_layout()
plt.savefig(graphs_path+'/TSDP violin metrics.eps',format='EPS', dpi=2000)
plt.savefig(graphs_path+'/TSDP violin metrics.tif',format='TIFF', dpi=1200)
plt.savefig(graphs_path+'/TSDP violin metrics.pdf',format='PDF', dpi=1200)
plt.show()
