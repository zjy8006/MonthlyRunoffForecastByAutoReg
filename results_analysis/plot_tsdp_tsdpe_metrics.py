import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.size'] = 6
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/graphs/'
results_path = root_path+'/results_analysis/results/'
print("root path:{}".format(root_path))
sys.path.append(root_path)
from tools.results_reader import read_two_stage, read_pure_esvr,read_pure_arma

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


nse_data = [
    [h_eemd_1['test_nse'][0],x_eemd_1['test_nse'][0],z_eemd_1['test_nse'][0]],
    [h_eemda_1['test_nse'][0],x_eemda_1['test_nse'][0],z_eemda_1['test_nse'][0]],
    [h_ssa_1['test_nse'][0],x_ssa_1['test_nse'][0],z_ssa_1['test_nse'][0]],
    [h_ssaa_1['test_nse'][0],x_ssaa_1['test_nse'][0],z_ssaa_1['test_nse'][0]],
    [h_dwt_1['test_nse'][0],x_dwt_1['test_nse'][0],z_dwt_1['test_nse'][0]],
    [h_dwta_1['test_nse'][0],x_dwta_1['test_nse'][0],z_dwta_1['test_nse'][0]],
    [h_vmd_1['test_nse'][0],x_vmd_1['test_nse'][0],z_vmd_1['test_nse'][0]],
    [h_vmda_1['test_nse'][0],x_vmda_1['test_nse'][0],z_vmda_1['test_nse'][0]],
]

nrmse_data = [
    [h_eemd_1['test_nrmse'][0],x_eemd_1['test_nrmse'][0],z_eemd_1['test_nrmse'][0]],
    [h_eemda_1['test_nrmse'][0],x_eemda_1['test_nrmse'][0],z_eemda_1['test_nrmse'][0]],
    [h_ssa_1['test_nrmse'][0],x_ssa_1['test_nrmse'][0],z_ssa_1['test_nrmse'][0]],
    [h_ssaa_1['test_nrmse'][0],x_ssaa_1['test_nrmse'][0],z_ssaa_1['test_nrmse'][0]],
    [h_dwt_1['test_nrmse'][0],x_dwt_1['test_nrmse'][0],z_dwt_1['test_nrmse'][0]],
    [h_dwta_1['test_nrmse'][0],x_dwta_1['test_nrmse'][0],z_dwta_1['test_nrmse'][0]],
    [h_vmd_1['test_nrmse'][0],x_vmd_1['test_nrmse'][0],z_vmd_1['test_nrmse'][0]],
    [h_vmda_1['test_nrmse'][0],x_vmda_1['test_nrmse'][0],z_vmda_1['test_nrmse'][0]],
]

ppts_data = [
    [h_eemd_1['test_ppts'][0],x_eemd_1['test_ppts'][0],z_eemd_1['test_ppts'][0]],
    [h_eemda_1['test_ppts'][0],x_eemda_1['test_ppts'][0],z_eemda_1['test_ppts'][0]],
    [h_ssa_1['test_ppts'][0],x_ssa_1['test_ppts'][0],z_ssa_1['test_ppts'][0]],
    [h_ssaa_1['test_ppts'][0],x_ssaa_1['test_ppts'][0],z_ssaa_1['test_ppts'][0]],
    [h_dwt_1['test_ppts'][0],x_dwt_1['test_ppts'][0],z_dwt_1['test_ppts'][0]],
    [h_dwta_1['test_ppts'][0],x_dwta_1['test_ppts'][0],z_dwta_1['test_ppts'][0]],
    [h_vmd_1['test_ppts'][0],x_vmd_1['test_ppts'][0],z_vmd_1['test_ppts'][0]],
    [h_vmda_1['test_ppts'][0],x_vmda_1['test_ppts'][0],z_vmda_1['test_ppts'][0]],
]

time_cost=[
    [h_eemd_1['time_cost'][0],x_eemd_1['time_cost'][0],z_eemd_1['time_cost'][0]],
    [h_eemda_1['time_cost'][0],x_eemda_1['time_cost'][0],z_eemda_1['time_cost'][0]],
    [h_ssa_1['time_cost'][0],x_ssa_1['time_cost'][0],z_ssa_1['time_cost'][0]],
    [h_ssaa_1['time_cost'][0],x_ssaa_1['time_cost'][0],z_ssaa_1['time_cost'][0]],
    [h_dwt_1['time_cost'][0],x_dwt_1['time_cost'][0],z_dwt_1['time_cost'][0]],
    [h_dwta_1['time_cost'][0],x_dwta_1['time_cost'][0],z_dwta_1['time_cost'][0]],
    [h_vmd_1['time_cost'][0],x_vmd_1['time_cost'][0],z_vmd_1['time_cost'][0]],
    [h_vmda_1['time_cost'][0],x_vmda_1['time_cost'][0],z_vmda_1['time_cost'][0]],
]
face_colors={
    0:'#00FF9F',#eemd-svr
    1:'#00FF9F',#eemd-svr-a
    2:'#FFE4E1',#ssa-svr
    3:'#FFE4E1',#ssa-svr-a
    4:'#FF4500',#vmd-svr
    5:'#FF4500',#vmd-svr-a
    6:'#FAA460',#dwt-svr
    7:'#FAA460',#dwt-svr-a
}

labels = [ 'EEMD-SVR','EEMD-SVR-A','SSA-SVR','SSA-SVR-A', 'DWT-SVR', 'DWT-SVR-A', 'VMD-SVR','VMD-SVR-A']
y_labels = [
    r"$NSE$", r"$NRMSE\ (10^8m^3)$", r"$PPTS(5)\ (\%)$", r"$Time\ (s)$"
]
x = list(range(len(nse_data)))
plt.figure(figsize=(7.48, 3.48))
x_s=[-0.4,7.1,-0.4,7.1]
y_s = [0.88, 1.62, 3, 170]
fig_id=['(a)','(b)','(c)','(d)']
all_datas=[nse_data,nrmse_data,ppts_data,time_cost]
for i in range(len(all_datas)):
    ax1 = plt.subplot(2, 2, i+1)
    if i in [0,1]:
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    elif i==2:
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.text(x_s[i],y_s[i],fig_id[i],fontsize=7)
    ax1.yaxis.grid(True)
    vplot1 = plt.violinplot(
        dataset=all_datas[i],
        positions=x,
        showmeans=True,
    )
    plt.ylabel(y_labels[i])
    if i==3 or i==2:
        plt.xticks(x, labels, rotation=45)
    else:
        plt.xticks([])
    for pc,k in zip(vplot1['bodies'],range(len(face_colors))):
        print(pc)
        # pc.set_facecolor('#D43F3A')
        pc.set_facecolor(face_colors[k])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
plt.subplots_adjust(left=0.066, bottom=0.16, right=0.99,top=0.99, hspace=0.05, wspace=0.2)
plt.savefig(graphs_path+'/TSDP and TSDPE violin metrics.eps',format='EPS', dpi=2000)
plt.savefig(graphs_path+'/TSDP and TSDPE violin metrics.tif',format='TIFF', dpi=500)
plt.savefig(graphs_path+'/TSDP and TSDPE violin metrics.pdf',format='PDF', dpi=1200)
plt.show()
