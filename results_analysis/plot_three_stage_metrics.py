import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/graphs/'
import sys
sys.path.append(root_path)
from tools.results_reader import read_two_stage,read_pure_esvr,read_pure_arma

h_arma = read_pure_arma("Huaxian")
x_arma = read_pure_arma("Xianyang")
z_arma = read_pure_arma("Zhangjiashan")

h_esvr=read_pure_esvr("Huaxian")
x_esvr=read_pure_esvr("Xianyang")
z_esvr=read_pure_esvr("Zhangjiashan")

huaxian_eemd = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
huaxian_ssa = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
huaxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
huaxian_dwt = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')

xianyang_eemd = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
xianyang_ssa = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
xianyang_vmd = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
xianyang_dwt = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')

zhangjiashan_eemd = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
zhangjiashan_ssa = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
zhangjiashan_vmd = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/multi_step_1_ahead_forecast_pacf/optimal_results.csv')
zhangjiashan_dwt = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/multi_step_1_ahead_forecast_pacf/optimal_results.csv')

huaxian_nse = [h_arma['test_nse'],h_esvr['test_nse'],huaxian_eemd['test_nse'][0],huaxian_ssa['test_nse'][0],huaxian_dwt['test_nse'][0],huaxian_vmd['test_nse'][0],]
huaxian_nrmse = [h_arma['test_nrmse'],h_esvr['test_nrmse'],huaxian_eemd['test_nrmse'][0],huaxian_ssa['test_nrmse'][0],huaxian_dwt['test_nrmse'][0],huaxian_vmd['test_nrmse'][0],]
huaxian_ppts = [h_arma['test_ppts'],h_esvr['test_ppts'],huaxian_eemd['test_ppts'][0],huaxian_ssa['test_ppts'][0],huaxian_dwt['test_ppts'][0],huaxian_vmd['test_ppts'][0],]
huaxian_time = [h_arma['time_cost'],h_esvr['time_cost'],huaxian_eemd['time_cost'][0],huaxian_ssa['time_cost'][0],huaxian_dwt['time_cost'][0],huaxian_vmd['time_cost'][0],]

xianyang_nse = [x_arma['test_nse'],x_esvr['test_nse'],xianyang_eemd['test_nse'][0],xianyang_ssa['test_nse'][0],xianyang_dwt['test_nse'][0],xianyang_vmd['test_nse'][0],]
xianyang_nrmse = [x_arma['test_nrmse'],x_esvr['test_nrmse'],xianyang_eemd['test_nrmse'][0],xianyang_ssa['test_nrmse'][0],xianyang_dwt['test_nrmse'][0],xianyang_vmd['test_nrmse'][0],]
xianyang_ppts = [x_arma['test_ppts'],x_esvr['test_ppts'],xianyang_eemd['test_ppts'][0],xianyang_ssa['test_ppts'][0],xianyang_dwt['test_ppts'][0],xianyang_vmd['test_ppts'][0],]
xianyang_time = [x_arma['time_cost'],x_esvr['time_cost'],xianyang_eemd['time_cost'][0],xianyang_ssa['time_cost'][0],xianyang_dwt['time_cost'][0],xianyang_vmd['time_cost'][0],]

zhangjiashan_nse = [z_arma['test_nse'],z_esvr['test_nse'],zhangjiashan_eemd['test_nse'][0],zhangjiashan_ssa['test_nse'][0],zhangjiashan_dwt['test_nse'][0],zhangjiashan_vmd['test_nse'][0],]
zhangjiashan_nrmse = [z_arma['test_nrmse'],z_esvr['test_nrmse'],zhangjiashan_eemd['test_nrmse'][0],zhangjiashan_ssa['test_nrmse'][0],zhangjiashan_dwt['test_nrmse'][0],zhangjiashan_vmd['test_nrmse'][0],]
zhangjiashan_ppts = [z_arma['test_ppts'],z_esvr['test_ppts'],zhangjiashan_eemd['test_ppts'][0],zhangjiashan_ssa['test_ppts'][0],zhangjiashan_dwt['test_ppts'][0],zhangjiashan_vmd['test_ppts'][0],]
zhangjiashan_time = [z_arma['time_cost'],z_esvr['time_cost'],zhangjiashan_eemd['time_cost'][0],zhangjiashan_ssa['time_cost'][0],zhangjiashan_dwt['time_cost'][0],zhangjiashan_vmd['time_cost'][0],]

def autolabels(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height,2)
        ax.text(
            x=rect.get_x() + rect.get_width() / 2,
            y=height,
            s='{}'.format(height),
            rotation=90,
            ha='center', va='bottom',
                    )

nse_data = [
    [h_arma['test_nse'],x_arma['test_nse'],z_arma['test_nse']],
    [h_esvr['test_nse'],x_esvr['test_nse'],z_esvr['test_nse']],
    [huaxian_eemd['test_nse'][0],xianyang_eemd["test_nse"][0],zhangjiashan_eemd["test_nse"][0]],
    [huaxian_ssa["test_nse"][0],xianyang_ssa["test_nse"][0],zhangjiashan_ssa["test_nse"][0]],
    [huaxian_dwt["test_nse"][0],xianyang_dwt["test_nse"][0],zhangjiashan_dwt["test_nse"][0]],
    [huaxian_vmd["test_nse"][0],xianyang_vmd["test_nse"][0],zhangjiashan_vmd["test_nse"][0]],
]

mean_nse =[]
for i in range(len(nse_data)):
    mean_nse.append(sum(nse_data[i])/len(nse_data[i]))
for i in range(1,len(mean_nse)):
    print("Compared with SVR, mean NSE increased by {}%".format((mean_nse[i]-mean_nse[0])/mean_nse[0]*100))

nrmse_data = [
    [h_arma['test_nrmse'],x_arma['test_nrmse'],z_arma['test_nrmse']],
    [h_esvr['test_nrmse'],x_esvr['test_nrmse'],z_esvr['test_nrmse']],
    [huaxian_eemd["test_nrmse"][0],xianyang_eemd["test_nrmse"][0],zhangjiashan_eemd["test_nrmse"][0]],
    [huaxian_ssa["test_nrmse"][0],xianyang_ssa["test_nrmse"][0],zhangjiashan_ssa["test_nrmse"][0]],
    [huaxian_dwt["test_nrmse"][0],xianyang_dwt["test_nrmse"][0],zhangjiashan_dwt["test_nrmse"][0]],
    [huaxian_vmd["test_nrmse"][0],xianyang_vmd["test_nrmse"][0],zhangjiashan_vmd["test_nrmse"][0]],
]

mean_nrmse =[]
for i in range(len(nrmse_data)):
    mean_nrmse.append(sum(nrmse_data[i])/len(nrmse_data[i]))
for i in range(1,len(mean_nrmse)):
    print("Compared with SVR, mean NRMSE increased by {}%".format((mean_nrmse[i]-mean_nrmse[0])/mean_nrmse[0]*100))

ppts_data = [
    [h_arma['test_ppts'],x_arma['test_ppts'],z_arma['test_ppts']],
    [h_esvr['test_ppts'],x_esvr['test_ppts'],z_esvr['test_ppts']],
    [huaxian_eemd["test_ppts"][0],xianyang_eemd["test_ppts"][0],zhangjiashan_eemd["test_ppts"][0]],
    [huaxian_ssa["test_ppts"][0],xianyang_ssa["test_ppts"][0],zhangjiashan_ssa["test_ppts"][0]],
    [huaxian_dwt["test_ppts"][0],xianyang_dwt["test_ppts"][0],zhangjiashan_dwt["test_ppts"][0]],
    [huaxian_vmd["test_ppts"][0],xianyang_vmd["test_ppts"][0],zhangjiashan_vmd["test_ppts"][0]],
]

mean_ppts =[]
for i in range(len(ppts_data)):
    mean_ppts.append(sum(ppts_data[i])/len(ppts_data[i]))
for i in range(1,len(mean_ppts)):
    print("Compared with SVR, mean PPTS increased by {}%".format((mean_ppts[i]-mean_ppts[0])/mean_ppts[0]*100))

timecost_data = [
    [h_arma['time_cost'],x_arma['time_cost'],z_arma['time_cost']],
    [h_esvr['time_cost'],x_esvr['time_cost'],z_esvr['time_cost']],
    [huaxian_eemd["time_cost"][0],xianyang_eemd["time_cost"][0],zhangjiashan_eemd["time_cost"][0]],
    [huaxian_ssa["time_cost"][0],xianyang_ssa["time_cost"][0],zhangjiashan_ssa["time_cost"][0]],
    [huaxian_dwt["time_cost"][0],xianyang_dwt["time_cost"][0],zhangjiashan_dwt["time_cost"][0]],
    [huaxian_vmd["time_cost"][0],xianyang_vmd["time_cost"][0],zhangjiashan_vmd["time_cost"][0]],
]

mean_time =[]
for i in range(len(timecost_data)):
    mean_time.append(sum(timecost_data[i])/len(timecost_data[i]))
for i in range(1,len(mean_time)):
    print("Compared with SVR, mean TIME increased by {}%".format((mean_time[i]-mean_time[0])/mean_time[0]*100))

all_datas = [
    nse_data,nrmse_data,ppts_data,timecost_data
]



x = list(range(6))
x_s = [-0.38, -0.38, -0.38, -0.38]
y_s = [0.9, 1.4, 65, 3100]
fig_ids = ['(a)', '(b)', '(c)', '(d)']
labels=['ARMA','SVR','EEMD-SVR-A','SSA-SVR-A','DWT-SVR-A','VMD-SVR-A']
y_labels=[
    r"$NSE$",r"$NRMSE(10^8m^3)$",r"$PPTS(5)(\%)$",r"$Time(s)$"
]
plt.figure(figsize=(3.54, 5.54))
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
    plt.text(x_s[i], y_s[i], fig_ids[i], fontweight='normal', fontsize=7)
    for pc in vplot1['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
plt.tight_layout()
# plt.subplots_adjust(left=0.14, bottom=0.18, right=0.96,top=0.98, hspace=0.6, wspace=0.45)
plt.savefig(graphs_path+'/TSDPE violin metrics.eps',format='EPS', dpi=2000)
plt.savefig(graphs_path+'/TSDPE violin metrics.tif',format='TIFF', dpi=1200)
plt.savefig(graphs_path+'/TSDPE violin metrics.pdf',format='PDF', dpi=1200)
plt.show()
