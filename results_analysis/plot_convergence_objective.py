import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
from skopt import dump, load
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.skopt_plots import plot_convergence,plot_objective,plot_evaluations


graphs_path = root_path+'/graphs/'
res = load(root_path+'/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/result.pkl')
fig = plt.figure(num=1,figsize=(3.54,2.51))
ax0 = fig.add_subplot(111)
plot_convergence(res,ax=ax0, true_minimum=0.0,)
plt.title("")
ax0.set_ylabel(r"Minimum $MSE$ after $n$ calls")
plt.tight_layout()
plt.savefig(graphs_path+'convergence_huaxian_vmd.eps',format="EPS",dpi=2000)
plt.savefig(graphs_path+'convergence_huaxian_vmd.tif',format="TIFF",dpi=1200)

# plot_objective(res,figsize=(4.5,4.5),dimensions=[r'$C$',r'$\epsilon$',r'$\sigma$'])
plot_objective(res,figsize=(3.5717,4.2),dimensions=[r'$C$',r'$\epsilon$',r'$\sigma$'])
plt.subplots_adjust(left=0.11, bottom=0.12, right=0.89, top=0.92, hspace=0.5, wspace=0.1)
# plt.tight_layout()
plt.savefig(graphs_path+'Fig.7.Objective of VMD-SVR at Huaxian.eps',format="EPS",dpi=2000)
plt.savefig(graphs_path+'Fig.7.Objective of VMD-SVR at Huaxian.tif',format="TIFF",dpi=1200)
plt.savefig(graphs_path+'Fig.7.Objective of VMD-SVR at Huaxian.pdf',format="PDF",dpi=1200)
plt.show()