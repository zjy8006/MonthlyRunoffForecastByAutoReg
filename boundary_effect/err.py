
import pandas as pd
import numpy as np
from sklearn.svm import SVR,NuSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.externals.joblib import Parallel, delayed
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize,forest_minimize, dummy_minimize
from skopt.plots import plot_convergence,plot_objective,plot_evaluations
from skopt import dump, load
from skopt import Optimizer
from skopt.benchmarks import branin
from functools import partial
from statsmodels.tsa.arima_model import ARIMA
from random import seed
from random import random
seed(1)
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.rcParams['font.size'] = 6
import os
import sys
root_path = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(root_path)
from tools.fit_line import compute_linear_fit
graphs_path = root_path+'/graphs/'
if not os.path.exists(graphs_path):
    os.makedirs(graphs_path)

vmd_train = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_TRAIN.csv")
vmd_full = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_FULL.csv")
seq_val_dec = pd.DataFrame()
for subsignal in ['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8']:
    test_imf = []
    for i in range(553,792+1):
        data=pd.read_csv(root_path+"/Huaxian_vmd/data/vmd-test/vmd_appended_test"+str(i)+".csv")
        test_imf.append((data[subsignal].iloc[data.shape[0]-1:]).values.flatten()[0])
    val_subsignal = pd.DataFrame(test_imf,columns=[subsignal])
    seq_val_dec = pd.concat([seq_val_dec,val_subsignal],axis=1)
seq_val_dec_sum = seq_val_dec.sum(axis=1)
t_e=list(range(1,793))
t_t=list(range(1,553))



orig = (vmd_full['ORIG'].iloc[vmd_full.shape[0]-240:]).values
con_imf1 = (vmd_full['IMF1'].iloc[vmd_full.shape[0]-240:]).values
con_imf2 = (vmd_full['IMF2'].iloc[vmd_full.shape[0]-240:]).values
con_imf3 = (vmd_full['IMF3'].iloc[vmd_full.shape[0]-240:]).values
con_imf4 = (vmd_full['IMF4'].iloc[vmd_full.shape[0]-240:]).values
con_imf5 = (vmd_full['IMF5'].iloc[vmd_full.shape[0]-240:]).values
con_imf6 = (vmd_full['IMF6'].iloc[vmd_full.shape[0]-240:]).values
con_imf7 = (vmd_full['IMF7'].iloc[vmd_full.shape[0]-240:]).values
con_imf8 = (vmd_full['IMF8'].iloc[vmd_full.shape[0]-240:]).values
seq_imf1=(seq_val_dec['IMF1']).values
seq_imf2=(seq_val_dec['IMF2']).values
seq_imf3=(seq_val_dec['IMF3']).values
seq_imf4=(seq_val_dec['IMF4']).values
seq_imf5=(seq_val_dec['IMF5']).values
seq_imf6=(seq_val_dec['IMF6']).values
seq_imf7=(seq_val_dec['IMF7']).values
seq_imf8=(seq_val_dec['IMF8']).values
err_imf1 = seq_imf1 - con_imf1
err_imf2 = seq_imf2 - con_imf2
err_imf3 = seq_imf3 - con_imf3
err_imf4 = seq_imf4 - con_imf4
err_imf5 = seq_imf5 - con_imf5
err_imf6 = seq_imf6 - con_imf6
err_imf7 = seq_imf7 - con_imf7
err_imf8 = seq_imf8 - con_imf8

samples = {
    'X1':err_imf1,
    'X2':err_imf2,
    'X3':err_imf3,
    'X4':err_imf4,
    'X5':err_imf5,
    'X6':err_imf6,
    'X7':err_imf7,
    'X8':err_imf8,
    'Y':orig,
}

samples = pd.DataFrame(samples)
cal = samples[:192]
val = samples[192:]
cal = cal.reset_index(drop=True)
val = val.reset_index(drop=True)

sample_X=samples.drop('Y',axis=1)
print('sample_X.sum(axis=0)={}'.format(sample_X.sum(axis=1)))

sMax=cal.max(axis=0)
sMin=cal.min(axis=0)
print('sMax={}'.format(sMax))
print('sMin={}'.format(sMin))
cal = 2 * (cal - sMin) / (sMax - sMin) - 1
val = 2 * (val - sMin) / (sMax - sMin) - 1
cal_y=cal['Y']
cal_X=cal.drop('Y',axis=1)
val_y=val['Y']
val_X=val.drop('Y',axis=1)

reg = SVR(tol=1e-4)
space = [
    Real(0.1, 200, name='C'),   
    Real(10**-6, 10**0, name='epsilon'),    
    Real(10**-6, 10**0, name='gamma'),  
]
@use_named_args(space)
def objective(**params):
    reg.set_params(**params)
    return -np.mean(cross_val_score(reg,cal_X,cal_y,cv=10,n_jobs=-1,scoring='neg_mean_squared_error'))
res = gp_minimize(objective,space,n_calls=100 ,random_state=0,verbose=True,n_jobs=-1)
model = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2])
cal_pre=model.fit(cal_X,cal_y).predict(cal_X)
val_pre=model.fit(cal_X,cal_y).predict(val_X)
sMax=sMax[sMax.shape[0]-1]
sMin=sMin[sMin.shape[0]-1]
print('sMax={}'.format(sMax))
print('sMin={}'.format(sMin))
cal_pre = np.multiply(cal_pre + 1, sMax -sMin) / 2 + sMin
cal_y = np.multiply(cal_y + 1,sMax - sMin) / 2 + sMin
val_pre = np.multiply(val_pre + 1, sMax -sMin) / 2 + sMin
val_y = np.multiply(val_y + 1,sMax - sMin) / 2 + sMin

cal_nse = r2_score(cal_y, cal_pre)
val_nse = r2_score(val_y, val_pre)
print('cal_nse={}'.format(cal_nse))
print('val_nse={}'.format(val_nse))

plt.figure(figsize=(3.54,2))
plt.subplot(1,2,1,aspect='equal')
xx,linear_fit,xymin,xymax=compute_linear_fit(cal_y,cal_pre)
plt.scatter(cal_pre,cal_y,c='tab:blue',edgecolors='black',label='calibration')
plt.plot(xx, linear_fit, '--', color='red',linewidth=1.0)
# plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
plt.xlim([xymin,xymax])
plt.ylim([xymin,xymax])
plt.xlabel(r'Predictions($10^8m^3$)', )
plt.ylabel(r'Records($10^8m^3$)', )
plt.legend()
plt.subplot(1,2,2,aspect='equal')
xx,linear_fit,xymin,xymax=compute_linear_fit(val_y,val_pre)
plt.scatter(val_pre,val_y,c='tab:red',edgecolors='black',label='validation')
plt.plot(xx, linear_fit, '--', color='red',linewidth=1.0)
# plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
plt.xlim([xymin,xymax])
plt.ylim([xymin,xymax])
plt.xlabel(r'Predictions($10^8m^3$)', )
plt.legend()
plt.subplots_adjust(left=0.12, bottom=0.15, right=0.99,top=0.99, hspace=0.15, wspace=0.15)
plt.savefig(graphs_path+'/scatter of error model at Huaxian.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/scatter of error model at Huaxian.tif',format='TIFF',dpi=500)
plt.savefig(graphs_path+'/scatter of error model at Huaxian.pdf',format='PDF',dpi=1200)






con=[con_imf1,con_imf2,con_imf3,con_imf4,con_imf5,con_imf6,con_imf7,con_imf8]
seq=[seq_imf1,seq_imf2,seq_imf3,seq_imf4,seq_imf5,seq_imf6,seq_imf7,seq_imf8]

con_df = pd.DataFrame(con)
sum_con = con_df.sum(axis=0).values


# 两种分解与原始的差异变化一致
plt.figure(figsize=(7.48,4.))
for i in range(len(con)):
    plt.subplot(2,4,i+1,)
    # plt.xlim(0,32)
    # plt.ylim(0,32)
    if i==0 or i==4:
        plt.ylabel(r"Runoff($10^8m^3$)")
    # else:
    #     plt.yticks([])
    if i in [4,5,6,7]:
        plt.xlabel(r"Runoff($10^8m^3$)")
    # else:
    #     plt.xticks([])
    # plt.scatter(con[i],seq[i],c='r',edgecolors='black')
    plt.scatter(orig,con[i]-seq[i],c='r',edgecolors='black')
    # plt.scatter(sum_con,con[i]-seq[i],c='r',edgecolors='black')
    # plt.scatter(con[i]-seq[i],orig,c='r',edgecolors='black')
    # plt.scatter(orig-seq[i],orig-con[i],c='r',edgecolors='black')
    # plt.plot(orig-seq[i])
    # plt.plot(orig-con[i])
# plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.1, right=0.99,top=0.99, hspace=0.15, wspace=0.15)
plt.savefig(graphs_path+'/scatter of errors of sequential and concurrent decompositions at Huaxian.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/scatter of errors of sequential and concurrent decompositions at Huaxian.tif',format='TIFF',dpi=500)
plt.savefig(graphs_path+'/scatter of errors of sequential and concurrent decompositions at Huaxian.pdf',format='PDF',dpi=1200)
plt.show()