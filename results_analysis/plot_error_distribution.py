#%%
from matplotlib.pyplot import legend
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from distutils.version import LooseVersion
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from datetime import datetime
plt.rcParams['font.size'] = 6
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
graphs_path = root_path+'/graphs/'
if not os.path.exists(graphs_path):
    os.makedirs(graphs_path)

time = pd.read_csv(root_path+'/time_series/MonthlyRunoffWeiRiver.csv')['Time']
time = time.values
time = [datetime.strptime(t,'%Y/%m') for t in time]
time = [t.strftime('%b %Y') for t in time]

# VMD decompositions
vmd_x0_imf = pd.read_csv(root_path+'/boundary_effect/vmd-decompositions-huaxian/x0_imf.csv')
vmd_x1_imf = pd.read_csv(root_path+'/boundary_effect/vmd-decompositions-huaxian/x1_imf.csv')
vmd_x_1_552_imf = pd.read_csv(root_path+"/boundary_effect/vmd-decompositions-huaxian/x_1_552_imf.csv")
vmd_x_1_791_imf = pd.read_csv(root_path+'/boundary_effect/vmd-decompositions-huaxian/x_1_791_imf.csv')
vmd_x_1_792_imf = pd.read_csv(root_path+'/boundary_effect/vmd-decompositions-huaxian/x_1_792_imf.csv')

vmd_x0_imf1_2_791 = vmd_x0_imf['IMF1'][1:790]
vmd_x0_imf1_2_791 = vmd_x0_imf1_2_791.reset_index(drop=True)
vmd_x1_imf1_1_790 = vmd_x1_imf['IMF1'][0:789]
vmd_x1_imf1_1_790 = vmd_x1_imf1_1_790.reset_index(drop=True)

vmd_err = vmd_x0_imf1_2_791-vmd_x1_imf1_1_790

vmd_x_1_552_imf1 = vmd_x_1_552_imf['IMF1']
vmd_x_1_791_imf1 = vmd_x_1_791_imf['IMF1']
vmd_x_1_792_imf1 = vmd_x_1_792_imf['IMF1']

vmd_err_ap1 = vmd_x_1_792_imf1[0:790]-vmd_x_1_791_imf1[0:790]
vmd_err_aps = vmd_x_1_792_imf1[0:551]-vmd_x_1_552_imf1[0:551]

vmd_train = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_TRAIN.csv")
vmd_full = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_FULL.csv")
vmd_seq_val_dec = pd.DataFrame()
for subsignal in ['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8']:
    test_imf = []
    for i in range(553,792+1):
        data=pd.read_csv(root_path+"/Huaxian_vmd/data/vmd-test/vmd_appended_test"+str(i)+".csv")
        test_imf.append((data[subsignal].iloc[data.shape[0]-1:]).values.flatten()[0])
    val_subsignal = pd.DataFrame(test_imf,columns=[subsignal])
    vmd_seq_val_dec = pd.concat([vmd_seq_val_dec,val_subsignal],axis=1)
vmd_con_val_dec = vmd_full.iloc[vmd_full.shape[0]-240:]
vmd_sc_val_error = vmd_seq_val_dec['IMF1'].values - vmd_con_val_dec['IMF1'].values


# DWT decompositions
dwt_x0_dec = pd.read_csv(root_path+'/boundary_effect/dwt-decompositions-huaxian/x0_dec.csv')
dwt_x1_dec = pd.read_csv(root_path+'/boundary_effect/dwt-decompositions-huaxian/x1_dec.csv')
dwt_x_1_552_dec = pd.read_csv(root_path+"/boundary_effect/dwt-decompositions-huaxian/x_1_552_dec.csv")
dwt_x_1_791_dec = pd.read_csv(root_path+'/boundary_effect/dwt-decompositions-huaxian/x_1_791_dec.csv')
dwt_x_1_792_dec = pd.read_csv(root_path+'/boundary_effect/dwt-decompositions-huaxian/x_1_792_dec.csv')


dwt_x0_D1_2_791 = dwt_x0_dec['D1'][1:790]
dwt_x0_D1_2_791 = dwt_x0_D1_2_791.reset_index(drop=True)
dwt_x1_D1_1_790 = dwt_x1_dec['D1'][0:789]
dwt_x1_D1_1_790 = dwt_x1_D1_1_790.reset_index(drop=True)

dwt_err = dwt_x0_D1_2_791-dwt_x1_D1_1_790

dwt_x_1_552_D1 = dwt_x_1_552_dec['D1']
dwt_x_1_791_D1 = dwt_x_1_791_dec['D1']
dwt_x_1_792_D1 = dwt_x_1_792_dec['D1']

dwt_err_ap1 = dwt_x_1_792_D1[0:790]-dwt_x_1_791_D1[0:790]
dwt_err_aps = dwt_x_1_792_D1[0:551]-dwt_x_1_552_D1[0:551]


dwt_train = pd.read_csv(root_path+"/Huaxian_dwt/data/db10-2/DWT_TRAIN.csv")
dwt_full = pd.read_csv(root_path+"/Huaxian_dwt/data/db10-2/DWT_FULL.csv")
dwt_seq_val_dec = pd.DataFrame()
for subsignal in ['D1','D2','A2',]:
    test_imf = []
    for i in range(553,792+1):
        data=pd.read_csv(root_path+"/Huaxian_dwt/data/db10-2/dwt-test/dwt_appended_test"+str(i)+".csv")
        test_imf.append((data[subsignal].iloc[data.shape[0]-1:]).values.flatten()[0])
    val_subsignal = pd.DataFrame(test_imf,columns=[subsignal])
    dwt_seq_val_dec = pd.concat([dwt_seq_val_dec,val_subsignal],axis=1)
dwt_con_val_dec = dwt_full.iloc[dwt_full.shape[0]-240:]
dwt_sc_val_error = dwt_seq_val_dec['D1'].values - dwt_con_val_dec['D1'].values

# EEMD decompositions
eemd_x0_imf = pd.read_csv(root_path+'/boundary_effect/eemd-decompositions-huaxian/x0_imf.csv')
eemd_x1_imf = pd.read_csv(root_path+'/boundary_effect/eemd-decompositions-huaxian/x1_imf.csv')
eemd_x_1_552_imf = pd.read_csv(root_path+"/boundary_effect/eemd-decompositions-huaxian/x_1_552_imf.csv")
eemd_x_1_791_imf = pd.read_csv(root_path+'/boundary_effect/eemd-decompositions-huaxian/x_1_791_imf.csv')
eemd_x_1_792_imf = pd.read_csv(root_path+'/boundary_effect/eemd-decompositions-huaxian/x_1_792_imf.csv')


eemd_x0_imf1_2_791 = eemd_x0_imf['IMF1'][1:790]
eemd_x0_imf1_2_791 = eemd_x0_imf1_2_791.reset_index(drop=True)
eemd_x1_imf1_1_790 = eemd_x1_imf['IMF1'][0:789]
eemd_x1_imf1_1_790 = eemd_x1_imf1_1_790.reset_index(drop=True)

eemd_err = eemd_x0_imf1_2_791-eemd_x1_imf1_1_790


eemd_x_1_552_imf1 = eemd_x_1_552_imf['IMF1']
eemd_x_1_791_imf1 = eemd_x_1_791_imf['IMF1']
eemd_x_1_792_imf1 = eemd_x_1_792_imf['IMF1']

eemd_err_ap1 = eemd_x_1_792_imf1[0:790]-eemd_x_1_791_imf1[0:790]
eemd_err_aps = eemd_x_1_792_imf1[0:551]-eemd_x_1_552_imf1[0:551]

eemd_train = pd.read_csv(root_path+"/Huaxian_eemd/data/EEMD_TRAIN.csv")
eemd_full = pd.read_csv(root_path+"/Huaxian_eemd/data/EEMD_FULL.csv")
eemd_seq_val_dec = pd.DataFrame()
for subsignal in ['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9']:
    test_imf = []
    for i in range(553,792+1):
        data=pd.read_csv(root_path+"/Huaxian_eemd/data/eemd-test/eemd_appended_test"+str(i)+".csv")
        test_imf.append((data[subsignal].iloc[data.shape[0]-1:]).values.flatten()[0])
    val_subsignal = pd.DataFrame(test_imf,columns=[subsignal])
    eemd_seq_val_dec = pd.concat([eemd_seq_val_dec,val_subsignal],axis=1)
eemd_con_val_dec = eemd_full.iloc[eemd_full.shape[0]-240:]
eemd_sc_val_error = eemd_seq_val_dec['IMF1'].values - eemd_con_val_dec['IMF1'].values

# SSA decompositions
ssa_x0_dec = pd.read_csv(root_path+'/boundary_effect/ssa-decompositions-huaxian/x0_dec.csv')
ssa_x1_dec = pd.read_csv(root_path+'/boundary_effect/ssa-decompositions-huaxian/x1_dec.csv')
ssa_x_1_552_dec = pd.read_csv(root_path+"/boundary_effect/ssa-decompositions-huaxian/x_1_552_dec.csv")
ssa_x_1_791_dec = pd.read_csv(root_path+'/boundary_effect/ssa-decompositions-huaxian/x_1_791_dec.csv')
ssa_x_1_792_dec = pd.read_csv(root_path+'/boundary_effect/ssa-decompositions-huaxian/x_1_792_dec.csv')


ssa_x0_s1_2_791 = ssa_x0_dec['Trend'][1:790]
ssa_x0_s1_2_791 = ssa_x0_s1_2_791.reset_index(drop=True)
ssa_x1_s1_1_790 = ssa_x1_dec['Trend'][0:789]
ssa_x1_s1_1_790 = ssa_x1_s1_1_790.reset_index(drop=True)

ssa_err = ssa_x0_s1_2_791-ssa_x1_s1_1_790

ssa_x_1_552_s1 = ssa_x_1_552_dec['Trend']
ssa_x_1_791_s1 = ssa_x_1_791_dec['Trend']
ssa_x_1_792_s1 = ssa_x_1_792_dec['Trend']

ssa_err_ap1 = ssa_x_1_792_s1[0:790]-ssa_x_1_791_s1[0:790]
ssa_err_aps = ssa_x_1_792_s1[0:551]-ssa_x_1_552_s1[0:551]

ssa_train = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_TRAIN.csv")
ssa_full = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_FULL.csv")
ssa_seq_val_dec = pd.DataFrame()
for subsignal in ['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5','Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise']:
    test_imf = []
    for i in range(553,792+1):
        data=pd.read_csv(root_path+"/Huaxian_ssa/data/ssa-test/ssa_appended_test"+str(i)+".csv")
        test_imf.append((data[subsignal].iloc[data.shape[0]-1:]).values.flatten()[0])
    val_subsignal = pd.DataFrame(test_imf,columns=[subsignal])
    ssa_seq_val_dec = pd.concat([ssa_seq_val_dec,val_subsignal],axis=1)
ssa_con_val_dec = ssa_full.iloc[ssa_full.shape[0]-240:]
ssa_sc_val_error = ssa_seq_val_dec['Trend'].values - ssa_con_val_dec['Trend'].values



#%%

vmd_err_aps = pd.DataFrame(vmd_err_aps.values,
columns=[r'''Calibration error distribution 
for VMD $IMF_{1}$'''])
vmd_sc_val_error = pd.DataFrame(vmd_sc_val_error,
columns=[r'''Validation error distribution 
for VMD $IMF_{1}$'''])
dwt_err_aps = pd.DataFrame(dwt_err_aps.values,
columns=[r'''Calibration error distribution 
for DWT $D_{1}$'''])
dwt_sc_val_error = pd.DataFrame(dwt_sc_val_error,
columns=[r'''Validation error distribution 
for DWT $D_{1}$'''])
eemd_err_aps = pd.DataFrame(eemd_err_aps.values,
columns=[r'''Calibration error distribution 
for EEMD $IMF_{1}$'''])
eemd_sc_val_error = pd.DataFrame(eemd_sc_val_error,
columns=[r'''Validation error distribution 
for EEMD $IMF_{1}$'''])
ssa_err_aps = pd.DataFrame(ssa_err_aps.values,
columns=[r'''Calibration error distribution 
for SSA $S_{1}$'''])
ssa_sc_val_error = pd.DataFrame(ssa_sc_val_error,
columns=[r'''Validation error distribution 
for SSA $S_{1}$'''])

legends = [
    r'''Calibration error distribution 
    for VMD $IMF_{1}$''',
    r'''Validation error distribution 
    for VMD $IMF_{1}$''',
    r'''Calibration error distribution 
    for DWT $D_{1}$''',
    r'''Validation error distribution 
    for DWT $D_{1}$''',
    r'''Calibration error distribution 
    for EEMD $IMF_{1}$''',
    r'''Validation error distribution 
    for EEMD $IMF_{1}$''',
    r'''Calibration error distribution 
    for SSA $S_{1}$''',
    r'''Validation error distribution 
    for SSA $S_{1}$''',
]
import seaborn as sns
data = [vmd_err_aps,vmd_sc_val_error,dwt_err_aps,dwt_sc_val_error,eemd_err_aps,eemd_sc_val_error,ssa_err_aps,ssa_sc_val_error,]

#%%
plt.figure(figsize=(7.48,3.74))
ax1 = plt.subplot(4,2,1)
ax2 = plt.subplot(4,2,2)
ax3 = plt.subplot(4,2,3)
ax4 = plt.subplot(4,2,4)
ax5 = plt.subplot(4,2,5)
ax6 = plt.subplot(4,2,6)
ax7 = plt.subplot(4,2,7)
ax8 = plt.subplot(4,2,8)
ax=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,]
fig_id = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
fig_id_x = [-0.325,-4.95,-0.195,-15,-3.3,-24,-0.21,-6.5]
fig_id_y = [8,0.05,15,0.05,0.2,0.02,8,0.06]
for i in range(len(data)):
    if i in [0,2,4,6]:
        print(i)
        ax[i].set_ylabel('Density')
    if i in [6,7]:
        ax[i].set_xlabel('Error')
    ax[i].text(fig_id_x[i],fig_id_y[i],fig_id[i])
    # ax[i].set_title(fig_id[i],loc='left',pad=-60)
    # data[i].plot(kind='kde',color='purple',linewidth=2,legend=legends[i],ax=ax[i])
    data[i].plot.kde(color='purple',legend=legends[i],linewidth=1,ax=ax[i])
    # data[i].columns = legends[i]
    columns = data[i].columns
    # sns.kdeplot(data[i][columns[0]],cumulative=False,kernel='gau', shade=True,ax=ax[i])
    ax[i].legend(frameon=False,)
plt.tight_layout()
plt.savefig(graphs_path+'Error distribution of Huaxian.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'Error distribution of Huaxian.pdf',format='PDF',dpi=1200)
plt.savefig(graphs_path+'Error distribution of Huaxian.tif',format='TIFF',dpi=500)
plt.show()

# %%
