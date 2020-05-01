import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
graphs_path = root_path+'/graphs/'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import statsmodels.api as sm

# hua_vmd = pd.read_csv(root_path+'/Huaxian_vmd/data/VMD_TRAIN.csv')
# hua_vmd_imf1 = hua_vmd['IMF1']
# # hua_vmd_imf1 = hua_vmd_imf1.dropna()
# hua_vmd_imf1 = hua_vmd_imf1.fillna(0)
# lags=20
# plt.figure()
# ax1=plt.subplot(2,1,1)
# sm.graphics.tsa.plot_acf(hua_vmd_imf1, lags=lags, ax=ax1)
# ax2=plt.subplot(2,1,2)
# sm.graphics.tsa.plot_pacf(hua_vmd_imf1, lags=lags, ax=ax2)
# plt.show()

y1=[
    # 0.02,#0
    # 0.051553,#1f
    # 0.100121,#2f
    0.0,#3f
    0.184233,#4
    0.310456,#5f
    0.370174,#6
    0.370174,#7
    0.370174,#8
    0.370174,#9
    0.370174,#10
    0.370174,#11
    0.370174,#12
    0.370174,#13
    0.370174,#14
    0.370174,#15
    0.370174,#16
    0.370174,#17
    0.370174,#18
    0.370174,#19
    0.370174,#20
]

y2=[-v for v in y1]


pacfs = pd.read_csv(root_path+'/Huaxian_vmd/data/PACF.csv')
up_bounds = pacfs['UP']
low_bounds = pacfs['LOW']
pacf = pacfs['IMF1']
print(pacf)
plt.figure(figsize=(3.54,2.5))
lags=list(range(0,pacfs.shape[0]))
t=list(range(-1,pacfs.shape[0]))
z_line=np.zeros(len(t))
# plt.title(r'PACF of $IMF_1}$',loc='left',)
plt.xlim(-1,20)
plt.ylim(-1,1)
x=list(range(3,21))
plt.fill_between(x, y1, y2, where=(y1 > y2), color='C0', alpha=0.3,
                 interpolate=True)
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],)
plt.yticks()
plt.xlabel('Lag (month)')
plt.ylabel('PACF')
plt.bar(lags,pacf,color='b',width=0.8)
plt.plot([-1,21],[up_bounds[0],up_bounds[0]], '--', color='r', label='')
plt.plot([-1,21],[low_bounds[0],low_bounds[0]], '--', color='r', label='')
plt.plot(t,z_line, '-', color='blue', label='',linewidth=0.5)
# plt.subplots_adjust(left=0.09, bottom=0.06, right=0.98,top=0.96, hspace=0.4, wspace=0.3)
plt.tight_layout()
plt.savefig(graphs_path+'Fig.6.PACF of VMD IMF1 at Huaxian.tif',  format='TIFF', dpi=500)
plt.savefig(graphs_path+'Fig.6.PACF of VMD IMF1 at Huaxian.PDF',  format='PDF', dpi=1200)
plt.savefig(graphs_path+'Fig.6.PACF of VMD IMF1 at Huaxian.eps',  format='EPS', dpi=2000)
plt.show()