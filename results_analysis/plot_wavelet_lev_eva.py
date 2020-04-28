import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['image.cmap']='Purples'
# plt.rcParams['axes.linewidth']=0.8
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)

wavelet=[
    'coif1','coif2','db1','db5','db10','fk4','haar','sym4',
]
level = [1,2,3,4]
train_nse_dict={}
dev_nse_dict={}
nse_dict = {}
for wave in wavelet:
    train_nse = []
    dev_nse = []
    nse=[]
    for lev in level:
        data = pd.read_csv(root_path+'/Huaxian_modwt/projects/esvr-wddff/'+wave+'-'+str(lev)+'/single_hybrid_1_ahead_mi_ts0.1/optimal_model_results.csv')
        train_nse.append(data['train_nse'][0])
        dev_nse.append(data['dev_nse'][0])
        nse.append(data['train_nse'][0])
        nse.append(data['dev_nse'][0])
    train_nse_dict[wave] = train_nse
    dev_nse_dict[wave] = dev_nse
    nse_dict[wave]=nse
y=['1T','1D','2T','2D','3T','3D','4T','4D']
train_nse_df = pd.DataFrame(train_nse_dict,index=level)
dev_nse_df = pd.DataFrame(dev_nse_dict,index=level)
nse_df = pd.DataFrame(nse_dict,index=y)
print(train_nse_df)
print(dev_nse_df)
print(nse_df)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.48,3.2))
max_v = round(max(nse_df.max(axis=1)),1)
min_v = round(min(nse_df.min(axis=1)),1)
interval=round((max_v-min_v)/4,1)

im = ax.imshow(nse_df, extent=[0,len(wavelet),0,len(y)],vmin=0,vmax=1,interpolation='none',aspect='equal')  

# im = ax.imshow(nse_df, extent=[0,len(wavelet),0,len(y)],vmin=min_v,vmax=max_v,interpolation='none',aspect='equal')  
ax.set_xticks(np.arange(0.5,len(wavelet)+0.5,1))
ax.set_yticks(np.arange(0.5,len(y)+0.5,1))
ax.set_xticklabels(wavelet,rotation=45)
ax.set_yticklabels(y[::-1])
cb_ax = fig.add_axes([0.84, 0.148, 0.05, 0.807])#[left,bottom,width,height]
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_label(r'$NSE$')
cbar.set_ticks(np.arange(0, 1.1, 0.2))
# cbar.set_ticks(np.arange(min_v,max_v+0.1, interval))
fig.subplots_adjust(bottom=0.11, top=0.99, left=0.08, right=0.82,wspace=0.3, hspace=0.1)
plt.savefig(root_path+'/graphs/Fig.8 NSE of MODWT-SVR with different wavelets and levels.tif',format='TIFF',dpi=1200)
plt.savefig(root_path+'/graphs/Fig.8 NSE of MODWT-SVR with different wavelets and levels.pdf',format='PDF',dpi=1200)
plt.show()



