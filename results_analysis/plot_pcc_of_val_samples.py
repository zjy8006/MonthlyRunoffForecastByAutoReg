import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)

graphs_path = root_path+'/graphs/'

# dev = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_hindcast_pacf/dev_samples.csv')
# test = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_hindcast_pacf/test_samples.csv')
# val = pd.concat([dev,test],axis=0)
# ap_val = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_forecast_pacf/dev_test_samples.csv')
# v_val = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_forecast_pacf_train_val/dev_test_samples.csv')

# train=train[8:]
# print(train.shape[0])

# std_train=train.std(axis=1)
# std_ap_dev_test=ap_dev_test.std(axis=1)
# std_v_dev_test=v_dev_test.std(axis=1)


# month_train_std=pd.DataFrame()
# n_train_yr=int(std_train.shape[0]/12)
# for i in range(n_train_yr):
#     # i=0,0:12
#     # i=1,12:24
#     std_i = std_train[i*12:(i+1)*12]
#     std_i = std_i.reset_index(drop=True)
#     month_train_std = pd.concat([month_train_std,std_i],axis=1)

# month_train_std=month_train_std.sum(axis=1)/n_train_yr
# print(month_train_std)

# month_ap_std=pd.DataFrame()
# n_val_yr=int(std_ap_dev_test.shape[0]/12)
# for i in range(n_val_yr):
#     # i=0,0:12
#     # i=1,12:24
#     std_i = std_ap_dev_test[i*12:(i+1)*12]
#     std_i = std_i.reset_index(drop=True)
#     month_ap_std = pd.concat([month_ap_std,std_i],axis=1)

# month_ap_std=month_ap_std.sum(axis=1)/n_val_yr

# month_v_std=pd.DataFrame()
# n_val_yr=int(std_v_dev_test.shape[0]/12)
# for i in range(n_val_yr):
#     # i=0,0:12
#     # i=1,12:24
#     std_i = std_v_dev_test[i*12:(i+1)*12]
#     std_i = std_i.reset_index(drop=True)
#     month_v_std = pd.concat([month_v_std,std_i],axis=1)

# month_v_std=month_v_std.sum(axis=1)/n_val_yr


# month_train_std.plot(label='train')
# month_ap_std.plot(label='append')
# month_v_std.plot(label='val')
# plt.legend()
# plt.show()


dev = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_hindcast_pacf/dev_samples.csv')
test = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_hindcast_pacf/test_samples.csv')
val = pd.concat([dev,test],axis=0)
val=val.reset_index(drop=True)
ap_val = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_forecast_pacf/dev_test_samples.csv')
v_dev = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_forecast_pacf_train_val/dev_samples.csv')
v_test = pd.read_csv(root_path+'/Huaxian_vmd/data/one_step_1_ahead_forecast_pacf_train_val/test_samples.csv')
v_val=pd.concat([v_dev,v_test],axis=0)
v_val = v_val.reset_index(drop=True)
print(val.shape[0])
print(ap_val.shape[0])
print(v_val.shape[0])


val_cor=abs(val.corr())
val_x_cor = val_cor['Y']
val_x_cor = val_x_cor[:val_x_cor.shape[0]-1]
print(val_x_cor)

ap_val_cor=abs(ap_val.corr())
ap_val_x_cor = ap_val_cor['Y']
ap_val_x_cor = ap_val_x_cor[:ap_val_x_cor.shape[0]-1]

v_val_cor=abs(v_val.corr())
v_val_x_cor = v_val_cor['Y']
v_val_x_cor = v_val_x_cor[:v_val_x_cor.shape[0]-1]


metrics_lists = [ap_val_x_cor,v_val_x_cor]
samples = ['validation samples generated from appended decompositions', 'validation samples generated from validation decompositions',]
pos=[]
for i in range(ap_val_x_cor.shape[0]):
    pos.append(2*(i+1))
print(pos)
width = 0.8
action = [ -1, 0]
labels = list(ap_val_x_cor.index.values)
print(labels)
colors=['b','r']
fig = plt.figure(figsize=(7.48, 2))
ax=plt.subplot(1,1,1)
for j in range(len(metrics_lists)):
    x=[p+action[j]*width for p in pos]
    print(x)
    bars = ax.bar([p+action[j]*width for p in pos],metrics_lists[j], width,color=colors[j], alpha=0.75, label=samples[j])
    # for bar in bars:
    #     bar.set_hatch(hatch_str[j])
    # autolabels(bars,ax)
# ax.set_ylim(ylims[i])
ax.set_xlabel('Predictors')
ax.set_ylabel('Pearson correlation coefficients')
pos=[p-width/2 for p in pos]
ax.set_xlim(0,61)
ax.set_xticks(pos)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
plt.subplots_adjust(left=0.06, bottom=0.225, right=0.99,top=0.99, hspace=0.5, wspace=0.25)
plt.savefig(graphs_path+'Fig.3 PCC of appended samples and validation samples.eps',format='EPS', dpi=2000)
plt.savefig(graphs_path+'Fig.3 PCC of appended samples and validation samples.tif',format='TIFF', dpi=500)
plt.savefig(graphs_path+'Fig.3 PCC of appended samples and validation samples.pdf',format='PDF', dpi=1200)
plt.show()