import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [7.48, 5.61]
plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from config.globalLog import logger
graphs_path = root_path+'/graphs/'


vmd_train = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_TRAIN.csv")
eemd_train = pd.read_csv(root_path+"/Huaxian_eemd/data/EEMD_TRAIN.csv")
ssa_train = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_TRAIN.csv")
dwt_train = pd.read_csv(root_path+"/Huaxian_dwt/data/db10-2/DWT_TRAIN.csv")
modwt_train = pd.read_csv(root_path+"/Huaxian_modwt/data-wddff/db1-4/1_ahead_calibration_X12dec.csv")

vmd_train=vmd_train.drop("ORIG",axis=1)
eemd_train=eemd_train.drop("ORIG",axis=1)
ssa_train=ssa_train.drop("ORIG",axis=1)
dwt_train=dwt_train.drop("ORIG",axis=1)

vmd_corrs = vmd_train.corr(method="pearson")
eemd_corrs = eemd_train.corr(method="pearson")
ssa_corrs = ssa_train.corr(method="pearson")
dwt_corrs = dwt_train.corr(method="pearson")
modwt_corrs = modwt_train.corr(method="pearson")


corrs=[abs(eemd_corrs),abs(ssa_corrs),abs(vmd_corrs),abs(dwt_corrs),abs(modwt_corrs)]
titles=["EEMD","SSA","VMD","DWT(db10-2)",'BCMODWT(db1-4)']
# plt.figure(figsize=(3.54,3.4))
# for i in range(len(corrs)):
#     plt.subplot(2,2,i+1)
#     plt.title(titles[i],fontsize=6)
#     sign_num=corrs[i].shape[1]
#     ticks = list(range(sign_num))
#     labels=[]
#     for j in ticks:
#         if titles[i].find('VMD')>=0:
#             labels.append(r'$IMF_{'+str(j+1)+'}$')
#         elif titles[i].find('EEMD')>=0:
#             if j==sign_num-1:
#                 labels.append(r'$R$')
#             else:
#                 labels.append(r'$IMF_{'+str(j+1)+'}$')
#         elif titles[i].find('DWT')>=0 or titles[i].find('MODWT')>=0:
#             if j==sign_num-1:
#                 labels.append(r'$A_{'+str(j)+'}$')
#             else:
#                 labels.append(r'$D_{'+str(j+1)+'}$')
#     ax1=plt.imshow(corrs[i])
#     plt.xticks(ticks=ticks,labels=labels,rotation=90)
#     plt.yticks(ticks=ticks,labels=labels)
#     plt.xlim(-0.5,sign_num-0.5)
#     plt.ylim(-0.5,sign_num-0.5)
#     # plt.xlabel(r"${S}_i$")
#     # plt.ylabel(r"${S}_j$")
#     plt.colorbar(ax1.colorbar, fraction=0.045)
#     ax1.colorbar.set_label("$Corr_{i,j}$")
#     plt.clim(0,1)
# plt.tight_layout()
# plt.show()
series_len=[9,12,8,3]
# fig = plt.figure(figsize=(7.48,5.0))
# ax1 = plt.subplot2grid((2,6), (0,0), colspan=2,aspect='equal')
# ax2 = plt.subplot2grid((2,6), (0,2), colspan=2,aspect='equal')
# ax3 = plt.subplot2grid((2,6), (0,4), colspan=2,aspect='equal')
# ax4 = plt.subplot2grid((2,6), (1,1), colspan=2,aspect='equal')
# ax5 = plt.subplot2grid((2,6), (1,3), colspan=2,aspect='equal')
# axs = [ax1,ax2,ax3,ax4,ax5]
# for i in range(len(corrs)):
#     ax = axs [i]
#     ax.set_title(titles[i],fontsize=6)
#     sign_num=corrs[i].shape[1]
#     logger.info('Number of sub-signals:{}'.format(sign_num))
#     ticks = list(range(sign_num))
#     logger.info('ticks:{}'.format(ticks))
#     labels=[]
#     for j in ticks:
#         if titles[i].find('VMD')>=0:
#             labels.append(r'$IMF_{'+str(j+1)+'}$')
#         elif titles[i].find('EEMD')>=0:
#             if j==sign_num-1:
#                 labels.append(r'$R$')
#             else:
#                 labels.append(r'$IMF_{'+str(j+1)+'}$')
#         elif titles[i].find('MODWT')>=0:
#             if j==sign_num-1:
#                 labels.append(r'$V_{'+str(j)+'}$')
#             else:
#                 labels.append(r'$W_{'+str(j+1)+'}$')
#         elif titles[i].find('DWT')>=0:
#             if j==sign_num-1:
#                 labels.append(r'$A_{'+str(j)+'}$')
#             else:
#                 labels.append(r'$D_{'+str(j+1)+'}$')
        
#         elif titles[i].find('SSA')>=0:
#             labels.append(r'$S_{'+str(j+1)+'}$')
#     logger.info('Labels:{}'.format(labels))
#     im = ax.imshow(corrs[i],vmin=0, vmax=1)
#     # plt.xticks(ticks=ticks,labels=labels,rotation=45)
#     # plt.yticks(ticks=ticks,labels=labels)
#     # plt.xlim(-0.5,sign_num-0.5)
#     # plt.ylim(-0.5,sign_num-0.5)
#     ax.set_xticks(ticks=ticks)
#     ax.set_xticklabels(labels=labels,rotation=45)
#     ax.set_yticks(ticks=ticks)
#     ax.set_yticklabels(labels=labels)
#     ax.set_xlim(-0.5,sign_num-0.5)
#     ax.set_ylim(-0.5,sign_num-0.5)

# fig.subplots_adjust(bottom=0.06, top=0.95, left=0.04, right=0.99,wspace=0.3, hspace=0.35)
# # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

# cb_ax = fig.add_axes([0.85, 0.06, 0.05, 0.38])#[x,y,width,height]
# cbar = fig.colorbar(im, cax=cb_ax)
# cbar.set_ticks(np.arange(0, 1.1, 0.5))
# cbar.set_label(r"$Corr_{i,j}$")
# # cbar.set_ticklabels(['low', 'medium', 'high'])
# plt.savefig(graphs_path+"Fig.9.Pearson corr of Huaxian.tif",format="TIFF",dpi=1200)
# plt.savefig(graphs_path+"Fig.9.Pearson corr of Huaxian.pdf",format="PDF",dpi=1200)
# plt.show()

fig = plt.figure(figsize=(7.4861,1.7))
for i in range(len(corrs)):
    ax = plt.subplot(1,5,i+1)
    ax.set_title(titles[i],fontsize=6)
    sign_num=corrs[i].shape[1]
    logger.info('Number of sub-signals:{}'.format(sign_num))
    ticks = list(range(sign_num))
    logger.info('ticks:{}'.format(ticks))
    labels=[]
    for j in ticks:
        if titles[i].find('VMD')>=0:
            labels.append(r'$IMF_{'+str(j+1)+'}$')
        elif titles[i].find('EEMD')>=0:
            if j==sign_num-1:
                labels.append(r'$R$')
            else:
                labels.append(r'$IMF_{'+str(j+1)+'}$')
        elif titles[i].find('MODWT')>=0:
            if j==sign_num-1:
                labels.append(r'$V_{'+str(j)+'}$')
            else:
                labels.append(r'$W_{'+str(j+1)+'}$')
        elif titles[i].find('DWT')>=0:
            if j==sign_num-1:
                labels.append(r'$A_{'+str(j)+'}$')
            else:
                labels.append(r'$D_{'+str(j+1)+'}$')
        elif titles[i].find('SSA')>=0:
            labels.append(r'$S_{'+str(j+1)+'}$')
    logger.info('Labels:{}'.format(labels))
    im = ax.imshow(corrs[i],vmin=0, vmax=1)
    # plt.xticks(ticks=ticks,labels=labels,rotation=45)
    # plt.yticks(ticks=ticks,labels=labels)
    # plt.xlim(-0.5,sign_num-0.5)
    # plt.ylim(-0.5,sign_num-0.5)
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels=labels,rotation=90)
    ax.set_yticks(ticks=ticks)
    ax.set_yticklabels(labels=labels)
    ax.set_xlim(-0.5,sign_num-0.5)
    ax.set_ylim(-0.5,sign_num-0.5)

fig.subplots_adjust(bottom=0.09, top=0.97, left=0.04, right=0.92,wspace=0.3, hspace=0.35)
# add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

cb_ax = fig.add_axes([0.93, 0.22, 0.02, 0.625])#[x,y,width,height]
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_ticks(np.arange(0, 1.1, 0.5))
cbar.set_label(r"Correlation")
# cbar.set_ticklabels(['low', 'medium', 'high'])
plt.savefig(graphs_path+"Fig.9.Pearson corr of Huaxian.tif",format="TIFF",dpi=1200)
plt.savefig(graphs_path+"Fig.9.Pearson corr of Huaxian.pdf",format="PDF",dpi=1200)
plt.show()





