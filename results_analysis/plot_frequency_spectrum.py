import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [7.48, 5.61]
# plt.rcParams['image.cmap']='plasma'
plt.rcParams['axes.linewidth']=0.8
import math
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/graphs/'

huaxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/data/VMD_TRAIN.csv')
huaxian_dwt = pd.read_csv(root_path+"/Huaxian_dwt/data/db10-2/DWT_TRAIN.csv")
huaxian_eemd = pd.read_csv(root_path+"/Huaxian_eemd/data/EEMD_TRAIN.csv")
huaxian_ssa = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_TRAIN.csv")
huaxian_modwt = pd.read_csv(root_path+"/Huaxian_modwt/data-wddff/db1-4/1_ahead_calibration_X12dec.csv")



huaxian_vmd=huaxian_vmd.drop('ORIG',axis=1)
plt.figure(figsize=(3.54,2.8))
y=[2300,950,170,330,140,130,90,140]
h_v_cols=huaxian_vmd.columns.values
titles=[r'$IMF_1$',r'$IMF_2$',r'$IMF_3$',r'$IMF_4$'
,r'$IMF_5$',r'$IMF_6$',r'$IMF_7$',r'$IMF_8$']
for i in range(len(h_v_cols)):
    subsignal=huaxian_vmd[h_v_cols[i]]
    T=subsignal.shape[0]
    t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
    freqs = t-0.5-1/T
    plt.subplot(len(h_v_cols)/2,2,i+1)
    plt.xlim(-0.55,0.65)
    plt.text(0.45,y[i],titles[i])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.plot(freqs,abs(fft(subsignal.values)),c='b',lw=0.8)
    if i==len(h_v_cols)-1 or i==len(h_v_cols)-2:
        plt.xlabel('Frequency (1/month)')
    else:
        plt.xticks([])
    # if i in [0,2,4,6]:
    plt.ylabel('Amplitude')
# plt.subplots_adjust(left=0.08, bottom=0.14, right=0.99,top=0.99, hspace=0.05, wspace=0.2)
plt.subplots_adjust(left=0.12, bottom=0.14, right=0.99,top=0.95, hspace=0.35, wspace=0.35)
plt.savefig(graphs_path+"Frequency spectrum of VMD at Huaxian.eps",format="EPS",dpi=2000)
plt.savefig(graphs_path+"Frequency spectrum of VMD at Huaxian.tif",format="TIFF",dpi=500)
plt.savefig(graphs_path+"Frequency spectrum of VMD at Huaxian.pdf",format="PDF",dpi=1200)
# plt.show()

huaxian_eemd=huaxian_eemd.drop('ORIG',axis=1)
plt.figure(figsize=(3.54,3.1))
ax1 = plt.subplot2grid((5,2), (0,0), colspan=1,)
ax2 = plt.subplot2grid((5,2), (0,1), colspan=1,)
ax3 = plt.subplot2grid((5,2), (1,0), colspan=1,)
ax4 = plt.subplot2grid((5,2), (1,1), colspan=1,)
ax5 = plt.subplot2grid((5,2), (2,0), colspan=1,)
ax6 = plt.subplot2grid((5,2), (2,1), colspan=1,)
ax7 = plt.subplot2grid((5,2), (3,0), colspan=1,)
ax8 = plt.subplot2grid((5,2), (3,1), colspan=1,)
ax9 = plt.subplot2grid((5,2), (4,0), colspan=2,)
ax_list = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
y=[240,520,450,230,150,160,310,19,2500]
h_e_cols=huaxian_eemd.columns.values
titles=[r'$IMF_1$',r'$IMF_2$',r'$IMF_3$',r'$IMF_4$'
,r'$IMF_5$',r'$IMF_6$',r'$IMF_7$',r'$IMF_8$',r'$R$']
for i in range(len(h_e_cols)):
    subsignal=huaxian_eemd[h_e_cols[i]]
    T=subsignal.shape[0]
    t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
    freqs = t-0.5-1/T
    # plt.subplot(len(h_e_cols),1,i+1)
    ax = ax_list[i]
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.plot(freqs,abs(fft(subsignal.values)),c='b',lw=0.8)
    if i==len(h_e_cols)-1:
        ax.text(0.49,y[i],titles[i])
        ax.set_xlabel('Frequency (1/month)')
    else:
        ax.set_xlim(-0.55,0.75)
        ax.text(0.55,y[i],titles[i])
        ax.set_xticks([])
    ax.set_ylabel('Amplitude')
plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98,top=0.95, hspace=0.35, wspace=0.35)
plt.savefig(graphs_path+"Frequency spectrum of EEMD at Huaxian.eps",format="EPS",dpi=2000)
plt.savefig(graphs_path+"Frequency spectrum of EEMD at Huaxian.tif",format="TIFF",dpi=500)
plt.savefig(graphs_path+"Frequency spectrum of EEMD at Huaxian.pdf",format="PDF",dpi=1200)
# plt.show()

huaxian_ssa=huaxian_ssa.drop('ORIG',axis=1)
# plt.figure(figsize=(3.54,7.4))
plt.figure(figsize=(3.54,3.8))
y=[2400,460,450,155,150,75,70,45,110,65,40,40]
h_s_cols=huaxian_ssa.columns.values
titles=[r'$S_1$',r'$S_2$',r'$S_3$',r'$S_4$'
,r'$S_5$',r'$S_6$',r'$S_7$',r'$S_8$',r'$S_9$',r'$S_{10}$',r'$S_{11}$',r'$S_{12}$']
for i in range(len(h_s_cols)):
    subsignal=huaxian_ssa[h_s_cols[i]]
    T=subsignal.shape[0]
    t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
    freqs = t-0.5-1/T
    plt.subplot(len(h_s_cols)/2,2,i+1)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.text(0.45,y[i],titles[i])
    plt.xlim(-0.55,0.6)
    plt.plot(freqs,abs(fft(subsignal.values)),c='b',lw=0.8)
    if i==len(h_s_cols)-1 or i==len(h_s_cols)-2:
        plt.xlabel('Frequency (1/month)')
    else:
        plt.xticks([])
    plt.ylabel('Amplitude')
# plt.subplots_adjust(left=0.08, bottom=0.1, right=0.99,top=0.99, hspace=0.05, wspace=0.2)
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.99,top=0.95, hspace=0.35, wspace=0.35)
plt.savefig(graphs_path+"Frequency spectrum of SSA at Huaxian.eps",format="EPS",dpi=2000)
plt.savefig(graphs_path+"Frequency spectrum of SSA at Huaxian.tif",format="TIFF",dpi=500)
plt.savefig(graphs_path+"Frequency spectrum of SSA at Huaxian.pdf",format="PDF",dpi=1200)
# plt.show()

huaxian_dwt=huaxian_dwt.drop('ORIG',axis=1)
plt.figure(figsize=(3.54,1.6))
ax1 = plt.subplot2grid((2,2), (0,0), colspan=1,)
ax2 = plt.subplot2grid((2,2), (0,1), colspan=1,)
ax3 = plt.subplot2grid((2,2), (1,0), colspan=2,)
ax_list=[ax1,ax2,ax3]
y=[200,320,2500,]
h_d_cols=huaxian_dwt.columns.values
titles=[r'$D_1$',r'$D_2$',r'$A_2$',]
for i in range(len(h_d_cols)):
    subsignal=huaxian_dwt[h_d_cols[i]]
    T=subsignal.shape[0]
    t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
    freqs = t-0.5-1/T
    # plt.subplot(len(h_d_cols),1,i+1)
    ax=ax_list[i]
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # ax.axvline(0,c='black',lw=0.5,linestyle='--')
    ax.plot(freqs,abs(fft(subsignal.values)),c='b',lw=0.8)
    if i==len(h_d_cols)-1:
        ax.set_xlim(-0.55,0.55)
        ax.text(0.49,y[i],titles[i])
        ax.set_xlabel('Frequency (1/month)')
    else:
        ax.set_xlim(-0.55,0.6)
        ax.text(0.45,y[i],titles[i])
        ax.set_xticks([])
    ax.set_ylabel('Amplitude')
plt.subplots_adjust(left=0.12, bottom=0.25, right=0.98,top=0.9, hspace=0.35, wspace=0.35)
plt.savefig(graphs_path+"Frequency spectrum of DWT at Huaxian.eps",format="EPS",dpi=2000)
plt.savefig(graphs_path+"Frequency spectrum of DWT at Huaxian.tif",format="TIFF",dpi=500)
plt.savefig(graphs_path+"Frequency spectrum of DWT at Huaxian.pdf",format="PDF",dpi=1200)
# plt.show()

plt.figure(figsize=(3.54,2.2))
ax1 = plt.subplot2grid((3,2), (0,0), colspan=1,)
ax2 = plt.subplot2grid((3,2), (0,1), colspan=1,)
ax3 = plt.subplot2grid((3,2), (1,0), colspan=1,)
ax4 = plt.subplot2grid((3,2), (1,1), colspan=1,)
ax5 = plt.subplot2grid((3,2), (2,0), colspan=2,)
ax_list=[ax1,ax2,ax3,ax4,ax5]
y=[200,380,601,300,2400]
h_m_cols=huaxian_modwt.columns.values
titles=[r'$W_1$',r'$W_2$',r'$W_3$',r'$W_4$',r'$V_4$']
for i in range(len(h_m_cols)):
    subsignal=huaxian_modwt[h_m_cols[i]]
    T=subsignal.shape[0]
    t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
    freqs = t-0.5-1/T
    # plt.subplot(len(h_m_cols),1,i+1)
    ax=ax_list[i]
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.plot(freqs,abs(fft(subsignal.values)),c='b',lw=0.8)
    if i==len(h_m_cols)-1:
        ax.set_xlim(-0.55,0.55)
        ax.text(0.49,y[i],titles[i])
        ax.set_xlabel('Frequency(1/month)')
    else:
        ax.set_xlim(-0.55,0.6)
        ax.text(0.45,y[i],titles[i])
        ax.set_xticks([])
    ax.set_ylabel('Amplitude')
# plt.subplots_adjust(left=0.14, bottom=0.13, right=0.98,top=0.99, hspace=0.05, wspace=0.05)
plt.subplots_adjust(left=0.1, bottom=0.18, right=0.99,top=0.92, hspace=0.35, wspace=0.35)
plt.savefig(graphs_path+"Frequency spectrum of BCMODWT at Huaxian.eps",format="EPS",dpi=2000)
plt.savefig(graphs_path+"Frequency spectrum of BCMODWT at Huaxian.tif",format="TIFF",dpi=500)
plt.savefig(graphs_path+"Frequency spectrum of BCMODWT at Huaxian.pdf",format="PDF",dpi=1200)
# plt.show()

plt.figure(figsize=(7.4861,5))
ax1  = plt.subplot2grid((5,360), (0,0), colspan=30,)#ssa
ax2  = plt.subplot2grid((5,360), (0,30), colspan=30,)
ax3  = plt.subplot2grid((5,360), (0,60), colspan=30,)
ax4  = plt.subplot2grid((5,360), (0,90), colspan=30,)
ax5  = plt.subplot2grid((5,360), (0,120), colspan=30,)
ax6  = plt.subplot2grid((5,360), (0,150), colspan=30,)
ax7  = plt.subplot2grid((5,360), (0,180), colspan=30,)
ax8  = plt.subplot2grid((5,360), (0,210), colspan=30,)
ax9  = plt.subplot2grid((5,360), (0,240), colspan=30,)
ax10 = plt.subplot2grid((5,360), (0,270), colspan=30,)
ax11 = plt.subplot2grid((5,360), (0,300), colspan=30,)
ax12 = plt.subplot2grid((5,360), (0,330), colspan=30,)
ax13 = plt.subplot2grid((5,360), (1,0), colspan=40,)#eemd
ax14 = plt.subplot2grid((5,360), (1,40), colspan=40,)
ax15 = plt.subplot2grid((5,360), (1,80), colspan=40,)
ax16 = plt.subplot2grid((5,360), (1,120), colspan=40,)
ax17 = plt.subplot2grid((5,360), (1,160), colspan=40,)
ax18 = plt.subplot2grid((5,360), (1,200), colspan=40,)
ax19 = plt.subplot2grid((5,360), (1,240), colspan=40,)
ax20 = plt.subplot2grid((5,360), (1,280), colspan=40,)
ax21 = plt.subplot2grid((5,360), (1,320), colspan=40,)
ax22 = plt.subplot2grid((5,360), (2,0), colspan=45,)#vmd
ax23 = plt.subplot2grid((5,360), (2,45), colspan=45,)
ax24 = plt.subplot2grid((5,360), (2,90), colspan=45,)
ax25 = plt.subplot2grid((5,360), (2,135), colspan=45,)
ax26 = plt.subplot2grid((5,360), (2,180), colspan=45,)
ax27 = plt.subplot2grid((5,360), (2,225), colspan=45,)
ax28 = plt.subplot2grid((5,360), (2,270), colspan=45,)
ax29 = plt.subplot2grid((5,360), (2,315), colspan=45,)
ax30 = plt.subplot2grid((5,360), (3,0), colspan=72,)#modwt
ax31 = plt.subplot2grid((5,360), (3,72), colspan=72,)
ax32 = plt.subplot2grid((5,360), (3,144), colspan=72,)
ax33 = plt.subplot2grid((5,360), (3,216), colspan=72,)
ax34 = plt.subplot2grid((5,360), (3,288), colspan=72,)
ax35 = plt.subplot2grid((5,360), (4,0), colspan=120,)#DWT
ax36 = plt.subplot2grid((5,360), (4,120), colspan=120,)
ax37 = plt.subplot2grid((5,360), (4,240), colspan=120,)
axs=[
    [ax1 ,ax2 ,ax3 ,ax4 ,ax5 ,ax6 ,ax7 ,ax8 ,ax9 ,ax10,ax11,ax12,],
    [ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20,ax21,],
    [ax22,ax23,ax24,ax25,ax26,ax27,ax28,ax29,],
    [ax30,ax31,ax32,ax33,ax34,],
    [ax35,ax36,ax37,],
]
signal_labels=[
    [r'$SSA\ S_1$',r'$SSA\ S_2$',r'$SSA\ S_3$',r'$SSA\ S_4$',r'$SSA\ S_5$',r'$SSA\ S_6$',r'$SSA\ S_7$',r'$SSA\ S_8$',r'$SSA\ S_{9}$',r'$SSA\ S_{10}$',r'$SSA\ S_{11}$',r'$SSA\ S_{12}$'],
    [r'$EEMD\ IMF_1$',r'$EEMD\ IMF_2$',r'$EEMD\ IMF_3$',r'$EEMD\ IMF_4$',r'$EEMD\ IMF_5$',r'$EEMD\ IMF_6$',r'$EEMD\ IMF_7$',r'$EEMD\ IMF_8$',r'$EEMD\ R$'],
    [r'$VMD\ IMF_1$',r'$VMD\ IMF_2$',r'$VMD\ IMF_3$',r'$VMD\ IMF_4$',r'$VMD\ IMF_5$',r'$VMD\ IMF_6$',r'$VMD\ IMF_7$',r'$VMD\ IMF_8$'],
    [r'$BCMODWT\ W_1$',r'$BCMODWT\ W_2$',r'$BCMODWT\ W_3$',r'$BCMODWT\ W_4$',r'$BCMODWT\ V_4$'],
    [r'$DWT\ D_1$',r'$DWT\ D_2$',r'$DWT\ A_2$',],
]
dec=[r'SSA',r'EEMD',r'VMD',r'MODWT',r'DWT']
colors=['b','g','r','c','m']
signals=[huaxian_ssa,huaxian_eemd,huaxian_vmd,huaxian_modwt,huaxian_dwt]
for i in range(len(axs)):
    subsignals=signals[i]
    cols=subsignals.columns.values
    ax_list=axs[i]
    for j in range(len(ax_list)):
        ax=ax_list[j]
        ax.set_title(signal_labels[i][j],pad=3)
        subsignal=subsignals[cols[j]]
        T=subsignal.shape[0]
        t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
        freqs = t-0.5-1/T
        ax.set_xticks([-0.4,0,0.4])
        # ax.set_xticklabels([-0.4,0,0.4],rotation=45)
        ax.set_yticks([])
        if j==0:
            ax.set_ylabel('Amplitude')
        if i==len(axs)-1:
            ax.set_xlabel('Frequency (1/month)')
        # else:
        #     ax.set_xticks([])
        ax.plot(freqs,abs(fft(subsignal.values)),c=colors[i],lw=0.8,label=signal_labels[i][j])
plt.subplots_adjust(left=0.026, bottom=0.07, right=0.99,top=0.96, hspace=0.75, wspace=0.05)
plt.savefig(graphs_path+"Fig.10.Frequency spectrum of Huaxian.eps",format="EPS",dpi=2000)
plt.savefig(graphs_path+"Fig.10.Frequency spectrum of Huaxian.tif",format="TIFF",dpi=500)
plt.savefig(graphs_path+"Fig.10.Frequency spectrum of Huaxian.pdf",format="PDF",dpi=1200)
plt.show()

# most_diff_signals=[
#     huaxian_eemd['IMF1'],
#     huaxian_ssa['Periodic5'],
#     huaxian_vmd['IMF8'],
#     huaxian_dwt['D1'],
#     huaxian_modwt['W1'],
# ]

# plt.figure(figsize=(3.54,5.54))
# y=[320,98,188,235,320]
# fig_idx=[
#     r'(a) $IMF_1$ of EEMD',
#     r'(b) $P_5$ of SSA',
#     r'(c) $IMF_8$ of VMD',
#     r'(d) $D_1$ of DWT',
#     r'(e) $W_1$ of BCMODWT',
# ]
# for i in range(len(most_diff_signals)):
#     signal = most_diff_signals[i]
#     signal = signal.values
#     print(signal)
#     T=signal.shape[0]
#     t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
#     freqs = t-0.5-1/T
#     plt.subplot(5,1,i+1)
#     plt.text(-0.52,y[i],fig_idx[i],fontsize=7)
#     if i==len(most_diff_signals)-1:
#         plt.xlabel('Frequence(1/month)')
#     if i<len(most_diff_signals)-1:
#         plt.xticks([])
#     if i==0:
#         plt.ylim(-20,380)
#     elif i==len(most_diff_signals)-1:
#         plt.ylim(-20,380)
#     plt.ylabel('Amplitude')
#     freq_am = abs(fft(signal))
#     plt.plot(freqs,freq_am,c='b',lw=0.8)
# plt.subplots_adjust(left=0.14, bottom=0.08, right=0.98,top=0.99, hspace=0.05, wspace=0.05)
# plt.savefig(graphs_path+"Frequency spectrum of decompositions at Huaxian.eps",format="EPS",dpi=2000)
# plt.savefig(graphs_path+"Frequency spectrum of decompositions at Huaxian.tif",format="TIFF",dpi=1200)
# plt.savefig(graphs_path+"Frequency spectrum of decompositions at Huaxian.pdf",format="PDF",dpi=1200)
# plt.show()







