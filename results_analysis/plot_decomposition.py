import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
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
huaxian_modwt = pd.read_csv(root_path+"/Huaxian_modwt/data/db10-2/MODWT_TRAIN.csv")
huaxian_eemd = pd.read_csv(root_path+"/Huaxian_eemd/data/EEMD_TRAIN.csv")
huaxian_ssa = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_TRAIN.csv")
T=huaxian_vmd.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T

T=huaxian_vmd.shape[0] #sampling frequency
fs=1/T #sampling period(interval)
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T #sampling times
freqs = t-0.5-1/T
L = huaxian_vmd.shape[1]-1
plt.figure(figsize=(7.48,7.48))
for i in range(1,L+1):
    plt.subplot(L,2,2*i-1)
    if i==L:
        plt.xlabel('Time(month)')
    plt.ylabel('IMF'+str(i))
    plt.plot(huaxian_vmd['IMF'+str(i)],color='b',label='',linewidth=0.8)

    plt.subplot(L,2,2*i)
    plt.plot(freqs,abs(fft(huaxian_vmd['IMF'+str(i)])),c='b',lw=0.8)
    if i==L:
        plt.xlabel('Frequence(1/month)')
    plt.ylabel('Amplitude')
plt.tight_layout()




T=huaxian_eemd.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T
L = huaxian_eemd.shape[1]-1
plt.figure(figsize=(7.48,7.48))
for i in range(1,L+1):
    plt.subplot(L,2,2*i-1)
    if i==L:
        plt.xlabel('Time(month)')
    plt.ylabel('IMF'+str(i))
    plt.plot(huaxian_eemd['IMF'+str(i)],color='b',label='',linewidth=0.8)

    plt.subplot(L,2,2*i)
    plt.plot(freqs,abs(fft(huaxian_eemd['IMF'+str(i)])),c='b',lw=0.8)
    if i==L:
        plt.xlabel('Frequence(1/month)')
    plt.ylabel('Amplitude')
plt.tight_layout()



T=huaxian_ssa.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T
L = huaxian_ssa.shape[1]-1
columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise']
plt.figure(figsize=(7.48,7.48))
for i in range(1,L+1):
    plt.subplot(L,2,2*i-1)
    if i==L:
        plt.xlabel('Time(month)')
    plt.ylabel('S'+str(i))
    plt.plot(huaxian_ssa[columns[i-1]],color='b',label='',linewidth=0.8)

    plt.subplot(L,2,2*i)
    plt.plot(freqs,abs(fft(huaxian_ssa[columns[i-1]])),c='b',lw=0.8)
    if i==L:
        plt.xlabel('Hz)')
    plt.ylabel('Amplitude')
plt.tight_layout()



T=huaxian_dwt.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T
L = huaxian_dwt.shape[1]-1
columns=['D1','D2','A2',]
plt.figure(figsize=(7.48,7.48))
for i in range(1,L+1):
    plt.subplot(L,2,2*i-1)
    if i==L:
        plt.xlabel('Time(month)')
    plt.ylabel(columns[i-1])
    plt.plot(huaxian_dwt[columns[i-1]],color='b',label='',linewidth=0.8)

    plt.subplot(L,2,2*i)
    plt.plot(freqs,abs(fft(huaxian_dwt[columns[i-1]])),c='b',lw=0.8)
    if i==L:
        plt.xlabel('Frequence(1/month)')
    plt.ylabel('Amplitude')
plt.tight_layout()

T=huaxian_modwt.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T
L = huaxian_modwt.shape[1]-1
columns=['D1','D2','A2',]
plt.figure(figsize=(7.48,7.48))
for i in range(1,L+1):
    plt.subplot(L,2,2*i-1)
    if i==L:
        plt.xlabel('Time(month)')
    plt.ylabel(columns[i-1])
    plt.plot(huaxian_modwt[columns[i-1]],color='b',label='',linewidth=0.8)

    plt.subplot(L,2,2*i)
    plt.plot(freqs,abs(fft(huaxian_modwt[columns[i-1]])),c='b',lw=0.8)
    if i==L:
        plt.xlabel('Frequence(1/month)')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

