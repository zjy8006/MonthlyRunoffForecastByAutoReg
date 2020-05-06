import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.size'] = 6
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
graphs_path = root_path+'/boundary_effect/graph/'
if not os.path.exists(graphs_path):
    os.makedirs(graphs_path)

time = pd.read_csv(root_path+'/time_series/MonthlyRunoffWeiRiver.csv')['Time']
time = time.values
time = [datetime.strptime(t,'%Y/%m') for t in time]
time = [t.strftime('%b %Y') for t in time]
print(time)





# CHECK 1: is SSA shift-invariant?
# If yes, any shifted copy of an IMF from a SSA decomposition, similar to a
# shifted copy of the original time series, should be maintained.

# For example, given the sunspot time series x (of length 792) we can
# generate a 1-step advanced copy of the original time series as follows:
# x0=(1:791)
# x1=(2:792) this is a 1-step advanced version of x0
# Observiously, shift-invariancy is preserved between x0 and x1 since
# x0(2:791)=x1(1:790)

# For shift-invariancy to be preserved for SSA, we would observe, for
# example, that the SSA Trend components for x0 (imf1 of x0) and x1 (imf1 of
# x1) should be exact copies of one another, advanced by a single step.
# i.e., x0_imf(2:791,1) should equal x1_imf(1:790,1) if shift-invariancy
# is preserved.

# As the case for SSA shown below, we can see the x0_imf(2:791,1) basically
# equal to x1_imf(1:790,1) except for a few samples close to the begin and
# end of x0 and x1. Interestingly, we see a low level of error close to the
# begin of the time series and a high level of error close to the end of
# the time series, of high importance in operational forecasting tasks. 
# The errors along the middle range are zeros indicating SSA is
# shift-invariant.
# We argue that the error close to the boundaries are
# caused by boundary effect, which is the exact problem this study designed
# to solve.

# CHECK 2: The impact of appedning data points to a time series then
# performing SSA, analogous the case in operational forecasting when new
# data becomes available and an updated forecast is made using the newly
# arrived data.

# Ideally, for forecasting situations, when new data is appended to a time
# series and some preprocessing is performed, it should not have an impact
# on previous measurements of the pre-processed time series.

# For example, if Trend_1:N represents the Trend, which has N total
# measurements and was derived by applying SSA to x_1:N the we would expect
# that when we perform SSA when x is appended with another measurement,
# i.e., x_1:N+1, resulting in Trend_1:N+1 that the first 1:N measurements in
# Trend_1:N+1 are equal to Trend_1:N. In other words, 
# Trend_1:N+1[1:N]=Trend_1:N[1:N].

# We see than is not the case. Appending an additional observation to the
# time series results in the updated SSA components to be entirely
# different then the original (as of yet updated) SSA components.
# Interesting, we see a high level of error at the boundaries of the time
# seriesm, of high importance in operational forecasting tasks.

x0_imf = pd.read_csv(root_path+'/boundary_effect/ssa-decompositions-huaxian/x0_dec.csv')
x1_imf = pd.read_csv(root_path+'/boundary_effect/ssa-decompositions-huaxian/x1_dec.csv')
x_1_552_imf = pd.read_csv(root_path+"/boundary_effect/ssa-decompositions-huaxian/x_1_552_dec.csv")
x_1_791_imf = pd.read_csv(root_path+'/boundary_effect/ssa-decompositions-huaxian/x_1_791_dec.csv')
x_1_792_imf = pd.read_csv(root_path+'/boundary_effect/ssa-decompositions-huaxian/x_1_792_dec.csv')


x0_imf1_2_791 = x0_imf['Trend'][1:790]
x0_imf1_2_791 = x0_imf1_2_791.reset_index(drop=True)
x1_imf1_1_790 = x1_imf['Trend'][0:789]
x1_imf1_1_790 = x1_imf1_1_790.reset_index(drop=True)

err = x0_imf1_2_791-x1_imf1_1_790
# err_df = pd.DataFrame(err.values,columns=['err'])
# print(err)
err.to_csv(root_path+'/results_analysis/results/shift_variance_err.csv')

x_1_552_imf1 = x_1_552_imf['Trend']
x_1_791_imf1 = x_1_791_imf['Trend']
x_1_792_imf1 = x_1_792_imf['Trend']

err_append_one = x_1_792_imf1[0:790]-x_1_791_imf1[0:790]
err_append_several = x_1_792_imf1[0:551]-x_1_552_imf1[0:551]
err_append_one_df = pd.DataFrame(err_append_one,columns=['err'])
err_append_several_df = pd.DataFrame(err_append_several,columns=['err'])
print(err_append_one_df)
print(err_append_several_df)
err_append_one.to_csv(root_path+'/results_analysis/results/err_append_one.csv')
err_append_several.to_csv(root_path+'/results_analysis/results/err_append_several.csv')


xx = -6
aceg_y = 13.5
bdf_y = 2.4
y_min = 0
y_max = 16
ye_min = -1.3
ye_max = 3.1


plt.figure(figsize=(7.48,6))
plt.subplot(4,2,1)
plt.text(xx,aceg_y,'(a)',fontsize=7,fontweight='bold',bbox=dict(facecolor='thistle', alpha=0.25))
plt.plot(x0_imf1_2_791,c='b',label=r'$S_{1}(2:791)$ of $x_{0}$')
plt.plot(x1_imf1_1_790,c='g',label=r'$S_{1}(1:790)$ of $x_{1}$')
plt.xlabel('Time (From '+time[1]+' to '+time[790]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.ylim(y_min,y_max)
plt.legend(ncol=2)
plt.subplot(4,2,2)
plt.text(760,-0.105,'(b)',fontsize=7,fontweight='bold',bbox=dict(facecolor='thistle', alpha=0.25))
shift_var=plt.plot(err,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,
label=R'''Error between  $S_{1}(2:791)$ 
of $x_{0}$ and  $S_{1}(1:790)$ of $x_{1}$''')
plt.xlabel('Time (From '+time[1]+' to '+time[790]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.legend(loc='lower center')
plt.subplot(4,2,3)
plt.text(xx,aceg_y,'(c)',fontsize=7,fontweight='bold',bbox=dict(facecolor='thistle', alpha=0.25))
plt.plot(x_1_791_imf1,c='b',label=r'$S_{1}$ of $x_{1-791}$')
plt.plot(x_1_792_imf1,c='g',label=r'$S_{1}$ of $x_{1-792}$')
plt.xlabel('Time (From '+time[0]+' to '+time[791]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.ylim(y_min,y_max)
plt.legend(ncol=2)
plt.subplot(4,2,4)
plt.text(xx,0.04,'(d)',fontsize=7,fontweight='bold',bbox=dict(facecolor='thistle', alpha=0.25))
plt.plot(err_append_one,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,
label=R'''Error between  $S_{1}(1:791)$ of 
$x_{1-791}$ and  $S_{1}(1:791)$ of $x_{1-792}$''')
plt.xlabel('Time (From '+time[0]+' to '+time[790]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.legend(loc='lower left')
plt.subplot(4,2,5)
plt.text(xx,aceg_y,'(e)',fontsize=7,fontweight='bold',bbox=dict(facecolor='thistle', alpha=0.25))
plt.plot(x_1_552_imf1,c='b',label=r'$S_{1}$ of $x_{1-552}$')
plt.plot(x_1_792_imf1,c='g',label=r'$S_{1}$ of $x_{1-792}$')
plt.xlabel('Time (From '+time[0]+' to '+time[791]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.ylim(y_min,y_max)
plt.legend(ncol=2)
plt.subplot(4,2,6)
plt.text(xx,0.18,'(f)',fontsize=7,fontweight='bold',bbox=dict(facecolor='thistle', alpha=0.25))
plt.plot(err_append_several,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,
label=R'''Error between  $S_{1}(1:552)$ of 
$x_{1-552}$ and  $S_{1}(1:552)$ of $x_{1-792}$''')
plt.xlabel('Time (From '+time[0]+' to '+time[551]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.legend(loc=9,ncol=2)

ssa_train = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_TRAIN.csv")
ssa_full = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_FULL.csv")
seq_val_dec = pd.DataFrame()
for subsignal in ['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5','Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise']:
    test_imf = []
    for i in range(553,792+1):
        data=pd.read_csv(root_path+"/Huaxian_ssa/data/ssa-test/ssa_appended_test"+str(i)+".csv")
        test_imf.append((data[subsignal].iloc[data.shape[0]-1:]).values.flatten()[0])
    val_subsignal = pd.DataFrame(test_imf,columns=[subsignal])
    seq_val_dec = pd.concat([seq_val_dec,val_subsignal],axis=1)
seq_val_dec_sum = seq_val_dec.sum(axis=1)
print(seq_val_dec_sum)
t_e=list(range(1,793))
t_t=list(range(1,553))


gh_x = 551
orig = (ssa_full['ORIG'].iloc[ssa_full.shape[0]-240:]).values
concurrent_dec = (ssa_full['Trend'].iloc[ssa_full.shape[0]-240:]).values
err = (seq_val_dec['Trend']).values - concurrent_dec
plt.subplot(4,2,7)
plt.text(gh_x,aceg_y,'(g)',fontsize=7,fontweight='bold',bbox=dict(facecolor='thistle', alpha=0.25))
t=list(range(553,793))
plt.plot(t,seq_val_dec['Trend'],c='b',label=r"$S_1$ of sequential validation decomposition")
plt.plot(t,concurrent_dec,c='g',label=r"$S_1$ of concurrent validation decomposition")
# plt.xlabel("Time(1999/01-2018/12)")
plt.xlabel('Time (From '+time[552]+' to '+time[791]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.ylim(y_min,y_max)
# plt.xlim([550,560])
plt.legend(loc='upper right')
plt.subplot(4,2,8)
plt.text(gh_x,46.5,'(h)',fontsize=7,fontweight='bold',bbox=dict(facecolor='thistle', alpha=0.25))
ssa_full=ssa_full.drop('ORIG',axis=1)
print('ssa_full=\n{}'.format(ssa_full))
con_val = ssa_full[ssa_full.shape[0]-240:]
con_val = con_val.reset_index(drop=True)
con_val_dec_sum = con_val.sum(axis=1)
plt.xlabel('Time (From '+time[552]+' to '+time[791]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.ylim(-1.5,55)
plt.plot(t,orig,c='b',label=r"Validation set")
plt.plot(t,con_val_dec_sum,c='black',label=r"Summation of concurrent validation decompositions")
plt.plot(t,seq_val_dec_sum,c='g',label=r"Summation of sequential validation decompositions")
# plt.plot(t,err,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,label=r'Error between sequentially and concurrently decomposed $S_1$')
plt.legend()
plt.subplots_adjust(left=0.066, bottom=0.06, right=0.99,top=0.99, hspace=0.35, wspace=0.2)
# plt.tight_layout()
plt.savefig(graphs_path+'/Boundary effect of SSA Trend at Huaxian.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/Boundary effect of SSA Trend at Huaxian.pdf',format='PDF',dpi=1200)


plt.figure(figsize=(0.4,0.4))
plt.plot(x0_imf1_2_791,c='b',label=r'$S_{1}[2:791]$ of $x_{0}$')
plt.plot(x1_imf1_1_790,c='g',label=r'$S_{1}[1:790]$ of $x_{1}$')
plt.xticks([])
plt.yticks([])
plt.xlim(776,790)
plt.ylim(4.8,6.2)
plt.subplots_adjust(left=0.00, bottom=0.00, right=0.99,top=0.99, hspace=0.0, wspace=0.0)
plt.savefig(graphs_path+'/Boundary effect of SSA at Huaxian(minimap-a11).eps',format='EPS',dpi=2000)

plt.figure(figsize=(0.4,0.4))
plt.plot(x0_imf1_2_791,c='b',label=r'$S_{1}[2:791]$ of $x_{0}$')
plt.plot(x1_imf1_1_790,c='g',label=r'$S_{1}[1:790]$ of $x_{1}$')
plt.xticks([])
plt.yticks([])
plt.xlim(-1,10)
plt.ylim(4.8,6.0)
plt.subplots_adjust(left=0.00, bottom=0.00, right=0.99,top=0.99, hspace=0.0, wspace=0.0)
plt.savefig(graphs_path+'/Boundary effect of SSA at Huaxian(minimap-a12).eps',format='EPS',dpi=2000)

plt.figure(figsize=(0.4,0.4))
plt.plot(x_1_791_imf1,c='b',label=r'$S_{1}$ of $x_{1-791}$')
plt.plot(x_1_792_imf1,c='g',label=r'$S_{1}$ of $x_{1-792}$')
plt.xticks([])
plt.yticks([])
plt.xlim(776,792)
plt.ylim(4.8,6.2)
plt.subplots_adjust(left=0.00, bottom=0.00, right=0.99,top=0.99, hspace=0.0, wspace=0.0)
plt.savefig(graphs_path+'/Boundary effect of SSA at Huaxian(minimap-b11).eps',format='EPS',dpi=2000)

plt.figure(figsize=(0.4,0.4))
plt.plot(x_1_791_imf1,c='b',label=r'$S_{1}$ of $x_{1-791}$')
plt.plot(x_1_792_imf1,c='g',label=r'$S_{1}$ of $x_{1-792}$')
plt.xticks([])
plt.yticks([])
plt.xlim(-1,10)
plt.ylim(4,5.5)
plt.subplots_adjust(left=0.00, bottom=0.00, right=0.99,top=0.99, hspace=0.0, wspace=0.0)
plt.savefig(graphs_path+'/Boundary effect of SSA at Huaxian(minimap-b12).eps',format='EPS',dpi=2000)

plt.figure(figsize=(0.4,0.4))
plt.plot(x_1_552_imf1,c='b',label=r'$S_{1}$ of $x_{1-552}$')
plt.plot(x_1_792_imf1,c='g',label=r'$S_{1}$ of $x_{1-792}$')
plt.xticks([])
plt.yticks([])
plt.xlim(540,555)
plt.ylim(2.7,3.4)
plt.subplots_adjust(left=0.00, bottom=0.00, right=0.99,top=0.99, hspace=0.0, wspace=0.0)
plt.savefig(graphs_path+'/Boundary effect of SSA at Huaxian(minimap-c11).eps',format='EPS',dpi=2000)

plt.show()
