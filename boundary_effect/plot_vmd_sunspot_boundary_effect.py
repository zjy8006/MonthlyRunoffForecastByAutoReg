import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 6
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
graphs_path = root_path+'/graphs/'
if not os.path.exists(graphs_path):
    os.makedirs(graphs_path)
# CHECK 1: is VMD shift-invariant?
# If yes, any shifted copy of an IMF from a VMD decomposition, similar to a
# shifted copy of the original time series, should be maintained.

# For example, given the sunspot time series x (of length 386) we can
# generate a 1-step advanced copy of the original time series as follows:
# x0=(1:385)
# x1=(2:386) this is a 1-step advanced version of x0
# Observiously, shift-invariancy is preserved between x0 and x1 since
# x0(2:385)=x1(1:384)

# For shift-invariancy to be preserved for VMD, we would observe, for
# example, that the VMD IMF1 components for x0 (imf1 of x0) and x1 (imf1 of
# x1) should be exact copies of one another, advanced by a single step.
# i.e., x0_imf(2:385,1) should equal x1_imf(1:384,1) if shift-invariancy
# is preserved.

# As the case for VMD shown below, we can see the x0_imf(2:385,1) basically
# equal to x1_imf(1:384,1) except for a few samples close to the begin and
# end of x0 and x1. Interestingly, we see a low level of error close to the
# begin of the time series and a high level of error close to the end of
# the time series, of high importance in operational forecasting tasks. 
# The errors along the middle range are zeros indicating VMD is
# shift-invariant.
# We argue that the error close to the boundaries are
# caused by boundary effect, which is the exact problem this study designed
# to solve.

# CHECK 2: The impact of appedning data points to a time series then
# performing VMD, analogous the case in operational forecasting when new
# data becomes available and an updated forecast is made using the newly
# arrived data.

# Ideally, for forecasting situations, when new data is appended to a time
# series and some preprocessing is performed, it should not have an impact
# on previous measurements of the pre-processed time series.

# For example, if IMF1_1:N represents the IMF1, which has N total
# measurements and was derived by applying VMD to x_1:N the we would expect
# that when we perform VMD when x is appended with another measurement,
# i.e., x_1:N+1, resulting in IMF1_1:N+1 that the first 1:N measurements in
# IMF1_1:N+1 are equal to IMF1_1:N. In other words, 
# IMF1_1:N+1[1:N]=IMF1_1:N[1:N].

# We see than is not the case. Appending an additional observation to the
# time series results in the updated VMD components to be entirely
# different then the original (as of yet updated) VMD components.
# Interesting, we see a high level of error at the boundaries of the time
# seriesm, of high importance in operational forecasting tasks.

x0_imf = pd.read_csv(root_path+'/boundary_effect/decompositions/x0_imf.csv')
x1_imf = pd.read_csv(root_path+'/boundary_effect/decompositions/x1_imf.csv')
x_1_300_imf = pd.read_csv(root_path+"/boundary_effect/decompositions/x_1_300_imf.csv")
x_1_385_imf = pd.read_csv(root_path+'/boundary_effect/decompositions/x_1_385_imf.csv')
x_1_386_imf = pd.read_csv(root_path+'/boundary_effect/decompositions/x_1_386_imf.csv')


x0_imf1_2_385 = x0_imf['IMF1'][1:384]
x0_imf1_2_385 = x0_imf1_2_385.reset_index(drop=True)
x1_imf1_1_384 = x1_imf['IMF1'][0:383]
x1_imf1_1_384 = x1_imf1_1_384.reset_index(drop=True)

err = x0_imf1_2_385-x1_imf1_1_384


x_1_300_imf1 = x_1_300_imf['IMF1']
x_1_385_imf1 = x_1_385_imf['IMF1']
x_1_386_imf1 = x_1_386_imf['IMF1']

err_append_one = x_1_386_imf1[0:384]-x_1_385_imf1[0:384]
err_append_several = x_1_386_imf1[0:299]-x_1_300_imf1[0:299]


plt.figure(figsize=(7.48,7.48))
plt.subplot(4,2,1)
plt.plot(x0_imf1_2_385,c='b',label=r'$IMF_{1}[2:385]$ of $x_{0}$')
plt.plot(x1_imf1_1_384,c='g',label=r'$IMF_{1}[1:384]$ of $x_{1}$')
plt.xlabel('Time(1611-1994)\n(a1)')
plt.ylabel('Sunspot')
plt.legend()
plt.subplot(4,2,2)
plt.plot(err,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,label=r'Error between $IMF_{1}[2:385]$ of $x_{0}$ and $IMF_{1}[1:384]$ of $x_{1}$')
plt.xlabel('Time(1611-1994)\n(a2)')
plt.ylabel('Sunspot')
plt.legend()
plt.subplot(4,2,3)
plt.plot(x_1_385_imf1,c='b',label=r'$IMF_{1}$ of $x_{1-385}$')
plt.plot(x_1_386_imf1,c='g',label=r'$IMF_{1}$ of $x_{1-386}$')
plt.xlabel('Time(1610-1995)\n(b1)')
plt.ylabel('Sunspot')
plt.legend()
plt.subplot(4,2,4)
plt.plot(err_append_one,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,label=r'Error between  $IMF_{1}[1:385]$ of $x_{1-385}$ and $IMF_{1}[1:385]$ of $x_{1-386}$')
plt.xlabel('Time(1610-1994)\n(b2)')
plt.ylabel('Sunspot')
plt.legend()
plt.subplot(4,2,5)
plt.plot(x_1_300_imf1,c='b',label=r'$IMF_{1}$ of $x_{1-300}$')
plt.plot(x_1_386_imf1,c='g',label=r'$IMF_{1}$ of $x_{1-386}$')
plt.xlabel('Time(1610-1995)\n(c1)')
plt.ylabel('Sunspot')
plt.legend()
plt.subplot(4,2,6)
plt.plot(err_append_several,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,label=r'Error between  $IMF_{1}[1:300]$ of $x_{1-300}$ and $IMF_{1}[1:300]$ of $x_{1-386}$')
plt.xlabel('Time(1610-1909)\n(c2)')
plt.ylabel('Sunspot')
plt.legend()


vmd_train = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_TRAIN.csv")
vmd_full = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_FULL.csv")
subsignal="IMF1"
test_imf = []
for i in range(553,792+1):
    data=pd.read_csv(root_path+"/Huaxian_vmd/data/vmd-test/vmd_appended_test"+str(i)+".csv")
    test_imf.append((data[subsignal].iloc[data.shape[0]-1:]).values.flatten()[0])
t_e=list(range(1,793))
t_t=list(range(1,553))

plt.subplot(4,2,7)
plt.plot(t_t,vmd_train[subsignal],c='b',label="Concurrent decomposition of training set")
plt.plot(t_e,vmd_full[subsignal],c='g',label="Concurrent decomposition of entire streamflow")
plt.xlabel("Time(1997/02-1999/08)\n(d1)")
plt.ylabel(r"Runoff($10^8m^3$)")
plt.xlim([530,560])
plt.ylim([2,4])
plt.legend()

plt.subplot(4,2,8)
t=list(range(553,793))
plt.plot(t,test_imf,c='b',label="Sequential decomposition of validation set")
plt.plot(t,vmd_full[subsignal].iloc[vmd_full.shape[0]-240:],c='g',label="Concurrent decomposition of entire streamflow")
plt.xlabel("Time(1999/01-2018/12)\n(d2)")
plt.ylabel(r"Runoff($10^8m^3$)")
# plt.xlim([550,560])
plt.ylim([0,12])
plt.legend()
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.98,top=0.99, hspace=0.4, wspace=0.15)
# plt.tight_layout()
plt.savefig(graphs_path+'/Boundary effect for sunspot.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/Boundary effect for sunspot.tif',format='TIFF',dpi=1200)
plt.savefig(graphs_path+'/Boundary effect for sunspot.pdf',format='PDF',dpi=1200)
plt.show()
