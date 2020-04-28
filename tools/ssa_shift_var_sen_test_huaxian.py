# Example MATLAB script, inspired by John Quilty, to show the
# shift-invariancy and sensitivity to adding additional data points when
# using SSA for to provide inputs for operational forecasting tasks.
import pandas as pd
import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.SSA_tool import SSA_decomposition

runoff = pd.read_csv(root_path+'/time_series/MonthlyRunoffWeiRiver.csv')
x = runoff['Huaxian']
time = runoff['Time']
# CHECK 1: is SSA shift-invariant?
# If yes, any shifted copy of an IMF from a SSA decomposition, similar to a
# shifted copy of the original time series, should be maintained.

# For example, given the sunspot time series x (of length 792) we can
# generate a 1-step advanced copy of the original time series as follows:
# x0=(0:790)
# x1=(1:791) this is a 1-step advanced version of x0
# Observiously, shift-invariancy is preserved between x0 and x1 since
# x0(1:790)=x1(0:789)

# For shift-invariancy to be preserved for SSA, we would observe, for
# example, that the SSA Trend components for x0 (Trend of x0) and x1 (Trend of
# x1) should be exact copies of one another, advanced by a single step.
# i.e., x0_trend(1:790) should equal x1_trend(0:789) if shift-invariancy
# is preserved.

# As the case for SSA shown below, we can see the x0_trend(1:790) basically
# equal to x1_trend(0:789) except for a few samples close to the begin and
# end of x0 and x1. Interestingly, we see a low level of error close to the
# begin of the time series and a high level of error close to the end of
# the time series, of high importance in operational forecasting tasks. 
# The errors along the middle range are zeros indicating SSA is
# shift-invariant.
# We argue that the error close to the boundaries are
# caused by boundary effect, which is the exact problem this study designed
# to solve.
x0=x[0:791];
print(x0)
x0=x0.reset_index(drop=True)
x1=x[1:792];
print(x1)
x1=x1.reset_index(drop=True)
x0_dec = SSA_decomposition(time_series=x0,window=12)
x1_dec = SSA_decomposition(time_series=x1,window=12)
x0_dec_trend_1_790 = x0_dec['Trend'][1:791]
x0_dec_trend_1_790 = x0_dec_trend_1_790.reset_index(drop=True)
x1_dec_trend_0_789 = x1_dec['Trend'][0:790]
x1_dec_trend_0_789 = x1_dec_trend_0_789.reset_index(drop=True)

plt.figure(figsize=(7.48,3.48))
plt.subplot(1,2,1)
plt.xlabel('Time('+time[1]+'-'+time[790]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.plot(x0_dec_trend_1_790,c='b',label='Trend of X0('+time[1]+'-'+time[790]+')')
plt.plot(x1_dec_trend_0_789,c='r',label='Trend of X0('+time[1]+'-'+time[790]+')')
plt.legend()
plt.subplot(1,2,2)
err = x0_dec_trend_1_790-x1_dec_trend_0_789
plt.xlabel('Time('+time[1]+'-'+time[790]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.plot(err,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,
    label=r'Error between $Trend[1:790]$ of $x_{0}$ and $Trend[0:789]$ of $x_{1}$')
plt.tight_layout()

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
x_0_790=x[0:791];
x_0_791=x[0:792];
x_0_790_dec = SSA_decomposition(time_series=x_0_790,window=12)
x_0_791_dec = SSA_decomposition(time_series=x_0_791,window=12)
plt.figure(figsize=(7.48,3.48))
plt.subplot(1,2,1)
plt.xlabel('Time('+time[0]+'-'+time[791]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.plot(x_0_790_dec['Trend'],c='b',label='Trend of X0('+time[0]+'-'+time[790]+')')
plt.plot(x_0_791_dec['Trend'],c='r',label='Trend of X0('+time[0]+'-'+time[791]+')')
plt.legend()
plt.subplot(1,2,2)
x_0_790_dec_trend_0_790 = x_0_790_dec['Trend'][0:791]
x_0_791_dec_trend_0_790 = x_0_791_dec['Trend'][0:791]
err = x_0_790_dec_trend_0_790-x_0_791_dec_trend_0_790
plt.xlabel('Time('+time[0]+'-'+time[790]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.plot(err,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,
    label=r'Error between $Trend[0:790]$ of $x_{0}$ and $Trend[0:791]$ of $x_{1}$')
plt.tight_layout()

# Check 3: append several data points
x_0_551=x[0:552];
x_0_551_dec = SSA_decomposition(time_series=x_0_551,window=12)
plt.figure(figsize=(7.48,3.48))
plt.subplot(1,2,1)
plt.xlabel('Time('+time[0]+'-'+time[791]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.plot(x_0_551_dec['Trend'],c='b',label='Trend of X0('+time[0]+'-'+time[551]+')')
plt.plot(x_0_791_dec['Trend'],c='r',label='Trend of X0('+time[0]+'-'+time[791]+')')
plt.legend()
plt.subplot(1,2,2)
x_0_551_dec_trend_0_551 = x_0_551_dec['Trend'][0:552]
x_0_791_dec_trend_0_790 = x_0_791_dec['Trend'][0:552]
err = x_0_551_dec_trend_0_551-x_0_791_dec_trend_0_790
plt.xlabel('Time('+time[0]+'-'+time[551]+')')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.plot(err,'o',markerfacecolor='w',markeredgecolor='r',markersize=4.5,
    label=r'Error between $Trend[0:551]$ of $x_{0}$ and $Trend[0:551]$ of $x_{1}$')
plt.tight_layout()
plt.show()


