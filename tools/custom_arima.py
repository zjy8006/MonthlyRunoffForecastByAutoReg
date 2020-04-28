import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.ADF_test import is_stationary

def diff(timeseries):
    timeseries_diff1 = timeseries.diff(1)
    timeseries_diff1 = timeseries_diff1.fillna(0)
    timeseries_diff2 = timeseries_diff1.diff(1)
    timeseries_diff2 = timeseries_diff2.fillna(0)
    print(timeseries_diff1)
    print(timeseries_diff2)
    is_stationary(timeseries)
    is_stationary(timeseries_diff1)
    is_stationary(timeseries_diff2)
    plt.figure()
    plt.plot(timeseries,label='Original')
    plt.plot(timeseries_diff1,label='Diff1')
    plt.plot(timeseries_diff2,label='Diff2')
    plt.legend()
    plt.show()

def autocorrelation(timeseries, lags):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()

def decomposing(timeseries):
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(16, 12))
    plt.subplot(411)
    plt.plot(timeseries, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonarity')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.show()

def ARIMA_Model(timeseries, order):
    model = ARIMA(timeseries, order=order)
    return model.fit(disp=0)

Hua = pd.read_excel(root_path+'/time_series/HuaxianRunoff1951-2018(1953-2018).xlsx')['MonthlyRunoff'][24:]
Hua = Hua.reset_index(drop=True)
Hua_diff1 = Hua.diff(1)
Hua_diff1 = Hua_diff1.fillna(0)
Hua_diff2 = Hua_diff1.diff(1)
Hua_diff2 = Hua_diff2.fillna(0)

Hua_train = Hua[:552]
Hua_dev = Hua[552:672]
Hua_test = Hua[672:]
Hua_train = Hua_train.reset_index(drop=True)
Hua_dev = Hua_dev.reset_index(drop=True)
Hua_test = Hua_test.reset_index(drop=True)
print(Hua_train.shape[0])
print(Hua_dev.shape[0])
print(Hua_test.shape[0])
autocorrelation(Hua,20)
autocorrelation(Hua_diff1,20)
model = ARIMA_Model(Hua_diff1,(12,1,12))
Hua_dev_pred=model.predict(start=552, end=672, dynamic=True)

print(Hua_dev_pred)

