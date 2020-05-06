import os
root = os.path.abspath(os.path.dirname('__file__'))
import sys
sys.path.append(root)
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from sklearn.metrics import mean_squared_error


station='Huaxian'
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m')
runoff = pd.read_csv(root+'/time_series/MonthlyRunoffWeiRiver.csv',parse_dates=['Time'],index_col='Time',date_parser=dateparse)

series = runoff[station]
series.plot()
plt.show()

print('ADF for original series:\n{}'.format(ADF(series.values)))

def autocorrelation(timeseries, lags=20):
    fig = plt.figure(figsize=(7.48, 7.48))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.tight_layout()
    plt.show()



def decomposing(timeseries):
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(7.48, 7.48))
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
    plt.tight_layout()
    plt.show()
    trend = trend.fillna(0)
    seasonal = seasonal.fillna(0)
    residual = residual.fillna(0)
    return trend,seasonal,residual




def ARIMA_Model(timeseries, order):
    model = ARIMA(timeseries, order=order)
    return model.fit(disp=0)


train = series['1953-01-01':'2008-12-01']
test = series['2009-01-01':'2018-12-01']

trend,seasonal,residual = decomposing(train)

# autocorrelation(trend)
# tr_p = int(input("Trend p="))
# tr_q = int(input("Trend q="))
# autocorrelation(residual)
# res_p = int(input("Residual p="))
# res_q = int(input("Residual q="))

trend_model = ARIMA_Model(trend, (4, 0, 3))
trend_fit_seq = trend_model.fittedvalues
trend_predict_seq = trend_model.predict(start='2009-01-01', end='2018-12-01', dynamic=True)

seasonal_model = ARIMA_Model(seasonal, (6, 0, 0))
seasonal_fit_seq = seasonal_model.fittedvalues
seasonal_predict_seq = seasonal_model.predict(start='2009-01-01', end='2018-12-01', dynamic=True)

residual_model = ARIMA_Model(residual, (3, 0, 2))
residual_fit_seq = residual_model.fittedvalues
residual_predict_seq = residual_model.predict(start='2009-01-01', end='2018-12-01', dynamic=True)


fit_seq = pd.Series()
fit_seq = fit_seq.add(trend_fit_seq, fill_value=0)
fit_seq = fit_seq.add(seasonal_fit_seq, fill_value=0)
fit_seq = fit_seq.add(residual_fit_seq, fill_value=0)

plt.plot(fit_seq, color='red', label='fit_seq')
plt.plot(train, color='blue', label='train')
plt.legend(loc='best')
plt.show()



predict_seq = pd.Series()
predict_seq = predict_seq.add(trend_predict_seq, fill_value=0)
predict_seq = predict_seq.add(seasonal_predict_seq, fill_value=0)
predict_seq = predict_seq.add(residual_predict_seq, fill_value=0)

plt.plot(predict_seq, color='red', label='predict_seq')
plt.plot(test, color='blue', label='test')
plt.legend(loc='best')
plt.show()
