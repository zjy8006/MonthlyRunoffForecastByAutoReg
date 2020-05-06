import os
root = os.path.abspath(os.path.dirname('__file__'))
import sys
sys.path.append(root)
from tools.plot_utils import plot_rela_pred
from tools.dump_data import dum_pred_results
from config.globalLog import logger
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import mean_squared_error

data = pd.read_csv(root+'/time_series/MonthlyRunoffWeiRiver.csv')

station='Huaxian'
model_path = root+'/'+station+'/projects/arima/history/'
if not os.path.exists(model_path):
        os.makedirs(model_path)


series = data[station]
series.plot()
plt.show()

X = series.values
train_len = 552
dev_len = 120
test_len = 120
train, dev, test = X[0:train_len], X[train_len:train_len+dev_len], X[train_len+dev_len:len(X)]

print(train_len)
print(dev_len)
print(test_len)

cal = np.append(train,dev)
print('cal_len={}'.format(len(cal)))

plot_acf(cal)
plot_pacf(cal)
plt.show()

order = (12,1,0)
start = time.process_time()
model = ARIMA(cal,order=order)
model_fit = model.fit(disp=0)
cal_pred = model_fit.fittedvalues
print(len(cal_pred))
train_pred = cal_pred[0:train_len]
dev_pred = cal_pred[train_len:]
print(model_fit.summary())
print(len(train_pred))
print(len(dev_pred))

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

if os.path.exists(model_path +'arima'+str(order)+'_results.csv'):
	logger.info("The arima"+str(order)+" was already tuned")

history = [x for x in cal]
test_pred = list()
for t in range(len(test)):
	model = ARIMA(history, order=order)
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	test_pred.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
end = time.process_time()
time_cost = end-start

# plot_rela_pred(train,train_pred,fig_savepath=model_path  + 'arima'+str(order)+'_train_pred.png')
# plot_rela_pred(dev,dev_pred,fig_savepath=model_path  + "arima"+str(order)+"_dev_pred.png")
plot_rela_pred(test,test_pred,fig_savepath=model_path  + "arima"+str(order)+"_test_pred.png")

dum_pred_results(
            path = model_path+'arima'+str(order)+'_results.csv',
            # train_y = train,
            # train_predictions=train_pred,
            # dev_y = dev,
            # dev_predictions = dev_pred,
            test_y = test,
            test_predictions = test_pred,
            time_cost = time_cost,
            )
