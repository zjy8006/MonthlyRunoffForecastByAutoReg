data = readtable('../../time_series/MonthlyRunoffWeiRiver.csv');
Huaxian = data.Huaxian;
train_dev = Huaxian(1:672);
test = Huaxian(673:792);
EsMd1 = modelTimeSeries303313G(train_dev);
[Y,YMSE] = forecast(EsMd1,120,train_dev);
plot_pred(test,Y,'./arima/history/sarima_303313G_test_pred.png')

EsMd2 = modelTimeSeries310313T(train_dev);
[Y,YMSE] = forecast(EsMd2,120,train_dev);
plot_pred(test,Y,'./arima/history/sarima_310313T_test_pred.png')