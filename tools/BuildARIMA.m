data = readtable('../time_series/MonthlyRunoffWeiRiver.csv');
hua = data.Huaxian;
xian =data.Xianyang;
zhang = data.Zhangjiashan;
p=12;
D=1;
q=12;
Mdl = arima(p,D,q);
EstMdl = estimate(Mdl,hua(1:672));
[Y,YMSE] = forecast(EstMdl,120,hua(1:672));