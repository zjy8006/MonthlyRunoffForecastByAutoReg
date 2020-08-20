# 1. 验证序列平稳性
library(forecast)
library(tseries)
plot.ts(hua_train)
hua_train<-as.numeric(unlist(hua_train))
ndiffs(hua_train)
dhua_train<-diff(hua_train)
plot.ts(dhua_train)
ADF<-adf.test(dhua_train)
ADF # 原假设：存在单位根（存在单位根为非平稳时间序列）；拒绝原假设：序列平稳

# 2. 模型自动定阶及拟合
fit <- auto.arima(hua_train)
fit
accuracy(fit)


pacf (dhua_train, 40, ylim=range(-1,1))
acf (dhua_train, 40, ylim=range(-1,1))
fit <- arima(hua_train,order=c(12,1,3))

# 3. 模型诊断
qqnorm(fit$residuals)
qqline(fit$residuals)
Box.test(fit$residuals,type = "Ljung-Box")


# 4. 用ARIMA进行预测
forecast(fit,1)
plot(forecast(fit,1),xlab="month",ylab="Monthly runoff")
