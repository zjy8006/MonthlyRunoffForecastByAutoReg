function ARIMA_data1 = modelTimeSeries(data)
%%Time Series Modeling Using the Econometric Modeler
% This code recreates the estimated model produced in the Econometric Modeler app. Use the code below to estimate the same model, or estimate a model with the same parameters on a new set of data.
%
%Input: A numeric matrix with the same number of columns as the data imported into the app (data)
%
%Output: The model containing estimated parameters (ARIMA_data1)
%
%Auto-generated by MATLAB Version 9.7 (R2019b) and Econometrics Toolbox Version 5.3 on 30-Apr-2020 15:17:01
data1 = data(:,1);

%% Autoregressive Integrated Moving Average Model
%Estimate an ARIMA Model of data1
ARIMA_data1 = arima('Constant',NaN,'ARLags',1:12,'D',1,'MALags',1:12,'Distribution','Gaussian');
ARIMA_data1 = estimate(ARIMA_data1,data1,'Display','off');
end