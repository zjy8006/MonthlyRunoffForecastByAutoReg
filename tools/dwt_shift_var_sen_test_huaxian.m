% Example MATLAB script, inspired by John Quilty, to show the
% shift-invariancy and sensitivity to adding additional data points when
% using DWT for to provide inputs for operational forecasting tasks.
clear;
close all;
clc;

data_path='../time_series/';
save_path = '../boundary_effect/dwt-decompositions-huaxian/';
if exist(save_path)==0
    mkdir(save_path);
end

runoff = readtable('../time_series/MonthlyRunoffWeiRiver.csv');
x = runoff.Huaxian;
time = runoff.Time;

x0=x(1:791);
x1=x(2:792);

%% some sample parameters for DWT             
mother_wavelet = 'db10';% set mother wavelet
lev = 2; %same decomposition level as DWT


% CHECK 1: is DWT shift-invariant?
% If yes, any shifted copy of an subsignal from a DWT decomposition, similar to a
% shifted copy of the original time series, should be maintained.

% For example, given the sunspot time series x (of length 792) we can
% generate a 1-step advanced copy of the original time series as follows:
% x0=(1:791)
% x1=(2:792) this is a 1-step advanced version of x0
% Observiously, shift-invariancy is preserved between x0 and x1 since
% x0(2:791)=x1(1:790)

% For shift-invariancy to be preserved for DWT, we would observe, for
% example, that the DWT 1st subsignal components for x0 (D1 of x0) and x1 (D1 of
% x1) should be exact copies of one another, advanced by a single step.
% i.e., x0_D(2:791,1) should equal x1_D(1:790,1) if shift-invariancy
% is preserved.

% As the case for DWT shown below, we can see the x0_D(2:791,1) basically
% equal to x1_D(1:790,1) except for a few samples close to the begin and
% end of x0 and x1. Interestingly, we see a low level of error close to the
% begin of the time series and a high level of error close to the end of
% the time series, of high importance in operational forecasting tasks. 
% The errors along the middle range are zeros indicating DWT is
% shift-invariant.
% We argue that the error close to the boundaries are
% caused by boundary effect, which is the exact problem this study designed
% to solve.
setdemorandstream(pi);

[x0_dec] = my_dwt(x0', mother_wavelet, lev);
writetable(x0_dec, [save_path,'x0_dec.csv']);
[x1_dec] = my_dwt(x1', mother_wavelet, lev);
writetable(x1_dec, [save_path,'x1_dec.csv']);


figure('Name','Test shift-invariancy for DWT');
hold on
x0_D1_2_791=plot(x0_dec.D1(2:791,1),'b');
label0=strcat('D1 of x0(',num2str(year(2)),'-',num2str(year(791)),')');
x1_D1_1_790=plot(x1_dec.D1(1:790,1),'r');
label1=strcat('D1 of x1(',num2str(year(2)),'-',num2str(year(791)),')');
legend([x0_D1_2_791;x1_D1_1_790],label0,label1,'Location','northwest');
hold off

err = x0_dec.D1(2:791,1)-x1_dec.D1(1:790,1);
figure('Name','error between x0 and x1');
scatter(linspace(1,790,790),err,'Marker','o');

% Check the level of error (as measuured by the mean square error) between
% the D components.
mse=mean(err.^2);

% CHECK 2: The impact of appedning data points to a time series then
% performing DWT, analogous the case in operational forecasting when new
% data becomes available and an updated forecast is made using the newly
% arrived data.

% Ideally, for forecasting situations, when new data is appended to a time
% series and some preprocessing is performed, it should not have an impact
% on previous measurements of the pre-processed time series.

% For example, if D1_1:N represents the D1, which has N total
% measurements and was derived by applying DWT to x_1:N the we would expect
% that when we perform DWT when x is appended with another measurement,
% i.e., x_1:N+1, resulting in D1_1:N+1 that the first 1:N measurements in
% D1_1:N+1 are equal to D1_1:N. In other words, 
% D1_1:N+1[1:N]=D1_1:N[1:N].

% We see than is not the case. Appending an additional observation to the
% time series results in the updated DWT components to be entirely
% different then the original (as of yet updated) DWT components.
% Interesting, we see a high level of error at the boundaries of the time
% seriesm, of high importance in operational forecasting tasks.

% Extend x with an additional measurement
x_1_791=x(1:791);
x_1_792=x(1:792);

% Perform DWT on original and extended time series.
[x_1_791_dec] = my_dwt(x_1_791', mother_wavelet, lev);
writetable(x_1_791_dec, [save_path,'x_1_791_dec.csv']);
[x_1_792_dec] = my_dwt(x_1_792', mother_wavelet, lev);
writetable(x_1_792_dec, [save_path,'x_1_792_dec.csv']);


figure('Name','Test sensitivity of adding additional data point');
hold on
x_1_792_D1=plot(x_1_792_dec.D1(1:792,1),'b');
label00=strcat('D1 of x\_1\_792(',num2str(year(1)),'-',num2str(year(792)),')');
x_1_791_D1=plot(x_1_791_dec.D1(1:791,1),'r');
label11=strcat('D1 of x\_1\_791(',num2str(year(1)),'-',num2str(year(791)),')');
legend([x_1_792_D1;x_1_791_D1],label00,label11,'Location','northwest');
hold off

% Plot error (which should be a straight line of 0s if appending an 
% additional observation has on impact on DWT)
err_append = x_1_792_dec.D1(1:791,1)-x_1_791_dec.D1(1:791,1)
figure('Name','error between x_1_792_dec(1:791,1) and x_1_791_dec(1:791,1)');
scatter(linspace(1,791,791),err_append,'Marker','o')

% Check the level of error (as measured by the mean square error) between
% the D components.
mse_append = mean(err_append.^2);

% The problem gets exasperated if it is not a single time point and that is
% updated, but several.
x_1_552=x(1:552);

% Perform DWT on the original and extended time series.
[x_1_552_dec] = my_dwt(x_1_552', mother_wavelet, lev);
writetable(x_1_552_dec, [save_path,'x_1_552_dec.csv']);

% Plot D for x_1_552 and x_1_792
figure('Name','Test sensitivity of adding several additional data points');
hold on
x_1_792_D1=plot(x_1_792_dec.D1(1:792,1),'b');
label00=strcat('D1 of x\_1\_792(',num2str(year(1)),'-',num2str(year(792)),')');
x_1_552_D1=plot(x_1_552_dec.D1(1:552,1),'r');
label11=strcat('D1 of x\_1\_552(',num2str(year(1)),'-',num2str(year(552)),')');
legend([x_1_792_D1;x_1_552_D1],label00,label11,'Location','northwest');
hold off

% Plot error (which should be a straight line of 0s if appending several
% additional observations has no impact on DWT)
err_append_several = x_1_552_dec.D1(1:552,1)-x_1_792_dec.D1(1:552,1);
figure('Name','error between x_1_552_dec(1:552,1) and x_1_792_dec(1:552,1)');
scatter(linspace(1,552,552),err_append_several,'Marker','o');

% Check the level of error (as measured by the mean square error) between
% the D components.
mse_append_several = mean(err_append_several.^2);





