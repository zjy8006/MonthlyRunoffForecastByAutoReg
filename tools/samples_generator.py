import os
root_path = os.path.dirname(os.path.abspath('__file__'))

import sys
import glob
import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.feature_selection import mutual_info_regression
import deprecated
import logging
sys.path.append(root_path)
from config.globalLog import logger

def PCA_transform(X,y,n_components):
    logger.info('X.shape={}'.format(X.shape))
    logger.info('y.shape={}'.format(y.shape))
    logger.info('X contains Nan:{}'.format(X.isnull().values.any()))
    logger.info("Input features before PAC:\n{}".format(X))
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    pca_X = pca.transform(X)
    columns = []
    for i in range(1, pca_X.shape[1]+1):
        columns.append('X'+str(i))
    pca_X = pd.DataFrame(pca_X, columns=columns)
    logger.info("pca_X.shape={}".format(pca_X.shape))
    print(pca_X)
    print(y)
    pca_samples = pd.concat([pca_X, y], axis=1)
    return pca_samples

def gen_direct_samples(input_df,output_df,lags_dict,lead_time):
    input_columns = list(input_df.columns)
    # Get the number of input features
    subsignals_num = input_df.shape[1]
    # Get the data size
    data_size = input_df.shape[0]
    # Compute the samples size
    max_lag = max(lags_dict.values())
    logger.info('max lag:{}'.format(max_lag))
    samples_size = data_size-max_lag
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags_dict.values())):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    logger.info('Samples columns:{}'.format(samples_cols))
    # Generate input colmuns for each input feature
    samples = pd.DataFrame()
    for i in range(subsignals_num):
        # Get one input feature
        one_in = (input_df[input_columns[i]]).values  # subsignal
        lag = lags_dict[input_columns[i]]
        logger.info('lag:{}'.format(lag))
        oness = pd.DataFrame()  # restor input features
        for j in range(lag):
            x = pd.DataFrame(one_in[j:data_size-(lag-j)], columns=['X' + str(j + 1)])
            x = x.reset_index(drop=True)
            oness = pd.concat([oness, x], axis=1, sort=False)
        logger.info("oness:\n{}".format(oness))
        oness = oness.iloc[oness.shape[0]-samples_size:]
        oness = oness.reset_index(drop=True)
        samples = pd.concat([samples, oness], axis=1, sort=False)
    # Get the target
    target = (output_df.values)[max_lag+lead_time-1:]
    target = pd.DataFrame(target, columns=['Y'])
    # Concat the features and target
    samples = samples[:samples.shape[0]-(lead_time-1)]
    samples = samples.reset_index(drop=True)
    samples = pd.concat([samples, target], axis=1)
    samples = pd.DataFrame(samples.values, columns=samples_cols)
    return samples

def get_last_direct_sample(input_df,output_df,lags_dict,lead_time):
    samples = gen_direct_samples(input_df,output_df,lags_dict,lead_time)
    last_sample = samples.iloc[samples.shape[0]-1:]
    return last_sample

def gen_signal_samples(signal,lag,lead_time):
    if type(signal)==pd.DataFrame or type(signal)==pd.Series or type(signal)==list:
        nparr = np.array(signal)
    # Create an empty pandas Dataframe
    samples = pd.DataFrame()
    # Generate input series based on lag and add these series to full dataset
    for i in range(lag):
        x = pd.DataFrame(nparr[i:signal.shape[0] -(lag - i)], columns=['X' + str(i + 1)])
        x = x.reset_index(drop=True)
        samples = pd.concat([samples, x], axis=1, sort=False)
    # Generate label data
    target = pd.DataFrame(nparr[lag+lead_time-1:], columns=['Y'])
    target = target.reset_index(drop=True)
    samples = samples[:samples.shape[0]-(lead_time-1)]
    samples = samples.reset_index(drop=True)
    # Add labled data to full_data_set
    samples = pd.concat([samples, target], axis=1, sort=False)
    return samples

def get_last_signal_sample(signal,lag,lead_time):
    samples = gen_signal_samples(signal,lag,lead_time)
    last_sample = samples.iloc[samples.shape[0]-1:]
    return last_sample
    

def gen_direct_pcc_samples(input_df,output_df,pre_times,lead_time,lags_dict,mode):
    input_columns = list(input_df.columns)
    pcc_lag = pre_times+lead_time
    pre_cols = []
    for i in range(1, pre_times+1):
        pre_cols.append("X"+str(i))
    logger.info("Previous columns of lagged months:\n{}".format(pre_cols))
    logger.info('PCC mode={}'.format(mode))
    if mode == 'global':
        cols = []
        for i in range(1, pre_times*input_df.shape[1]+1):
            cols.append('X'+str(i))
        cols.append('Y')
        logger.info("columns of lagged months:\n{}".format(cols))
        input_predictors = pd.DataFrame()
        for col in input_columns:
            logger.info("Perform subseries:{}".format(col))
            subsignal = np.array(input_df[col])
            inputs = pd.DataFrame()
            for k in range(pcc_lag):
                x = pd.DataFrame(subsignal[k:subsignal.size-(pcc_lag-k)], columns=["X"+str(k+1)])
                x = x.reset_index(drop=True)
                inputs = pd.concat([inputs, x], axis=1)
            pre_inputs = inputs[pre_cols]
            input_predictors = pd.concat([input_predictors,pre_inputs],axis=1)

        logger.info("Input predictors:\n{}".format(input_predictors.head()))
        target = output_df[pcc_lag:]
        target = target.reset_index(drop=True)
        samples = pd.concat([input_predictors, target], axis=1)
        samples = pd.DataFrame(samples.values,columns=cols)
        logger.info("Inputs and output:\n{}".format(samples.head()))
        corrs = samples.corr(method="pearson")
        logger.info("Entire pearson correlation coefficients:\n{}".format(corrs))
        corrs = (corrs['Y']).iloc[0:corrs.shape[0]-1]
        logger.info("Pearson correlation coefficients:\n{}".format(corrs))
        orig_corrs = abs(corrs.squeeze())
        orig_corrs = orig_corrs.sort_values(ascending=False)
        logger.info("Descending pearson coefficients:\n{}".format(orig_corrs))
        logger.info('Lags_dict.valus={}'.format(list(lags_dict.values())))
        PACF_samples_num = sum(list(lags_dict.values()))
        selected_corrs = orig_corrs[:PACF_samples_num]
        selected_cols = list(selected_corrs.index.values)
        logger.info("Selected columns:\n{}".format(selected_cols))
        selected_cols.append('Y')
        samples = samples[selected_cols]
        logger.info("Selected samples:\n{}".format(samples))
        columns = []
        for i in range(0, samples.shape[1]-1):
            columns.append("X"+str(i+1))
        columns.append("Y")
        samples = pd.DataFrame(samples.values, columns=columns)
        return samples
    elif mode=='local':
        target = output_df[pcc_lag:]
        target = target.reset_index(drop=True)
        input_predictors = pd.DataFrame()
        cols = []
        for i in range(1, pre_times+1):
            cols.append('X'+str(i))
        cols.append('Y')
        for col in input_columns:
            logger.info("Perform subseries:{}".format(col))
            lag = lags_dict[col]
            logger.info('lag={}'.format(lag))
            subsignal = np.array(input_df[col])
            inputs = pd.DataFrame()
            for k in range(pcc_lag):
                x = pd.DataFrame(subsignal[k:subsignal.size-(pcc_lag-k)], columns=["X"+str(k+1)])
                x = x.reset_index(drop=True)
                inputs = pd.concat([inputs, x], axis=1)
            pre_inputs = inputs[pre_cols]
            samples = pd.concat([pre_inputs, target], axis=1)
            samples = pd.DataFrame(samples.values,columns=cols)
            logger.info("Inputs and output:\n{}".format(samples.head()))
            corrs = samples.corr(method="pearson")
            logger.info("Entire pearson correlation coefficients:\n{}".format(corrs))
            corrs = (corrs['Y']).iloc[0:corrs.shape[0]-1]
            logger.info("Pearson correlation coefficients:\n{}".format(corrs))
            orig_corrs = abs(corrs.squeeze())
            orig_corrs = orig_corrs.sort_values(ascending=False)
            logger.info("Descending pearson coefficients:\n{}".format(orig_corrs))
            logger.info('Lags_dict.valus={}'.format(list(lags_dict.values())))
            selected_corrs = orig_corrs[:lag]
            selected_cols = list(selected_corrs.index.values)
            logger.info("Selected columns:\n{}".format(selected_cols))
            input_samples = samples[selected_cols]
            logger.info("Selected samples:\n{}".format(input_samples))
            input_predictors = pd.concat([input_predictors,input_samples],axis=1)
        logger.info("Input predictors:\n{}".format(input_predictors.head()))
        samples = pd.concat([input_predictors,target],axis=1)
        columns = []
        for i in range(0, samples.shape[1]-1):
            columns.append("X"+str(i+1))
        columns.append("Y")
        samples = pd.DataFrame(samples.values, columns=columns)
        return samples


def get_last_direct_pcc_sample(input_df,output_df,pre_times,lead_time,lags_dict,mode):
    samples = gen_direct_pcc_samples(input_df,output_df,pre_times,lead_time,lags_dict,mode)
    last_sample = samples[samples.shape[0]-1:]
    return last_sample

def dump_direct_samples(save_path,train_samples,dev_samples,test_samples):
    train_samples.to_csv(save_path+'train_samples.csv',index=None)
    dev_samples.to_csv(save_path+'dev_samples.csv',index=None)
    test_samples.to_csv(save_path+'test_samples.csv',index=None)
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    dev_samples = 2 * (dev_samples - series_min) / (series_max - series_min) - 1
    test_samples = 2*(test_samples-series_min)/(series_max-series_min)-1
    logger.info('Save path:{}'.format(save_path))
    logger.info('The size of training samples:{}'.format(train_samples.shape[0]))
    logger.info('The size of development samples:{}'.format(dev_samples.shape[0]))
    logger.info('The size of testing samples:{}'.format(test_samples.shape[0]))
    series_max = pd.DataFrame(series_max, columns=['series_max'])
    series_min = pd.DataFrame(series_min, columns=['series_min'])
    normalize_indicators = pd.concat([series_max, series_min], axis=1)
    normalize_indicators.to_csv(save_path+"norm_unsample_id.csv")
    train_samples.to_csv(save_path+'minmax_unsample_train.csv', index=None)
    dev_samples.to_csv(save_path+'minmax_unsample_dev.csv', index=None)
    test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)

def dump_multi_samples(save_path,signal_id,train_samples,dev_samples,test_samples):
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    dev_samples = 2 * (dev_samples - series_min) / (series_max - series_min) - 1
    test_samples = 2*(test_samples-series_min)/(series_max-series_min)-1
    series_max = pd.DataFrame(series_max, columns=['series_max'])
    series_min = pd.DataFrame(series_min, columns=['series_min'])
    normalize_indicators = pd.concat([series_max, series_min], axis=1)
    normalize_indicators.to_csv(save_path+'norm_unsample_id_s'+str(signal_id)+'.csv')
    train_samples.to_csv(save_path+'minmax_unsample_train_s'+str(signal_id)+'.csv', index=None)
    dev_samples.to_csv(save_path+'minmax_unsample_dev_s'+str(signal_id)+'.csv', index=None)
    test_samples.to_csv(save_path+'minmax_unsample_test_s'+str(signal_id)+'.csv', index=None)


def gen_wddff_samples(data_path,lead_time,lag,use_full,dev_len=120,mode='MI',threshold=0.1,n_components=None):
    full_samples = pd.read_csv(data_path+str(lead_time)+'_ahead_lag'+str(lag)+'_fullsamples.csv')
    cal_samples = pd.read_csv(data_path+str(lead_time)+'_ahead_lag'+str(lag)+'_calibration.csv')
    val_samples = pd.read_csv(data_path+str(lead_time)+'_ahead_lag'+str(lag)+'_validation.csv')
    if mode ==None:
        save_path = data_path+'single_hybrid_'+str(lead_time)+'_ahead_lag'+str(lag)+'/'
        if use_full:
            cal_samples = full_samples[:full_samples.shape[0]-240]
            val_samples = full_samples[full_samples.shape[0]-240:]
    elif mode == 'MI':
        if use_full:
            y = full_samples['Y']
            X = full_samples.drop('Y',axis=1) 
            mi = mutual_info_regression(X,y)
            mi_df = pd.DataFrame(mi,index=X.columns,columns=['MI'])
            bools = mi_df['MI']>=threshold
            # print(bools)
            select = list((mi_df.loc[bools == True]).index.values)
            # print(select)
            select.append('Y')
            # print(select)
            full_samples = full_samples[select]
            cal_samples = full_samples[:full_samples.shape[0]-240]
            val_samples = full_samples[full_samples.shape[0]-240:]
        else:
            y = cal_samples['Y']
            X = cal_samples.drop('Y',axis=1) 
            mi = mutual_info_regression(X,y)
            mi_df = pd.DataFrame(mi,index=X.columns,columns=['MI'])
            bools = mi_df['MI']>=threshold
            # print(bools)
            select = list((mi_df.loc[bools == True]).index.values)
            # print(select)
            select.append('Y')
            # print(select)
            cal_samples = cal_samples[select]
            val_samples = val_samples[select]
        save_path = data_path+'single_hybrid_'+str(lead_time)+'_ahead_lag'+str(lag)+'_mi_ts'+str(threshold)+'/'
    elif mode == 'PCA':
        if use_full:
            samples = full_samples
            samples = samples.reset_index(drop=True)
            samples_y = samples['Y']
            samples_X = samples.drop('Y', axis=1)
            pca_samples = PCA_transform(X=samples_X,y=samples_y,n_components=n_component)
            cal_samples = pca_samples.iloc[:pca_samples.shape[0]-240]
            cal_samples = cal_samples.reset_index(drop=True)
            logger.info('Calibration samples after PCA:\n{}'.format(cal_samples))
            val_samples = pca_samples.iloc[pca_samples.shape[0]-240:]
            val_samples = val_samples.reset_index(drop=True)
            logger.info('Validation samples after PCA:\n{}'.format(val_samples))
        else:
            samples = pd.concat([cal_samples,val_samples],axis=0)
            samples = samples.reset_index(drop=True)
            samples_y = samples['Y']
            samples_X = samples.drop('Y', axis=1)
            pca_samples = PCA_transform(X=samples_X,y=samples_y,n_components=n_components)
            cal_samples = pca_samples.iloc[:cal_samples.shape[0]]
            cal_samples = cal_samples.reset_index(drop=True)
            logger.info('Calibration samples after PCA:\n{}'.format(cal_samples))
            val_samples = pca_samples.iloc[cal_samples.shape[0]:]
            val_samples = val_samples.reset_index(drop=True)
            logger.info('Validation samples after PCA:\n{}'.format(val_samples))
        save_path = data_path+'single_hybrid_'+str(lead_time)+'_ahead_lag'+str(lag)+'_pca'+str(n_components)+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cal_samples.to_csv(save_path+'cal_samples.csv', index=None)
    

    series_min = cal_samples.min(axis=0)
    series_max = cal_samples.max(axis=0)
    train_samples = 2 * (cal_samples - series_min) / (series_max - series_min) - 1
    val_samples = 2 * (val_samples - series_min) / (series_max - series_min) - 1
    dev_samples = val_samples[:val_samples.shape[0]-dev_len]
    test_samples = val_samples[val_samples.shape[0]-dev_len:]
    train_samples = train_samples.reset_index(drop=True)
    dev_samples = dev_samples.reset_index(drop=True)
    test_samples = test_samples.reset_index(drop=True)
    series_max = pd.DataFrame(series_max, columns=['series_max'])
    series_min = pd.DataFrame(series_min, columns=['series_min'])
    normalize_indicators = pd.concat([series_max, series_min], axis=1)
        

    normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
    train_samples.to_csv(save_path+'minmax_unsample_train.csv', index=None)
    dev_samples.to_csv(save_path+'minmax_unsample_dev.csv', index=None)
    test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)

def generate_monoscale_samples(source_file, save_path, lags_dict, column, test_len, lead_time=1,regen=False):
    """Generate learning samples for autoregression problem using original time series. 
    Args:
    'source_file' -- ['String'] The source data file path.
    'save_path' --['String'] The path to restore the training, development and testing samples.
    'lags_dict' -- ['int dict'] The lagged time for original time series.
    'column' -- ['String']The column's name for read the source data by pandas.
    'test_len' --['int'] The length of development and testing set.
    'lead_time' --['int'] The lead time.
    """
    logger.info('Generating muliti-step decomposition-ensemble hindcasting samples')
    save_path = save_path+'/'+str(lead_time)+'_ahead_pacf_lag'+str(lags_dict['ORIG'])+'/'
    logger.info('Source file:{}'.format(source_file))
    logger.info('Save path:{}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
        #  Load data from local dick
        if '.xlsx' in source_file:
            dataframe = pd.read_excel(source_file)[column]
        elif '.csv' in source_file:
            dataframe = pd.read_csv(source_file)[column]
        lag = lags_dict['ORIG']
        full_samples=gen_signal_samples(signal=dataframe,lag=lag,lead_time=lead_time)
        full_samples.to_csv(save_path+'full_samples.csv',index=None)
        # Get the length of this series
        series_len = full_samples.shape[0]
        # Get the training and developing set
        train_dev_samples = full_samples[0:(series_len - test_len)]
        # Get the testing set.
        test_samples = full_samples[(series_len - test_len):series_len]
        # train_dev_len = train_dev_samples.shape[0]
        train_samples = full_samples[0:(series_len - test_len - test_len)]
        dev_samples = full_samples[(series_len - test_len - test_len):(series_len - test_len)]
        assert (train_samples.shape[0] + dev_samples.shape[0] +test_samples.shape[0]) == series_len
        dump_direct_samples(save_path,train_samples,dev_samples,test_samples)



def gen_direct_hindcast_samples(
    station, decomposer, lags_dict, input_columns, output_column, test_len,
    wavelet_level="db10-2", mode='PACF', pcc_mode='local', lead_time=1,pre_times=20,n_components=None,regen=False,
    ):
    """ 
    Generate one step hindcast decomposition-ensemble learning samples. 
    Args:
    'station'-- ['string'] The station where the original time series come from.
    'decomposer'-- ['string'] The decompositin algorithm used for decomposing the original time series.
    'lags_dict'-- ['int dict'] The lagged time for each subsignal.
    'input_columns'-- ['string list'] The input columns' name used for generating the learning samples.
    'output_columns'-- ['string'] The output column's name used for generating the learning samples.
    'test_len'-- ['int'] The size of development and testing samples ().
    """
    if decomposer=='dwt' or decomposer=='modwt':
        lags_dict=lags_dict[wavelet_level]
    logger.info('Generating one-step decomposition ensemble hindcasting samples')
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Lags_dict:{}'.format(lags_dict))
    logger.info('Input columns:{}'.format(input_columns))
    logger.info('Output column:{}'.format(output_column))
    logger.info('Testing sample length:{}'.format(test_len))
    logger.info('Mother wavelet and decomposition level:{}'.format(wavelet_level))
    logger.info('Lead time:{}'.format(lead_time))
    #  Load data from local dick
    if decomposer == 'modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data-tsdp/"+wavelet_level+"/"
    elif decomposer == "dwt":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    if mode == 'PACF' and n_components == None:
        save_path = data_path+"one_step_" + str(lead_time)+"_ahead_hindcast_pacf/"
    elif mode == 'PACF' and n_components != None:
        save_path = data_path+"one_step_" + str(lead_time)+"_ahead_hindcast_pacf_pca"+str(n_components)+"/"
    elif mode == 'Pearson' and n_components == None:
        save_path = data_path+"one_step_" + str(lead_time)+"_ahead_hindcast_pcc/"
    elif mode == 'Pearson' and n_components != None:
        save_path = data_path+"one_step_" + str(lead_time)+"_ahead_hindcast_pcc_pca"+str(n_components)+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
        if mode == 'PACF':
            decompose_file = data_path+decomposer.upper()+"_FULL.csv"
            decompositions = pd.read_csv(decompose_file)
            # Drop NaN
            decompositions.dropna()
            # Get the input data (the decompositions)
            input_data = decompositions[input_columns]
            # Get the output data (the original time series)
            output_data = decompositions[output_column]
            full_samples=gen_direct_samples(input_df=input_data,output_df=output_data,lags_dict=lags_dict,lead_time=lead_time)
            full_samples.to_csv(save_path+'full_samples.csv',index=None)
            # Get the training and developing set
            train_dev_samples = full_samples[:(full_samples.shape[0] - test_len)]
            # Get the testing set.
            test_samples = full_samples[(full_samples.shape[0] - test_len):]
            # train_dev_len = train_dev_samples.shape[0]
            train_samples = full_samples[0:(full_samples.shape[0] - test_len - test_len)]
            dev_samples = full_samples[(full_samples.shape[0] - test_len - test_len):(full_samples.shape[0] - test_len)]
            if n_components != None:
                logger.info('Performa PCA on samples based on PACF')
                samples = pd.concat([train_samples, dev_samples, test_samples], axis=0, sort=False)
                samples = samples.reset_index(drop=True)
                y = samples['Y']
                X = samples.drop('Y', axis=1)
                logger.info('X contains Nan:{}'.format(X.isnull().values.any()))
                logger.info("Input features before PAC:\n{}".format(X))
                pca_samples = PCA_transform(X=X,y=y,n_components=n_components)
                train_samples = pca_samples.iloc[:train_samples.shape[0]]
                train_samples = train_samples.reset_index(drop=True)
                logger.info('Training samples after PCA:\n{}'.format(train_samples))
                dev_samples = pca_samples.iloc[train_samples.shape[0]:train_samples.shape[0]+dev_samples.shape[0]]
                dev_samples = dev_samples.reset_index(drop=True)
                logger.info('Development samples after PCA:\n{}'.format(dev_samples))
                test_samples = pca_samples.iloc[train_samples.shape[0] +dev_samples.shape[0]:]
                test_samples = test_samples.reset_index(drop=True)
                logger.info('Testing samples after PCA:\n{}'.format(test_samples))
            dump_direct_samples(save_path,train_samples,dev_samples,test_samples)            
        elif mode == 'Pearson':
            decompose_file = data_path+decomposer.upper()+"_FULL.csv"
            decompositions = pd.read_csv(decompose_file)
            outputs_df = decompositions[output_column]
            inputs_df = decompositions.drop(output_column,axis=1)
            samples=gen_direct_pcc_samples(input_df=inputs_df,output_df=outputs_df,pre_times=pre_times,lead_time=lead_time,lags_dict=lags_dict,mode=pcc_mode)
            train_samples = samples.iloc[:samples.shape[0]-test_len-test_len]
            train_samples = train_samples.reset_index(drop=True)
            dev_samples = samples.iloc[samples.shape[0]-test_len-test_len:samples.shape[0]-test_len]
            dev_samples = dev_samples.reset_index(drop=True)
            test_samples = samples.iloc[samples.shape[0]-test_len:]
            test_samples = test_samples.reset_index(drop=True)
            # Perform PCA on samples based on Pearson
            if n_components != None:
                logger.info('Performa PCA on samples based on PACF')
                samples = pd.concat([train_samples, dev_samples, test_samples], axis=0, sort=False)
                samples = samples.reset_index(drop=True)
                y = samples['Y']
                X = samples.drop('Y', axis=1)
                pca_samples = PCA_transform(X=X,y=y,n_components=n_components)
                train_samples = pca_samples.iloc[:train_samples.shape[0]]
                train_samples = train_samples.reset_index(drop=True)
                dev_samples = pca_samples.iloc[train_samples.shape[0]:train_samples.shape[0]+dev_samples.shape[0]]
                dev_samples = dev_samples.reset_index(drop=True)
                test_samples = pca_samples.iloc[train_samples.shape[0] +dev_samples.shape[0]:]
                test_samples = test_samples.reset_index(drop=True)
            dump_direct_samples(save_path,train_samples,dev_samples,test_samples)

        


def gen_direct_forecast_samples(
    station, decomposer, lags_dict, input_columns, output_column, start, stop, test_len,
    gen_from,pcc_mode='local',
    wavelet_level="db10-2", lead_time=1,mode='PACF', pre_times=20,n_components=None,regen=False):
    """ 
    Generate one step forecast decomposition-ensemble samples. 
    Args:
    'station'-- ['string'] The station where the original time series come from.
    'decomposer'-- ['string'] The decompositin algorithm used for decomposing the original time series.
    'lags_dict'-- ['int dict'] The lagged time for subsignals.
    'input_columns'-- ['string lsit'] the input columns' name for read the source data by pandas.
    'output_columns'-- ['string'] the output column's name for read the source data by pandas.
    'start'-- ['int'] The start index of appended decomposition file.
    'stop'-- ['int'] The stop index of appended decomposotion file.
    'test_len'-- ['int'] The size of development and testing samples.
    """
    if decomposer=='dwt' or decomposer=='modwt':
        lags_dict=lags_dict[wavelet_level]
    logger.info('Generateing one-step decomposition ensemble forecasting samples (traindev-test pattern)')
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Lags_dict:{}'.format(lags_dict))
    logger.info('Input columns:{}'.format(input_columns))
    logger.info('Output column:{}'.format(output_column))
    logger.info('Validation start index:{}'.format(start))
    logger.info('Validation stop index:{}'.format(stop))
    logger.info('Testing sample length:{}'.format(test_len))
    logger.info('Mother wavelet and decomposition level:{}'.format(wavelet_level))
    logger.info('Lead time:{}'.format(lead_time))
    #  Load data from local dick
    if decomposer == 'modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data-tsdp/"+wavelet_level+"/"
    if decomposer == "dwt":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"


    if gen_from=='training and appended sets':
        if mode == 'PACF' and n_components == None:
            save_path = data_path+"one_step_" + str(lead_time)+"_ahead_forecast_pacf/"
        elif mode == 'PACF' and n_components != None:
            save_path = data_path+"one_step_" + str(lead_time)+"_ahead_forecast_pacf_pca"+str(n_components)+"/"
        elif mode == 'Pearson' and n_components == None:
            save_path = data_path+"one_step_" + str(lead_time)+"_ahead_forecast_pcc_"+pcc_mode+"/"
        elif mode == 'Pearson' and n_components != None:
            save_path = data_path+"one_step_" + str(lead_time)+"_ahead_forecast_pcc_"+pcc_mode+"_pca"+str(n_components)+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if len(os.listdir(save_path))>0 and not regen:
            logger.info('Learning samples have been generated!')
        else:
            if mode == 'PACF':
                train_dec = pd.read_csv(data_path+decomposer.upper()+"_TRAIN.csv")
                # Drop NaN
                train_dec.dropna()
                # Get the input data (the decompositions)
                train_in = train_dec[input_columns]
                # Get the output data (the original time series)
                train_out = train_dec[output_column]
                train_samples = gen_direct_samples(input_df=train_in,output_df=train_out,lags_dict=lags_dict,lead_time=lead_time)
                train_samples.to_csv(save_path+'train_samples.csv',index=None)
                dev_test_samples = pd.DataFrame()
                appended_file_path = data_path+decomposer+"-test/"
                for k in range(start, stop+1):
                    #  Load data from local dick
                    append_dec = pd.read_csv(appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')
                    # Drop NaN
                    append_dec.dropna()
                    # Get the input data (the decompositions)
                    input_data = append_dec[input_columns]
                    # Get the output data (the original time series)
                    output_data = append_dec[output_column]
                    last_appended_sample = get_last_direct_sample(input_df=input_data,output_df=output_data,lags_dict=lags_dict,lead_time=lead_time)
                    dev_test_samples = pd.concat([dev_test_samples, last_appended_sample], axis=0)
                dev_test_samples = dev_test_samples.reset_index(drop=True)
                dev_test_samples.to_csv(save_path+'dev_test_samples.csv',index=None)
                dev_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len-test_len: dev_test_samples.shape[0]-test_len]
                test_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]

                if n_components != None:
                    logger.info('Performa PCA on samples based on PACF')
                    samples = pd.concat([train_samples, dev_samples, test_samples], axis=0, sort=False)
                    samples = samples.reset_index(drop=True)
                    y = samples['Y']
                    X = samples.drop('Y', axis=1)
                    logger.info('X contains Nan:{}'.format(X.isnull().values.any()))
                    logger.info("Input features before PAC:\n{}".format(X))
                    pca_samples = PCA_transform(X=X,y=y,n_components=n_components)
                    train_samples = pca_samples.iloc[:train_samples.shape[0]]
                    train_samples = train_samples.reset_index(drop=True)
                    logger.info('Training samples after PCA:\n{}'.format(train_samples))
                    dev_samples = pca_samples.iloc[train_samples.shape[0]:train_samples.shape[0]+dev_samples.shape[0]]
                    dev_samples = dev_samples.reset_index(drop=True)
                    logger.info('Development samples after PCA:\n{}'.format(dev_samples))
                    test_samples = pca_samples.iloc[train_samples.shape[0] +dev_samples.shape[0]:]
                    test_samples = test_samples.reset_index(drop=True)
                    logger.info('Testing samples after PCA:\n{}'.format(test_samples))
                dump_direct_samples(save_path,train_samples,dev_samples,test_samples)
            elif mode == 'Pearson':
                train_dec = pd.read_csv(data_path+decomposer.upper()+"_TRAIN.csv")
                train_out = train_dec[output_column]
                train_in = train_dec.drop(output_column,axis=1)
                train_samples = gen_direct_pcc_samples(input_df=train_in,output_df=train_out,pre_times=pre_times,lead_time=lead_time,lags_dict=lags_dict,mode=pcc_mode)
                train_samples.to_csv(save_path+'/train_samples.csv',index=None)
                dev_test_samples = pd.DataFrame()
                for i in range(start, stop+1):
                    append_decompositions = pd.read_csv(data_path+decomposer+"-test/"+decomposer+"_appended_test"+str(i)+".csv")
                    append_out = append_decompositions[output_column]
                    append_in = append_decompositions.drop(output_column,axis=1)
                    last_append_sample = get_last_direct_pcc_sample(input_df=append_in,output_df=append_out,pre_times=pre_times,lead_time=lead_time,lags_dict=lags_dict,mode=pcc_mode)
                    dev_test_samples = pd.concat([dev_test_samples, last_append_sample], axis=0)
                dev_test_samples = dev_test_samples.reset_index(drop=True)
                dev_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len-test_len:dev_test_samples.shape[0]-test_len]
                test_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]
                dev_samples = dev_samples.reset_index(drop=True)
                test_samples = test_samples.reset_index(drop=True)

                # Perform PCA on samples based on Pearson
                if n_components != None:
                    logger.info('Performa PCA on samples based on PACF')
                    samples = pd.concat([train_samples, dev_samples, test_samples], axis=0, sort=False)
                    samples = samples.reset_index(drop=True)
                    y = samples['Y']
                    X = samples.drop('Y', axis=1)
                    pca_samples = PCA_transform(X=X,y=y,n_components=n_components)
                    train_samples = pca_samples.iloc[:train_samples.shape[0]]
                    train_samples = train_samples.reset_index(drop=True)
                    dev_samples = pca_samples.iloc[train_samples.shape[0]:train_samples.shape[0]+dev_samples.shape[0]]
                    dev_samples = dev_samples.reset_index(drop=True)
                    test_samples = pca_samples.iloc[train_samples.shape[0] +dev_samples.shape[0]:]
                    test_samples = test_samples.reset_index(drop=True)
                dump_direct_samples(save_path,train_samples,dev_samples,test_samples)
    elif gen_from == 'training and validation sets':
        save_path = data_path+"one_step_" + str(lead_time)+"_ahead_forecast_pacf_train_val/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if len(os.listdir(save_path))>0 and not regen:
            logger.info('Learning samples have been generated!')
        else:
            train_dec = pd.read_csv(data_path+decomposer.upper()+"_TRAIN.csv")
            # Get the input data (the decompositions)
            train_in = train_dec[input_columns]
            # Get the output data (the original time series)
            train_out = train_dec[output_column]
            train_samples = gen_direct_samples(input_df=train_in,output_df=train_out,lags_dict=lags_dict,lead_time=lead_time)
            train_samples.to_csv(save_path+'train_samples.csv',index=None)
            val_dec = pd.DataFrame()
            appended_file_path = data_path+decomposer+"-test/"
            for k in range(start, stop+1):
                #  Load data from local dick
                append_dec = pd.read_csv(appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')
                last_dec = append_dec[append_dec.shape[0]-1:]
                val_dec = pd.concat([val_dec,last_dec],axis=0)
            val_dec = val_dec.reset_index(drop=True)
            logger.info('Validation decompositions:\n{}'.format(val_dec))
            val_in = val_dec[input_columns]
            val_out = val_dec[output_column]
            val_samples = gen_direct_samples(input_df=val_in,output_df=val_out,lags_dict=lags_dict,lead_time=lead_time)
            val_samples.to_csv(save_path+'dev_test_samples.csv',index=None)
            dev_samples = val_samples[val_samples.shape[0]-test_len-test_len:val_samples.shape[0]-test_len]
            test_samples = val_samples[val_samples.shape[0]-test_len:]
            dev_samples = dev_samples.reset_index(drop=True)
            test_samples = test_samples.reset_index(drop=True)
            dump_direct_samples(save_path,train_samples,dev_samples,test_samples)
    elif gen_from == 'training-development and test sets':
        save_path = data_path+"one_step_" + str(lead_time)+"_ahead_forecast_pacf_traindev_test/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if len(os.listdir(save_path))>0 and not regen:
            logger.info('Learning samples have been generated!')
        else:
            traindev_dec = pd.read_csv(data_path+decomposer.upper()+"_TRAINDEV.csv")
            traindev_dec.dropna()
            # Get the input data (the decompositions)
            traindev_in = traindev_dec[input_columns]
            # Get the output data (the original time series)
            traindev_out = traindev_dec[output_column]
            train_dev_samples = gen_direct_samples(input_df=traindev_in,output_df=traindev_out,lags_dict=lags_dict,lead_time=lead_time)
            train_dev_samples.to_csv(save_path+'train_dev_samples.csv',index=None)
            train_samples = train_dev_samples[:train_dev_samples.shape[0]-test_len]
            dev_samples = train_dev_samples[train_dev_samples.shape[0]-test_len:]
            test_samples = pd.DataFrame()
            val_dec = pd.DataFrame()
            appended_file_path = data_path+decomposer+"-test/"
            for k in range(start, stop+1):
                #  Load data from local dick
                append_dec = pd.read_csv(appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')
                last_dec = append_dec[append_dec.shape[0]-1:]
                val_dec = pd.concat([val_dec,last_dec],axis=0)
            val_dec = val_dec.reset_index(drop=True)
            logger.info('Validation decompositions:\n{}'.format(val_dec))
            val_in = val_dec[input_columns]
            val_out = val_dec[output_column]
            val_samples = gen_direct_samples(input_df=val_in,output_df=val_out,lags_dict=lags_dict,lead_time=lead_time)
            test_samples = val_samples[val_samples.shape[0]-test_len:]
            test_samples = test_samples.reset_index(drop=True)
            dump_direct_samples(save_path,train_samples,dev_samples,test_samples)
    if gen_from == 'training-development and appended sets':
        save_path = data_path+"one_step_" + str(lead_time)+"_ahead_forecast_pacf_traindev_append/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if len(os.listdir(save_path))>0 and not regen:
            logger.info('Learning samples have been generated!')
        else:
            traindev_dec = pd.read_csv(data_path+decomposer.upper()+"_TRAINDEV.csv")
            traindev_dec.dropna()
            # Get the input data (the decompositions)
            traindev_in = traindev_dec[input_columns]
            # Get the output data (the original time series)
            traindev_out = traindev_dec[output_column]
            train_dev_samples = gen_direct_samples(input_df=traindev_in,output_df=traindev_out,lags_dict=lags_dict,lead_time=lead_time)
            train_dev_samples.to_csv(save_path+'train_dev_samples.csv',index=None)
            train_samples = train_dev_samples[:train_dev_samples.shape[0]-test_len]
            dev_samples = train_dev_samples[train_dev_samples.shape[0]-test_len:]
            test_samples = pd.DataFrame()
            appended_file_path = data_path+decomposer+"-test/"
            for k in range(start, stop+1):
                #  Load data from local dick
                append_dec = pd.read_csv(appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')
                # Drop NaN
                append_dec.dropna()
                # Get the input data (the decompositions)
                input_data = append_dec[input_columns]
                # Get the output data (the original time series)
                output_data = append_dec[output_column]
                last_appended_samples = get_last_direct_sample(input_df=input_data,output_df=output_data,lags_dict=lags_dict,lead_time=lead_time)
                test_samples = pd.concat([test_samples, last_appended_samples], axis=0)
            test_samples = test_samples[test_samples.shape[0]-test_len:]
            test_samples = test_samples.reset_index(drop=True)
            dump_direct_samples(save_path,train_samples,dev_samples,test_samples)
    

def gen_multi_hindcast_samples(
    station, decomposer, lags_dict, columns, test_len,
    wavelet_level="db10-2", lead_time=1,regen=False):
    """ 
    Generate muliti-step learning samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -station: The station where the original time series observed.
        -decomposer: The decomposition algorithm for decomposing the original time series.
        -lags_dict: The lags for autoregression.
        -columns: the columns' name for read the source data by pandas.
        -save_path: The path to restore the training, development and testing samples.
        -test_len: The length of validation(development or testing) set.
    """
    if decomposer=='dwt' or decomposer=='modwt':
        lags_dict=lags_dict[wavelet_level]
    logger.info(
        "Generating muliti-step decompositionensemble hindcasting samples")
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Lags_dict:{}'.format(lags_dict))
    logger.info('Signals:{}'.format(columns))
    logger.info('Testing sample length:{}'.format(test_len))
    logger.info(
        'Mother wavelet and decomposition level:{}'.format(wavelet_level))
    logger.info('Lead time:{}'.format(lead_time))
    if decomposer == "dwt" or decomposer == 'modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"multi_step_"+str(lead_time)+"_ahead_hindcast_pacf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
        decompose_file = data_path+decomposer.upper()+"_FULL.csv"
        decompositions = pd.read_csv(decompose_file)

        for k in range(len(columns)):
            lag = lags_dict[columns[k]]
            if lag == 0:
                logger.info("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
                continue
            # Obtain decomposed sub-signal
            sub_signal = decompositions[columns[k]]
            full_samples = gen_signal_samples(signal=sub_signal,lag=lag,lead_time=lead_time)
            # Get the length of this series
            series_len = full_samples.shape[0]
            # Get the training and developing set
            train_dev_samples = full_samples[0:(series_len - test_len)]
            # Get the testing set.
            test_samples = full_samples[(series_len - test_len):series_len]
            # Do sampling if 'sampling' is True
            train_samples = full_samples[0:(series_len - test_len - test_len)]
            dev_samples = full_samples[(series_len - test_len - test_len):(series_len - test_len)]
            # Get the max and min value of each series
            series_max = train_samples.max(axis=0)
            series_min = train_samples.min(axis=0)
            # Normalize each series to the range between -1 and 1
            train_samples = 2 * (train_samples - series_min)/(series_max - series_min) - 1
            dev_samples = 2 * (dev_samples - series_min)/(series_max - series_min) - 1
            test_samples = 2 * (test_samples - series_min)/(series_max - series_min) - 1

            logger.info('Series length:{}'.format(series_len))
            logger.info('Save path:{}'.format(save_path))
            logger.info('The size of training and development samples:{}'.format(
                train_dev_samples.shape[0]))
            logger.info('The size of training samples:{}'.format(
                train_samples.shape[0]))
            logger.info('The size of development samples:{}'.format(
                dev_samples.shape[0]))
            logger.info('The size of testing samples:{}'.format(
                test_samples.shape[0]))

            series_max = pd.DataFrame(series_max, columns=['series_max'])
            series_min = pd.DataFrame(series_min, columns=['series_min'])
            normalize_indicators = pd.concat([series_max, series_min], axis=1)
            normalize_indicators.to_csv(save_path+'norm_unsample_id_s'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_s'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev_s'+str(k+1)+'.csv', index=None)
            test_samples.to_csv(save_path+'minmax_unsample_test_s'+str(k+1)+'.csv', index=None)


def gen_multi_forecast_samples(
    station, decomposer, lags_dict, columns, start, stop, test_len, 
    wavelet_level="db10-2", lead_time=1,regen=False):
    """ 
    Generate multi-step training samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -station: The station where the original time series observed.
        -decomposer: The decomposition algorithm for decomposing the original time series.
        -lags_dict: The lags for autoregression.
        -columns: the columns name for read the source data by pandas
        -save_path: The path to save the training samples
    """
    if decomposer=='dwt' or decomposer=='modwt':
        lags_dict=lags_dict[wavelet_level]
    logger.info(
        "Generating muliti-step decompositionensemble forecasting samples")
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Lags_dict:{}'.format(lags_dict))
    logger.info('Signals:{}'.format(columns))
    logger.info('Validation start index:{}'.format(start))
    logger.info('Validation stop index:{}'.format(stop))
    logger.info('Testing sample length:{}'.format(test_len))
    logger.info(
        'Mother wavelet and decomposition level:{}'.format(wavelet_level))
    logger.info('Lead time:{}'.format(lead_time))
    if decomposer == "dwt" or decomposer == 'modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"multi_step_"+str(lead_time)+"_ahead_forecast_pacf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
        logger.info("Save path:{}".format(save_path))
        # !!!!!!!!!!Generate training samples
        train_dec = pd.read_csv(data_path+decomposer.upper()+"_TRAIN.csv")
        train_dec.dropna()
        for k in range(len(columns)):
            lag = lags_dict[columns[k]]
            if lag == 0:
                logger.info("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
                continue
            # Generate sample columns
            samples_columns = []
            for l in range(1, lag+1):
                samples_columns.append('X'+str(l))
            samples_columns.append('Y')
            # Obtain decomposed sub-signal
            sub_signal = train_dec[columns[k]]
            train_samples = gen_signal_samples(signal=sub_signal,lag=lag,lead_time=lead_time)
            # Do sampling if 'sampling' is True
            # Get the max and min value of each series
            series_max = train_samples.max(axis=0)
            series_min = train_samples.min(axis=0)
            # Normalize each series to the range between -1 and 1
            train_samples = 2 * (train_samples - series_min)/(series_max - series_min) - 1
    
            # !!!!!Generate development and testing samples
            dev_test_samples = pd.DataFrame()
            appended_file_path = data_path+decomposer+"-test/"
            for j in range(start, stop+1):  # 遍历每一个附加分解结果
                append_decompositions = pd.read_csv(appended_file_path+decomposer +'_appended_test'+str(j)+'.csv')
                sub_signal = append_decompositions[columns[k]]
                last_append_imf = get_last_signal_sample(signal=sub_signal,lag=lag,lead_time=lead_time)
                dev_test_samples = pd.concat([dev_test_samples, last_append_imf], axis=0)
            dev_test_samples = dev_test_samples.reset_index(drop=True)
            dev_test_samples = 2*(dev_test_samples-series_min)/(series_max-series_min)-1
            dev_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len-test_len:dev_test_samples.shape[0]-test_len]
            test_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]
            dev_samples = dev_samples.reset_index(drop=True)
            test_samples = test_samples.reset_index(drop=True)
    
            series_max = pd.DataFrame(series_max, columns=['series_max'])
            series_min = pd.DataFrame(series_min, columns=['series_min'])
            normalize_indicators = pd.concat([series_max, series_min], axis=1)
            normalize_indicators.to_csv(save_path+'norm_unsample_id_s'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_s'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev_s'+str(k+1)+'.csv', index=None)
            test_samples.to_csv(save_path+'minmax_unsample_test_s'+str(k+1)+'.csv', index=None)
    