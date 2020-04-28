import pandas as pd
import numpy as np
import math
from statistics import mean
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.metrics_ import PPTS,mean_absolute_percentage_error
from config.globalLog import logger

logger.info('results_reader')

def read_two_stage(station,decomposer,predict_pattern,wavelet_level="db10-2",framework='WDDFF'):
    if decomposer=='modwt':
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr-"+framework.lower()+"\\"+wavelet_level+"\\"+predict_pattern+"\\"
    elif decomposer=="dwt":
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\"+predict_pattern+"\\"
    else:
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+predict_pattern+"\\"
    results = pd.read_csv(model_path+'optimal_model_results.csv')
    test_pred = (results['test_pred'][0:120]).values.flatten()
    test_y=(results['test_y'][0:120]).values.flatten()
    time_cost = results['time_cost'][0]
    test_nse = results['test_nse'][0]
    test_mse = results['test_mse'][0]
    test_nrmse = results['test_nrmse'][0]
    test_ppts = results['test_ppts'][0]
    results={
        'test_y':test_y,
        'test_pred':test_pred,
        'test_nse':test_nse,
        'test_nrmse':test_nrmse,
        'test_ppts':test_ppts,
        'time_cost':time_cost,
    }
    return results

def read_two_stage_train_devtest(station,decomposer,predict_pattern,wavelet_level="db10-2",framework='WDDFF'):
    if decomposer=='modwt':
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr-"+framework.lower()+"\\"+wavelet_level+"\\"+predict_pattern+"\\"
    elif decomposer=="dwt":
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\"+predict_pattern+"\\"
    else:
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+predict_pattern+"\\"
    
    results = pd.read_csv(model_path+'optimal_model_results.csv')
    dev_nse = results['dev_nse'][0]
    dev_nrmse = results['dev_nrmse'][0]
    dev_ppts = results['dev_ppts'][0]
    test_nse = results['test_nse'][0]
    test_nrmse = results['test_nrmse'][0]
    test_ppts = results['test_ppts'][0]
    time_cost = results['time_cost'][0]
    dev_y = (results['dev_y'][0:120]).values.flatten()
    dev_predss=(results['dev_pred'][0:120]).values.flatten()
    test_y = (results['test_y'][0:120]).values.flatten()
    test_predss = (results['test_pred'][0:120]).values.flatten()
    results={
        "dev_y":dev_y,
        "dev_pred":dev_predss,
        "dev_nse":dev_nse,
        "dev_nrmse":dev_nrmse,
        "dev_ppts":dev_ppts,
        "test_y":test_y,
        "test_pred":test_predss,
        "test_nse":test_nse,
        "test_nrmse":test_nrmse,
        "test_ppts":test_ppts,
        "time_cost":time_cost,
    }
    return results



def read_pure_esvr(station,predict_pattern='1_ahead_pacf'):
    model_path = root_path+"\\"+station+"\\projects\\esvr\\"+predict_pattern+"\\"
    results = pd.read_csv(model_path+'optimal_model_results.csv')
    test_y = (results['test_y'][0:120]).values.flatten()
    test_pred = (results['test_pred'][0:120]).values.flatten()
    test_nse = results['test_nse'][0]
    test_nrmse = results['test_nrmse'][0]
    test_ppts = results['test_ppts'][0]
    time_cost = results['time_cost'][0]
    results={
        'test_y':test_y,
        'test_pred':test_pred,
        'test_nse':test_nse,
        'test_nrmse':test_nrmse,
        'test_ppts':test_ppts,
        'time_cost':time_cost,
    }
    return results

def read_pure_arma(station):
    model_path = root_path+"\\"+station+"\\projects\\arma\\"
    test = pd.read_csv(model_path+'test_pred.csv')
    metrics = pd.read_csv(model_path+'metrics.csv')
    test_y = (test['test_y'][0:120]).values.flatten()
    test_pred = (test['test_pred'][0:120]).values.flatten()
    test_nse = metrics['test_nse'][0]
    test_nrmse = metrics['test_nrmse'][0]
    test_ppts = metrics['test_ppts'][0]
    time_cost = metrics['time_cost'][0]
    results={
        'test_y':test_y,
        'test_pred':test_pred,
        'test_nse':test_nse,
        'test_nrmse':test_nrmse,
        'test_ppts':test_ppts,
        'time_cost':time_cost,
    }
    return results


def read_pca_metrics(station,decomposer,start_component,stop_component,wavelet_level="db10-2"):
    
    if decomposer=='modwt':
        data_path = root_path+"\\"+station+"_"+decomposer+"\\data-wddff\\"+wavelet_level+"\\single_hybrid_1_ahead\\"
    elif decomposer=="dwt":
        data_path = root_path+"\\"+station+"_"+decomposer+"\\data\\"+wavelet_level+"\\one_step_1_ahead_forecast_pacf\\"
    else:
        data_path = root_path+"\\"+station+"_"+decomposer+"\\data\\one_step_1_ahead_forecast_pacf\\"
    train = pd.read_csv(data_path+"minmax_unsample_train.csv")
    dev = pd.read_csv(data_path+"minmax_unsample_dev.csv")
    test = pd.read_csv(data_path+"minmax_unsample_test.csv")
    norm_id=pd.read_csv(data_path+"norm_unsample_id.csv")
    sMax = (norm_id['series_max']).values
    sMin = (norm_id['series_min']).values
    # Conncat the training, development and testing samples
    samples = pd.concat([train,dev,test],axis=0)
    samples = samples.reset_index(drop=True)
    # Renormalized the entire samples
    samples = np.multiply(samples + 1,sMax - sMin) / 2 + sMin
    y = samples['Y']
    X = samples.drop('Y',axis=1)
    pca = PCA(n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_
    print("n_components_pca_mle:{}".format(n_components_pca_mle))
    mle = X.shape[1]-n_components_pca_mle

    nse=[]
    nrmse=[]
    ppts=[]
    for i in range(start_component,stop_component+1):
        if decomposer=='modwt':
            model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr-wddff\\"+wavelet_level+"\\single_hybrid_1_ahead_pca"+str(i)+"\\"
        elif decomposer=="dwt":
            model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\one_step_1_ahead_forecast_pacf_pca"+str(i)+"\\"
        else:
            model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\one_step_1_ahead_forecast_pacf_pca"+str(i)+"\\"
        results = pd.read_csv(model_path+'optimal_model_results.csv')
        test_pred = (results['test_pred'][0:120]).values.flatten()
        test_y = (results['test_y'][0:120]).values.flatten()
        nse.append(results['test_nse'][0])
        nrmse.append(results['test_nrmse'][0])
        ppts.append(results['test_ppts'][0])

    if decomposer == 'modwt':
        pc0=read_two_stage(station=station,decomposer=decomposer,predict_pattern="single_hybrid_1_ahead",)
    else:
        pc0=read_two_stage(station=station,decomposer=decomposer,predict_pattern="one_step_1_ahead_forecast_pacf",)

    nse.append(pc0['test_nse'])
    nrmse.append(pc0['test_nrmse'])
    ppts.append(pc0['test_ppts'])

    nse.reverse()
    nrmse.reverse()
    ppts.reverse()

    results={
        'mle':mle,
        'nse':nse,
        'nrmse':nrmse,
        'ppts':ppts,
    }

    return results

def read_long_leading_time(station,decomposer,mode='pearson',pearson_threshold=0.2,wavelet_level="db10-2"):
    logger.info('reading long lead time model results...')
    logger.info('station:{}'.format(station))
    logger.info('decomposer:{}'.format(decomposer))
    logger.info('mode:{}'.format(mode))
    logger.info('pearson threshold:{}'.format(pearson_threshold))
    logger.info('wavelet level:{}'.format(wavelet_level))

    records=[]
    predictions=[]
    nse=[]
    nrmse=[]
    ppts=[]
    
    
    if decomposer == 'modwt':
        m1=read_two_stage(station=station,decomposer=decomposer,predict_pattern="single_hybrid_1_ahead_mi_ts0.1",)
    else:
        m1=read_two_stage(station=station,decomposer=decomposer,predict_pattern="one_step_1_ahead_forecast_pacf",)
    records.append(m1['test_y'])
    predictions.append(m1['test_pred'])
    nse.append(m1['test_nse'])
    nrmse.append(m1['test_nrmse'])
    ppts.append(m1['test_ppts'])
    # averaging the trained svr with different seed
    leading_times=[3,5,7,9]
    for leading_time in leading_times:
        if decomposer=='modwt':
            model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr-wddff\\"+wavelet_level+"\\"
        elif decomposer=="dwt":
            model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\"
        else:
            model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"
        print("Reading  mode:{}".format(mode))
        if mode=='pacf':
            model_path = model_path+"one_step_"+str(leading_time)+"_ahead_forecast_pacf//"
        elif mode=='pearson':
            model_path = model_path+"one_step_"+str(leading_time)+"_ahead_forecast_pearson"+str(pearson_threshold)+"//"
        elif mode=='mi':
            model_path = model_path+"single_hybrid_"+str(leading_time)+"_ahead_mi_ts0.1//"
        logger.info('model path:{}'.format(model_path))
        results = pd.read_csv(model_path+'optimal_model_results.csv')
        test_pred = (results['test_pred'][0:120]).values.flatten()
        test_y = (results['test_y'][0:120]).values.flatten()
        records.append(test_y)
        predictions.append(test_pred)
        nse.append(results['test_nse'][0])
        nrmse.append(results['test_nrmse'][0])
        ppts.append(results['test_ppts'][0])

    results={
        'records':records,
        'predictions':predictions,
        'nse':nse,
        'nrmse':nrmse,
        'ppts':ppts,
    }

    logger.info('results.records:{}'.format(pd.DataFrame(results)['records']))
    logger.info('results.predictions:{}'.format(pd.DataFrame(results)['predictions']))

    return results





def read_samples_num(station,decomposer,pre=20,wavelet_level="db10-2"):
    if decomposer=='modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data-wddff/"+wavelet_level+"/"
    elif decomposer=="dwt":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"

    leading_time=[3,5,7,9]
    thresh=[0.1,0.2,0.3,0.4,0.5]
    num_sampless=[]
    for lt in leading_time:
        num_samples=[]
        for t in thresh:
            if decomposer=='modwt':
                data = pd.read_csv(data_path+"single_hybrid_"+str(lt)+"_ahead_mi_ts"+str(t)+"/minmax_unsample_train.csv")
            else:
                data = pd.read_csv(data_path+"one_step_"+str(lt)+"_ahead_forecast_pearson"+str(t)+"/minmax_unsample_train.csv")
            data.drop("Y",axis=1)
            num_samples.append(data.shape[1])
        num_sampless.append(num_samples)
    return num_sampless