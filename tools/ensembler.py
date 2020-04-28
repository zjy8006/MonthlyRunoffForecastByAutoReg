import pandas as pd
import numpy as np
import math

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from skopt import dump,load
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.plot_utils import plot_subsignals_pred,plot_rela_pred
from tools.metrics_ import PPTS
from config.globalLog import logger
import json
with open(root_path+'/config/config.json') as handle:
    dictdump = json.loads(handle.read())
data_part=dictdump['data_part']



# ensemble models for multi-step decomposition-ensemble models
def ensemble(root_path,original_series,station,predictor,predict_pattern,variables,decomposer=None,wavelet_level='db10-2',framework='WDDFF'):

    if decomposer=='modwt':
        if framework=='TSDP':
            lags_dict = variables['lags_dict'][wavelet_level]
        else:
            lags_dict =None
    elif decomposer=='dwt':
        lags_dict = variables['lags_dict'][wavelet_level]
    else:
        lags_dict = variables['lags_dict']
    full_len = variables['full_len']
    train_len = variables['train_len']
    dev_len = variables['dev_len']
    test_len = variables['test_len']
    logger.info('Ensemble forecasting results...')
    logger.info('Root path:{}'.format(root_path))
    logger.info('original series:\n{}'.format(original_series))
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))   
    logger.info('Lags dict:{}'.format(lags_dict))
    logger.info('Predictor:{}'.format(predictor))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Training length:{}'.format(train_len))
    logger.info('Development length:{}'.format(test_len))
    logger.info('Testing length:{}'.format(test_len))
    logger.info('Entire length:{}'.format(full_len))
    logger.info('Wavelet and decomposition level of WA:{}'.format(wavelet_level))
    
    
    if decomposer=='modwt':
        models_path = root_path+'/'+station+'_'+decomposer+'/projects/'+predictor+'-'+framework.lower()+'/'+wavelet_level+'/'+predict_pattern+'/'
    elif decomposer=='dwt':
        models_path = root_path+'/'+station+'_'+decomposer+'/projects/'+predictor+'/'+wavelet_level+'/'+predict_pattern+'/'
    elif decomposer==None:
        models_path = root_path+'/'+station+'/projects/'+predictor+'/'+predict_pattern+'/'
    else:
        models_path = root_path+'/'+station+'_'+decomposer+'/projects/'+predictor+'/'+predict_pattern+'/'
    logger.info("Model path:{}".format(models_path))

    if 'multi_step' not in predict_pattern:
        models_history = models_path+'history/'
        optimal_model = ''
        min_dev_mse = np.inf
        for file_ in os.listdir(models_history):
            if '.csv' in file_ and 'optimized_params' not in file_:
                logger.info('read model results:{}'.format(file_))
                dev_mse = pd.read_csv(models_history+file_)['dev_mse'][0]
                if dev_mse < min_dev_mse:
                    min_dev_mse = dev_mse
                    optimal_model = file_
        logger.info('Optimal model:{}'.format(optimal_model))
        logger.info('Minimum MSE={}'.format(min_dev_mse))
        res = load(models_history+(optimal_model.split('.csv')[0]+'_result.pkl'))
        dump(res,models_path+'result.pkl')
        optimal_model = pd.DataFrame([optimal_model],columns=['optimal_model'])
        optimal_results = pd.read_csv(models_history+optimal_model['optimal_model'][0])
        if predictor=='esvr' or predictor=='gbrt':
            optimal_params = pd.read_csv(models_history+optimal_model['optimal_model'][0].split('.csv')[0]+'_optimized_params.csv')
            optimal_results = pd.concat([optimal_model,optimal_params,optimal_results],axis=1)
        elif predictor=='lstm':
            optimal_results = pd.concat([optimal_model,optimal_results],axis=1)
        optimal_results.to_csv(models_path+'optimal_model_results.csv')
        plot_rela_pred(optimal_results['train_y'],optimal_results['train_pred'],models_path+'train_pred.png')
        plot_rela_pred(optimal_results['dev_y'][0:data_part['dev_len']],optimal_results['dev_pred'][0:data_part['dev_len']],models_path+'dev_pred.png')
        plot_rela_pred(optimal_results['test_y'][0:data_part['test_len']],optimal_results['test_pred'][0:data_part['test_len']],models_path+'test_pred.png')
    else:
        for i in range(len(lags_dict)):
            print(len(lags_dict))
            model_path = models_path+'s'+str(i+1)+'/'
            models_history = model_path+'history/'
            optimal_model = ''
            min_dev_mse = np.inf
            for file_ in os.listdir(models_history):
                if '.csv' in file_ and 'optimized_params' not in file_:
                    logger.info('read model results:{}'.format(file_))
                    dev_mse = pd.read_csv(models_history+file_)['dev_mse'][0]
                    if dev_mse < min_dev_mse:
                        min_dev_mse = dev_mse
                        optimal_model = file_
            logger.info('Optimal model:{}'.format(optimal_model))
            logger.info('Minimum MSE={}'.format(min_dev_mse))
            res = load(models_history+(optimal_model.split('.csv')[0]+'_result.pkl'))
            dump(res,model_path+'result.pkl')
            optimal_model = pd.DataFrame([optimal_model],columns=['optimal_model'])
            optimal_results = pd.read_csv(models_history+optimal_model['optimal_model'][0])
            if predictor=='esvr' or predictor=='gbrt':
                optimal_params = pd.read_csv(models_history+optimal_model['optimal_model'][0].split('.csv')[0]+'_optimized_params.csv')
                optimal_results = pd.concat([optimal_model,optimal_params,optimal_results],axis=1)
            elif predictor=='lstm':
                optimal_results = pd.concat([optimal_model,optimal_results],axis=1)
            optimal_results.to_csv(model_path+'optimal_model_results.csv')
            plot_rela_pred(optimal_results['train_y'],optimal_results['train_pred'],model_path+'train_pred.png')
            plot_rela_pred(optimal_results['dev_y'][0:data_part['dev_len']],optimal_results['dev_pred'][0:data_part['dev_len']],model_path+'dev_pred.png')
            plot_rela_pred(optimal_results['test_y'][0:data_part['test_len']],optimal_results['test_pred'][0:data_part['test_len']],model_path+'test_pred.png')
        train_len_ = train_len - max(lags_dict.values())
        train_sum_pred = pd.DataFrame()
        dev_sum_pred = pd.DataFrame()
        test_sum_pred = pd.DataFrame()
        time_cost_sum = 0.0
        for i in range(len(lags_dict)):
            model_path = models_path+'s'+str(i+1)+'/'
            results = pd.read_csv(model_path+'optimal_model_results.csv')
            time_cost_sum = time_cost_sum+results['time_cost'][0]
            train_pred = results['train_pred']
            train_pred = train_pred[train_pred.shape[0]-train_len_:]
            train_pred = train_pred.reset_index(drop=True)
            dev_pred = results['dev_pred'][0:dev_len]
            test_pred = results['test_pred'][0:test_len]
            train_sum_pred = pd.concat([train_sum_pred,train_pred],axis=1)
            dev_sum_pred = pd.concat([dev_sum_pred,dev_pred],axis=1)
            test_sum_pred = pd.concat([test_sum_pred,test_pred],axis=1)
        train_sum_pred = train_sum_pred.sum(axis=1)
        dev_sum_pred = dev_sum_pred.sum(axis=1)
        test_sum_pred = test_sum_pred.sum(axis=1)
        train_sum_pred[train_sum_pred<0.0]=0.0
        dev_sum_pred[dev_sum_pred<0.0]=0.0
        test_sum_pred[test_sum_pred<0.0]=0.0
        original_series=original_series.reset_index(drop=True)
        train_y = original_series[train_len-train_len_:train_len]
        dev_y = original_series[train_len:train_len+dev_len]
        test_y = original_series[train_len+dev_len:]
        train_y = train_y.reset_index(drop=True)
        dev_y = dev_y.reset_index(drop=True)
        test_y = test_y.reset_index(drop=True)

        train_nse = r2_score(train_y.values, train_sum_pred.values)
        train_mse = mean_squared_error(train_y.values, train_sum_pred.values)
        train_nrmse = math.sqrt(mean_squared_error(train_y.values, train_sum_pred.values))/(sum(train_y.values)/len(train_y.values))
        train_ppts=PPTS(train_y.values,train_sum_pred.values,5)
        
        dev_nse = r2_score(dev_y.values, dev_sum_pred.values)
        dev_mse = mean_squared_error(dev_y.values, dev_sum_pred.values)
        dev_nrmse = math.sqrt(mean_squared_error(dev_y.values, dev_sum_pred.values))/(sum(dev_y.values)/len(dev_y.values))
        dev_ppts=PPTS(dev_y.values,dev_sum_pred.values,5)

        test_nse = r2_score(test_y.values, test_sum_pred.values)
        test_mse = mean_squared_error(test_y.values, test_sum_pred.values)
        test_nrmse = math.sqrt(mean_squared_error(test_y.values, test_sum_pred.values))/(sum(test_y.values)/len(test_y.values))
        test_ppts=PPTS(test_y.values,test_sum_pred.values,5)

        metrics = {
            'train_nse':train_nse,
            'train_mse':train_mse,
            'train_nrmse':train_nrmse,
            'train_ppts':train_ppts,
            'dev_nse':dev_nse,
            'dev_mse':dev_mse,
            'dev_nrmse':dev_nrmse,
            'dev_ppts':dev_ppts,
            'test_nse':test_nse,
            'test_mse':test_mse,
            'test_nrmse':test_nrmse,
            'test_ppts':test_ppts,
            'time_cost':time_cost_sum,
        }
        metrics_df = pd.DataFrame(metrics,index=[0])
        print(metrics_df)
        train_results = pd.concat([train_y,train_sum_pred],axis=1)
        train_results = pd.DataFrame(train_results.values,columns=['train_y','train_pred'])
        dev_results = pd.concat([dev_y,dev_sum_pred],axis=1)
        dev_results = pd.DataFrame(dev_results.values,columns=['dev_y','dev_pred'])
        test_results = pd.concat([test_y,test_sum_pred],axis=1)
        test_results = pd.DataFrame(test_results.values,columns=['test_y','test_pred'])
        optimal_results = pd.concat([train_results,dev_results,test_results,metrics_df],axis=1)
        optimal_results.to_csv(models_path+'optimal_results.csv')
        plot_rela_pred(train_y,train_sum_pred,models_path+'train_pred.png')
        plot_rela_pred(dev_y,dev_sum_pred,models_path+'dev_pred.png')
        plot_rela_pred(test_y,test_sum_pred,models_path+'test_pred.png')
        



    # if predictor=='lstm':
    #     optimal_models = []
    #     models_time_cost = []
    #     # Fine the trained file with lowest RMSE during development period
    #     for i in range(1,len(lags_dict)+1):
    #         models_path = models_path+'s'+str(i)+'/'
    #         models = []
    #         dev_rmse = []
    #         time_cost = []
    #         for files in os.listdir(models_path):
    #             if '.csv' in files and 'HISTORY' not in files:
    #                 # print(files)
    #                 models.append(files)
    #                 dev_rmse.append(pd.read_csv(models_path+files)['dev_rmse'][0])
    #                 time_cost.append(pd.read_csv(models_path+files)['time_cost'][0])
    #         models_time_cost.append(sum(time_cost))       
    #         # print(models)
    #         metrix_dict ={
    #             'models':models,
    #             'dev_rmse':dev_rmse,
    #         }
    #         metrix_df = pd.DataFrame(metrix_dict)
    #         min_idx = metrix_df['dev_rmse'].idxmin()
    #         optimal_models.append(metrix_df['models'].loc[min_idx])


    #     print("Optimal models:")
    #     for optimal_model in optimal_models:
    #         print(optimal_model)

    # # initialize empty evaluation metrics list
    # train_nrmse = []
    # train_r2 = []
    # train_mae = []
    # train_mape = []
    # train_ppts = []

    # dev_nrmse = []
    # dev_r2 = []
    # dev_mae = []
    # dev_mape = []
    # dev_ppts = []

    # test_nrmse = []
    # test_r2 = []
    # test_mae = []
    # test_mape = []
    # test_ppts = []
    # time_cost = []
    # # initialize empty hyper-parameters matrix
    # if predictor=='esvr':
    #     C=[]
    #     epsilon=[]
    #     gamma=[]
    # elif predictor=='gbrt':
    #     max_depth=[]
    #     learning_rate=[]
    #     max_features=[]
    #     min_samples_split=[]
    #     min_samples_leaf=[]
    # # Initialize two empty pandas DataFrame for sub-signals predictions and records
    # imf_test_pred = pd.DataFrame()
    # imf_test_y = pd.DataFrame()
    

    # # perform ensemble
    # for i in range(1,len(lags_dict)+1):
    #     model_path = models_path+'/s'+str(i)+'/'
    #     if predictor=='esvr' or predictor=='gbrt':
    #         model_name =station+'_'+decomposer+'_'+predictor+'_'+predict_pattern+'_s'+str(i)
    #         data = pd.read_csv(model_path+model_name+'.csv')
    #     elif predictor=='lstm':
    #         data = pd.read_csv(model_path+optimal_models[i-1])
    #     testYY=data['test_y'][0:test_len]
    #     testYY=testYY.reset_index(drop=True)
    #     imf_test_y = pd.concat([imf_test_y,testYY],axis=1)
    #     train_nrmse.append(data['train_nrmse'][0])
    #     train_r2.append(data['train_r2'][0])
    #     train_mae.append(data['train_mae'][0])
    #     train_mape.append(data['train_mape'][0])
    #     train_ppts.append(data['train_ppts'][0])
    #     dev_nrmse.append(data['dev_nrmse'][0])
    #     dev_r2.append(data['dev_r2'][0])
    #     dev_mae.append(data['dev_mae'][0])
    #     dev_mape.append(data['dev_mape'][0])
    #     dev_ppts.append(data['dev_ppts'][0])
    #     test_nrmse.append(data['test_nrmse'][0])
    #     test_r2.append(data['test_r2'][0])
    #     test_mae.append(data['test_mae'][0])
    #     test_mape.append(data['test_mape'][0])
    #     test_ppts.append(data['test_ppts'][0])
    #     time_cost.append(data['time_cost'][0])
    #     testPP=data['test_pred'][0:test_len]
    #     testPP=testPP.reset_index(drop=True)
    #     imf_test_pred = pd.concat([imf_test_pred,testPP],axis=1)

    #     hyper_params = pd.read_csv(model_path+model_name+'_optimized_params_s'+str(i)+'.csv')
    #     if predictor=='esvr':
    #         C.append(hyper_params['C'][0])
    #         epsilon.append(hyper_params['epsilon'][0])
    #         gamma.append(hyper_params['gamma'][0])
    #     elif predictor=='gbrt':
    #         max_depth.append(hyper_params['max_depth'][0])
    #         learning_rate.append(hyper_params['learning_rate'][0])
    #         max_features.append(hyper_params['max_features'][0])
    #         min_samples_split.append(hyper_params['min_samples_split'][0])
    #         min_samples_leaf.append(hyper_params['min_samples_leaf'][0])

    # plot_subsignals_pred(
    #     predictions=imf_test_pred,
    #     records=imf_test_y,
    #     test_len=test_len,
    #     full_len=full_len,
    #     fig_savepath=models_path+predictor+'_'+signals+'_multi_pred.eps',
    #     format='EPS',
    #     dpi=2000,
    # )

    # # Generate columns for sub-signals' predictions
    # columns = []
    # for i in range(1,len(lags_dict)+1):
    #     columns.append('IMF'+str(i)+'_P')
    # # Set columns for sub-signals's predictions
    # imf_test_pred_df = pd.DataFrame(imf_test_pred.values,columns=columns)

    # # Ensemble the sub-signals' predictions
    # test_pred_ensem_df = imf_test_pred_df.sum(axis=1)
    # # cap the negative predictions to 0
    # test_pred_ensem_df[test_pred_ensem_df<0.0]=0.0
    # # Set the column name of the ensemble predictions as 'pred'
    # ensem_test_pred_df = pd.DataFrame(test_pred_ensem_df,columns=['pred'])

    # # Get the original time series
    # orig=pd.read_excel(root_path+'/time_series/'+original)['MonthlyRunoff']
    # test_y_df=pd.DataFrame((orig[orig.shape[0]-test_len:]).values,columns=['orig'])
    # test_pred_df = pd.concat([imf_test_pred_df,ensem_test_pred_df,test_y_df],axis=1)
    # test_pred_df.to_csv(models_path+predictor+'_'+signals+'_sum_test_result.csv',index=None)

    # # Plot the ensemble predictions fitness fig and scatters
    # test_y = (test_y_df.values).flatten()
    # test_predictions = (test_pred_ensem_df.values).flatten()
    # plot_rela_pred(test_y,test_predictions,fig_savepath=models_path+predictor+'_'+signals+'_sum_test_pred_rela.png')

    # # Construct the evaluation metrics as dict
    # if predictor=='esvr':
    #     metrics_dict = {
    #         'C':C,
    #         'epsilon':epsilon,
    #         'gamma':gamma,
    #         'time_cost':time_cost,
    #         'train_nrmse':train_nrmse,
    #         'train_r2':train_r2,
    #         'train_mae':train_mae,
    #         'train_mape':train_mape,
    #         'train_ppts':train_ppts,
    #         'dev_nrmse':dev_nrmse,
    #         'dev_r2':dev_r2,
    #         'dev_mae':dev_mae,
    #         'dev_mape':dev_mape,
    #         'dev_ppts':dev_ppts,
    #         'test_nrmse':test_nrmse,
    #         'test_r2':test_r2,
    #         'test_mae':test_mae,
    #         'test_mape':test_mape,
    #         'test_ppts':test_ppts,
    #     }
    # elif predictor=='gbrt':
    #     metrics_dict = {
    #         'max_depth':max_depth,
    #         'learning_rate':learning_rate,
    #         'max_features':max_features,
    #         'min_samples_split':min_samples_split,
    #         'min_samples_leaf':min_samples_leaf,
    #         'time_cost':time_cost,
    #         'train_nrmse':train_nrmse,
    #         'train_r2':train_r2,
    #         'train_mae':train_mae,
    #         'train_mape':train_mape,
    #         'train_ppts':train_ppts,
    #         'dev_nrmse':dev_nrmse,
    #         'dev_r2':dev_r2,
    #         'dev_mae':dev_mae,
    #         'dev_mape':dev_mape,
    #         'dev_ppts':dev_ppts,
    #         'test_nrmse':test_nrmse,
    #         'test_r2':test_r2,
    #         'test_mae':test_mae,
    #         'test_mape':test_mape,
    #         'test_ppts':test_ppts,
    #     }

    # elif predictor=='lstm':
    #     metrics_dict = {
    #         'model':optimal_models,
    #         'time_cost':models_time_cost,
    #         'train_nrmse':train_nrmse,
    #         'dev_nrmse':dev_nrmse,
    #         'test_nrmse':test_nrmse,
    #         'train_r2':train_r2,
    #         'dev_r2':dev_r2,
    #         'test_r2':test_r2,
    #         'train_mae':train_mae,
    #         'dev_mae':dev_mae,
    #         'test_mae':test_mae,
    #         'train_mape':train_mape,
    #         'dev_mape':dev_mape,
    #         'test_mape':test_mape,
    #         'train_ppts':train_ppts,
    #         'dev_ppts':dev_ppts,
    #         'test_ppts':test_ppts,
    #     }

    # # Transform the evaluation metrics dict to pandas DataFrame
    # metrics_df = pd.DataFrame(metrics_dict)
    # # Save the evaluation metrics for sub-signals
    # metrics_df.to_csv(models_path+predictor+'_'+signals+'_imfs_model_metrics.csv')

    # # Compute the metrics for ensemble predictions
    # test_r2 = r2_score(test_y, test_predictions)
    # test_nrmse = math.sqrt(mean_squared_error(test_y, test_predictions))/(sum(test_y)/len(test_y))
    # test_mae = mean_absolute_error(test_y, test_predictions)
    # test_mape=np.mean(np.abs((test_y - test_predictions) / test_y)) * 100
    # test_ppts=PPTS(test_y,test_predictions,5)
    # if predictor=='esvr' or predictor=='gbrt':
    #     full_time_cost = sum(time_cost)
    # elif predictor=='lstm':
    #     full_time_cost = sum(models_time_cost)
    # # Transform the metrics into pandas DataFrame
    # test_r2 = pd.DataFrame([test_r2],columns=['test_r2'])
    # test_nrmse = pd.DataFrame([test_nrmse],columns=['test_nrmse'])
    # test_mae = pd.DataFrame([test_mae],columns=['test_mae'])
    # test_mape = pd.DataFrame([test_mape],columns=['test_mape'])
    # test_ppts = pd.DataFrame([test_ppts],columns=['test_ppts'])
    # full_time_cost = pd.DataFrame([full_time_cost],columns=['time_cost'])
    # # Concat the metrics DataFrame
    # ensemble_test_metrics_df = pd.concat([test_r2,test_nrmse,test_mae,test_mape,test_ppts,full_time_cost],axis=1)
    # ensemble_test_metrics_df.to_csv(models_path+predictor+'_'+signals+'_sum_model_test_metrics.csv')
