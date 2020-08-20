import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt import gp_minimize,forest_minimize, dummy_minimize
from skopt import dump, load
import datetime
import time
import os
root = os.path.abspath(os.path.dirname('__file__'))
import sys
sys.path.append(root)
from tools.plot_utils import plot_rela_pred
from tools.plot_utils import plot_history
from tools.plot_utils import plot_error_distribution
from tools.plot_utils import plot_convergence_
from tools.plot_utils import plot_evaluations_
from tools.plot_utils import plot_objective_
from tools.dump_data import dump_pred_results
import joblib

def BuildGBRT(train_samples,dev_samples,test_samples,model_path,n_calls,cast_to_zero=True,optimizer='gp',measurement_time='day',measurement_unit='$m^3/s$'):
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    sMin = train_samples.min(axis=0)
    sMax = train_samples.max(axis=0)
    norm = pd.concat([sMax,sMin],axis=1)
    norm =pd.DataFrame(norm.values,columns=['sMax','sMin'],index=train_samples.columns.values)
    norm.to_csv(model_path+'norm.csv')
    joblib.dump(norm,model_path+'norm.pkl')


    train_samples = 2*(train_samples-sMin)/(sMax-sMin)-1
    dev_samples = 2*(dev_samples-sMin)/(sMax-sMin)-1
    test_samples = 2*(test_samples-sMin)/(sMax-sMin)-1
    cal_samples = pd.concat([train_samples,dev_samples],axis=0)
    cal_samples = cal_samples.sample(frac=1)
    train_y = train_samples['Y']
    train_x = train_samples.drop('Y', axis=1)
    dev_y = dev_samples['Y']
    dev_x = dev_samples.drop('Y', axis=1)
    test_y = test_samples['Y']
    test_x = test_samples.drop('Y', axis=1)
    cal_y = cal_samples['Y']
    cal_x = cal_samples.drop('Y', axis=1)

    predictor_columns = list(train_x.columns)
    joblib.dump(predictor_columns, model_path+'predictor_columns.pkl')

    
    # Get the feature num
    n_features = cal_x.shape[1]
    reg = GradientBoostingRegressor(n_estimators=100,random_state=0)
    # The list hyper-parameters we want
    space = [
        Integer(1,25,name='max_depth'),
        Real(10**-5,10**0,'log-uniform',name='learning_rate'),
        Integer(1,n_features,name='max_features'),
        Integer(2,100,name='min_samples_split'),
        Integer(1,100,name='min_samples_leaf'),
    ]

    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        return -np.mean(cross_val_score(reg,cal_x,cal_y,cv=10,n_jobs=-1,scoring='neg_mean_squared_error'))

    start = time.process_time()
    if optimizer=='gp':
        res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True,n_jobs=-1)
    elif optimizer=='fr_bt':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True,n_jobs=-1)
    elif optimizer=='fr_rf':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True,n_jobs=-1)
    elif optimizer=='dm':
        res = dummy_minimize(objective,space,n_calls=n_calls)
    end=time.process_time()
    time_cost = end-start

    dump(res,model_path+'tune_history.pkl',store_objective=False)
    # returned_results = load(model_path+'tune_history.pkl')
    DIMENSION_GBRT = ['max depth','learning rate','max features','min samples split','min samples leaf']
    plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+'objective.png')
    plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+'evaluation.png')
    plot_convergence_(res,fig_savepath=model_path+'convergence.png')


    # logger.info('Best score=%.4f'%res.fun)
    # logger.info("""Best parameters:
    # - max_depth=%d
    # - learning_rate=%.6f
    # - max_features=%d
    # - min_samples_split=%d
    # - min_samples_leaf=%d""" % (res.x[0], res.x[1], res.x[2], res.x[3],
    #                             res.x[4]))
    # logger.info('Time cost:{}'.format(time_cost))

    params_dict={
        'max_depth':res.x[0],
        'learning_rate':res.x[1],
        'max_features':res.x[2],
        'min_samples_split':res.x[3],
        'min_samples_leaf':res.x[4],
        'time_cost':time_cost,
        'n_calls':n_calls,
    }

    params_df = pd.DataFrame(params_dict,index=[0])
    params_df.to_csv(model_path +'optimized_params.csv')

    GBR = GradientBoostingRegressor(
        max_depth=res.x[0],
        learning_rate=res.x[1],
        max_features=res.x[2],
        min_samples_split=res.x[3],
        min_samples_leaf=res.x[4]).fit(cal_x,cal_y)

    joblib.dump(GBR,model_path+'model.pkl')

    GBR = joblib.load(model_path+'model.pkl')
    train_predictions = GBR.predict(train_x)
    dev_predictions = GBR.predict(dev_x)
    test_predictions = GBR.predict(test_x)
    train_y=(train_y.values).flatten()
    dev_y=(dev_y.values).flatten()
    test_y=(test_y.values).flatten()
    sMax = sMax[sMax.shape[0]-1]
    sMin = sMin[sMin.shape[0]-1]
    train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
    dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
    test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
    train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
    dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
    test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
    if cast_to_zero:
        train_predictions[train_predictions<0.0]=0.0
        dev_predictions[dev_predictions<0.0]=0.0
        test_predictions[test_predictions<0.0]=0.0
    dump_pred_results(
        path = model_path+'opt_pred.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost = time_cost,
    )
    plot_rela_pred(train_y,train_predictions,measurement_time=measurement_time,measurement_unit=measurement_unit,fig_savepath=model_path  + 'TRAIN-PRED.png')
    plot_rela_pred(dev_y,dev_predictions,measurement_time=measurement_time,measurement_unit=measurement_unit,fig_savepath=model_path  + "DEV-PRED.png")
    plot_rela_pred(test_y,test_predictions,measurement_time=measurement_time,measurement_unit=measurement_unit,fig_savepath=model_path  + "TEST-PRED.png")
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path+"TEST-ERROR-DSTRI.png")
    plt.show()
    plt.close('all')
    