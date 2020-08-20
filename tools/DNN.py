import matplotlib.pyplot as plt
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras import layers
from tensorflow import keras
import json
import time
import datetime
import numpy as np
import pandas as pd
import os
import sys
root = os.path.abspath(os.path.dirname('__file__'))
sys.path.append(root)
from tools.dump_data import dump_pred_results
from tools.plot_utils import plot_error_distribution
from tools.plot_utils import plot_history
from tools.plot_utils import plot_rela_pred


def BuildDNN(train_samples, dev_samples, test_samples, norm_id,model_path, lags=None, seed=None,
              batch_size=512, n_epochs=5, max_trials=5, executions_per_trial=3,
              max_hidden_layers = 3, min_units = 16, max_units = 64, unit_step=16,
              min_droprate = 0.0, max_droprate=0.5, droprate_step=0.05,
              min_learnrate = 1e-4, max_learnrate=1e-1,
              n_tune_epochs=5, cast_to_zero=True, early_stop=True,early_stop_patience=10, retrain=False,
              warm_up=False, initial_epoch=None,measurement_time='day',measurement_unit='$m^3/s$'):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    setting_info = {
        "model_path":model_path,
        "lags":lags,
        "seed":seed,
        "batch_size":batch_size,
        "n_epoch":n_epochs,
        "max_trials":max_trials,
        "executions_per_trial":executions_per_trial,
        "max_hidden_layers":max_hidden_layers,
        "min_units":min_units,
        "max_units":max_units,
        "unit_step":unit_step,
        "min_droprate":min_droprate,
        "max_droprate":max_droprate,
        "droprate_step":droprate_step,
        "min_learnrate":min_learnrate,
        "max_learnrate":max_learnrate,
        "n_tune_epochs":n_tune_epochs,
        "cast_to_zero":cast_to_zero,
        "early_stop":early_stop,
        "early_stop_patience":early_stop_patience,
        "retrain":retrain,
    }

    with open(model_path+'setting.json', 'w') as outfile:
        json.dump(setting_info, outfile)

    sMin = norm_id['series_min']
    sMax = norm_id['series_max']
    # sMin = train_samples.min(axis=0)
    # sMax = train_samples.max(axis=0)
    # train_samples = 2*(train_samples-sMin)/(sMax-sMin)-1
    # dev_samples = 2*(dev_samples-sMin)/(sMax-sMin)-1
    # test_samples = 2*(test_samples-sMin)/(sMax-sMin)-1
    cal_samples = pd.concat([train_samples, dev_samples], axis=0)
    cal_samples = cal_samples.sample(frac=1)
    cal_samples = cal_samples.reset_index(drop=True)
    train_samples = cal_samples.iloc[:train_samples.shape[0]]
    dev_samples = cal_samples.iloc[train_samples.shape[0]:]
    X = cal_samples
    y = (cal_samples.pop('Y')).values
    train_x = train_samples
    train_y = train_samples.pop('Y')
    train_y = train_y.values
    dev_x = dev_samples
    dev_y = dev_samples.pop('Y')
    dev_y = dev_y.values
    test_x = test_samples
    test_y = test_samples.pop('Y')
    test_y = test_y.values

    # Config path to save optimal results
    opt_path = model_path+'\\optimal\\'
    cp_path = model_path+'\\optimal\\checkpoints\\'
    if not os.path.exists(cp_path):
        os.makedirs(cp_path)
    # restore only the latest checkpoint after every update
    checkpoint_path = cp_path+'cp.h5'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Define callbacks
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, mode='min', save_weights_only=True, verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=0.00001, factor=0.2, verbose=1, patience=10, mode='min')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience, restore_best_weights=True)

    def build_model(hp):
        input_shape = (train_x.shape[1],)
        model = keras.Sequential()
        num_layers = hp.Int('num_layers', min_value=1, max_value=max_hidden_layers, step=1, default=1)
        for i in range(num_layers):
            units = hp.Int('units_'+str(i), min_value=min_units,max_value=max_units, step=unit_step)
            dropout_rate = hp.Float('drop_rate_' + str(i), min_value=min_droprate, max_value=max_droprate, step=droprate_step)
            if i == 0:
                model.add(layers.Dense(units=units,activation='relu', input_shape=input_shape))
            else:
                model.add(layers.Dense(units=units, activation='relu'))
            model.add(layers.Dropout(rate=dropout_rate,noise_shape=None, seed=seed))
        model.add(layers.Dense(1))
        model.compile(
            optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=min_learnrate, max_value=max_learnrate, sampling='LOG', default=1e-2)),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_squared_error'])
        return model

    tuner = BayesianOptimization(
        build_model,
        objective='mean_squared_error',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=model_path,
        project_name='BayesianOpt')

    tuner.search_space_summary()
    start = time.process_time()
    tuner.search(x=train_x,
                 y=train_y,
                 epochs=n_tune_epochs,
                 validation_data=(dev_x, dev_y),
                 callbacks=[early_stopping])
    end = time.process_time()
    time_cost = end-start
    tuner.results_summary()
    best_hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
    model = build_model(best_hps)
    
    if retrain or not os.path.exists(checkpoint_path):
        history = model.fit(X, y, epochs=n_epochs, batch_size=batch_size, validation_data=(X, y), verbose=1,
                            callbacks=[cp_callback,early_stopping,])
        hist = pd.DataFrame(history.history)
        hist.to_csv(opt_path+'PARAMS-CAL-HISTORY.csv')
        plot_history(history, opt_path+'MAE-HISTORY.png',opt_path+'MSE-HISTORY.png')
    else:
        model.load_weights(checkpoint_path)

    train_predictions = model.predict(train_x).flatten()
    dev_predictions = model.predict(dev_x).flatten()
    test_predictions = model.predict(test_x).flatten()
    sMax = sMax[sMax.shape[0]-1]
    sMin = sMin[sMin.shape[0]-1]
    train_y = np.multiply(train_y + 1, sMax - sMin) / 2 + sMin
    dev_y = np.multiply(dev_y + 1, sMax - sMin) / 2 + sMin
    test_y = np.multiply(test_y + 1, sMax - sMin) / 2 + sMin
    train_predictions = np.multiply(train_predictions + 1, sMax - sMin) / 2 + sMin
    dev_predictions = np.multiply(dev_predictions + 1, sMax - sMin) / 2 + sMin
    test_predictions = np.multiply(test_predictions + 1, sMax - sMin) / 2 + sMin
    if cast_to_zero:
        train_predictions[train_predictions < 0.0] = 0.0
        dev_predictions[dev_predictions < 0.0] = 0.0
        test_predictions[test_predictions < 0.0] = 0.0
    dump_pred_results(
        path=opt_path+'/opt_pred.csv',
        train_y=train_y,
        train_predictions=train_predictions,
        dev_y=dev_y,
        dev_predictions=dev_predictions,
        test_y=test_y,
        test_predictions=test_predictions,
        time_cost=time_cost,
    )
    plot_rela_pred(train_y, train_predictions, measurement_time=measurement_time,measurement_unit=measurement_unit, fig_savepath=opt_path + 'TRAIN-PRED.png')
    plot_rela_pred(dev_y, dev_predictions, measurement_time=measurement_time,measurement_unit=measurement_unit, fig_savepath=opt_path + "DEV-PRED.png")
    plot_rela_pred(test_y, test_predictions, measurement_time=measurement_time,measurement_unit=measurement_unit, fig_savepath=opt_path + "TEST-PRED.png")
    plot_error_distribution(test_predictions, test_y,opt_path+'TEST-ERROR-DSTRI.png')
    plt.show()



