import sys
import os
root_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(root_path)
from tools.samples_generator import gen_direct_forecast_samples
from tools.samples_generator import gen_multi_forecast_samples
from Xianyang_ssa.projects.variables import variables

gen_direct_forecast_samples(
    station="Xianyang",
    decomposer="ssa",
    lags_dict=variables['lags_dict'],
    input_columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
               'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
    output_column=['ORIG'],
    start=533,
    stop=792,
    test_len=120,
    gen_from='training and validation sets',
)


gen_direct_forecast_samples(
    station="Xianyang",
    decomposer="ssa",
    lags_dict=variables['lags_dict'],
    input_columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
               'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
    output_column=['ORIG'],
    start=533,
    stop=792,
    test_len=120,
    gen_from='training-development and appended sets',
)

gen_direct_forecast_samples(
    station="Xianyang",
    decomposer="ssa",
    lags_dict=variables['lags_dict'],
    input_columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
               'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
    output_column=['ORIG'],
    start=533,
    stop=792,
    test_len=120,
    gen_from='training-development and test sets',
)

for lead_time in [1,3,5,7,9]:
    gen_direct_forecast_samples(
        station='Xianyang',
        decomposer="ssa",
        lags_dict=variables['lags_dict'],
        input_columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
                       'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
        output_column=['ORIG'],
        start=533,
        stop=792,
        test_len=120,
        mode='PACF',
        lead_time=lead_time,
        gen_from='training and appended sets',
    )

for lead_time in [1, 3, 5, 7, 9]:
    gen_direct_forecast_samples(
        station='Xianyang',
        decomposer="ssa",
        lags_dict=variables['lags_dict'],
        input_columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
                       'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
        output_column=['ORIG'],
        start=533,
        stop=792,
        test_len=120,
        mode='Pearson',
        lead_time=lead_time,
        gen_from='training and appended sets',
    )



gen_multi_forecast_samples(
    station='Xianyang',
    decomposer="ssa",
    lags_dict=variables['lags_dict'],
    columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
             'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
    start=533,
    stop=792,
    test_len=120,
)

gen_direct_forecast_samples(
    station='Xianyang',
    decomposer="ssa",
    lags_dict=variables['lags_dict'],
    input_columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
                   'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
    output_column=['ORIG'],
    start=533,
    stop=792,
    test_len=120,
    mode='PACF',
    lead_time=1,
    n_components=18,
    gen_from='training and appended sets',
)

gen_direct_forecast_samples(
    station='Xianyang',
    decomposer="ssa",
    lags_dict=variables['lags_dict'],
    input_columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
                   'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
    output_column=['ORIG'],
    start=533,
    stop=792,
    test_len=120,
    mode='PACF',
    lead_time=1,
    n_components='mle',
    gen_from='training and appended sets',
)

num_in_one=sum(variables['lags_dict'].values())
for n_components in range(num_in_one-16, num_in_one+1):
    gen_direct_forecast_samples(
        station='Xianyang',
        decomposer="ssa",
        lags_dict=variables['lags_dict'],
        input_columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
                       'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
        output_column=['ORIG'],
        start=533,
        stop=792,
        test_len=120,
        mode='PACF',
        lead_time=1,
        n_components=n_components,
        gen_from='training and appended sets',
    )
