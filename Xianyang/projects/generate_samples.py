import os
root_path = os.path.dirname(os.path.abspath("__file__"))
from variables import variables
import sys
sys.path.append(root_path)
from tools.samples_generator import generate_monoscale_samples



# Generate samples for Xianyang
for lead_time in [1,3,5,7,9]:
    generate_monoscale_samples(
        source_file=root_path+'/time_series/XianyangRunoff1951-2018(1953-2018).xlsx',
        save_path=root_path+'/Xianyang/data/',
         lags_dict = variables['lags_dict'],
        column=['MonthlyRunoff'],
        test_len=120,
        lead_time=lead_time,
    )

