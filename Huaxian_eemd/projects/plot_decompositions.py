import pandas as pd
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.plot_utils import plot_decompositions

signal = pd.read_csv(root_path+'/Huaxian_eemd/data/EEMD_TRAIN.csv')
plot_decompositions(signal)
