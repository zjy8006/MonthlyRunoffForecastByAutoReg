import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.plot_utils import plot_pacf

plot_pacf(
    station='Xianyang',
    decomposer=None,
    wavelet_level=None,
)