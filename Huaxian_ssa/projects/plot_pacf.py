import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.plot_utils import plot_pacf

plot_pacf(
    station='Huaxian',
    decomposer='SSA',
    wavelet_level=None,
)