import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.plot_utils import plot_pacf

plot_pacf(
    station="Huaxian",
    decomposer="modwt",
    wavelet_level='db10-2'
)

plot_pacf(
    station="Huaxian",
    decomposer="modwt",
    wavelet_level='db10-4'
)