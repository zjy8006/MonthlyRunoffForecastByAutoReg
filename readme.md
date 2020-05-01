# Code and data repository for ["Two-stage Variational Mode Decomposition and Support Vector Regression"](https://www.hydrol-earth-syst-sci-discuss.net/hess-2019-565/#discussion)


## Title

Code and data for "Two-stage Variational Mode Decomposition and Support Vector Regression for Streamflow Forecasting"

## Author

Ganggang Zuo (zuoganggang@163.com)

## Institutions

State Key Laboratory of Eco-hydraulics in Northwest Arid Region, Xi’an University of Technology, Xi’an, Shaanxi 710048, China

## Categories

hydro; signal preprocessing; surface water

## Description

This data repository contains code and data for the research article “_Two-stage Variational Mode Decomposition and Support Vector Regression for Streamflow Forecasting_”, which is currently under review for the journal _Hydrology and Earth System Sciences (HESS)_.

The underlying data of this study is the monthly runoff data sets (1953/01-2018/12) of Huaxian, Xianyang, and Zhangjiashan stations, Wei River, China, which is organized in "_time_series_" directory.

The fundamental code for decomposing runoff data, deciding input predictors and output target, generating machine learning samples, building support vector regression (SVR) models, and evaluating the model performance is organized in the “_tools_” directory. The execution code for forecasting different runoff series using different decomposition algorithms (e.g., variational mode decomposition (VMD), ensemble empirical mode decomposition (EEMD), discrete wavelet transform (DWT), Singular spectrum analysis (SSA) or non-decomposition-based (Orig)) are organized in “projects” directory (e.g., “_huaxian_vmd/projects/_”).

To reproduce the results of this paper, follow the instructions given in readme.md. Note that the same results demonstrated in this paper cannot be reproduced but similar results should be reproduced.


## Open-source software
This study heavily relies on the open-source software. Pandas (McKinney, 2010) and numpy (Stéfan et al., 2011) were used to manage and process streamflow data. Matlab was used to perform variational mode decomposition ([VMD](https://ieeexplore.ieee.org/document/6655981)), ensemble empirical mode decomposition ([EEMD](https://doi.org/10.1142/S1793536909000047)), and discrete wavelet transform ([DWT](https://www.mathworks.com/help/wavelet/ref/dwt.html)) of streamflow and compute the partial autocorrelation coefficient (PACF) of subsignals. The matlab implementations of VMD and EEMD come from Dragomiretskiy and Zosso (2014) and Wu and Huang (2009), respectively. The DWT was performed based on matlab build-in toolbox (“Wavelet 1-D” in “Wavelet Analyzer”). The SSA was performed based on [a python program developed by  Jordan D'Arcy](https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition). The SVR model in [scikit-learn](https://scikit-learn.org/stable/) (Pedregosa et al., 2011) was used to train SVR models. [scikit-optimize](https://scikit-optimize.github.io/) (Tim et al., 2018) was used to tune the SVR models. Matplotlib (Hunter, 2007) was used to draw figures.

## How to validate the research results

1. Clone this repository from Github. Run the following code in CMD or git-bash.

    ```
    git clone https://github.com/zjy8006/MonthlyRunoffForecastByAutoReg
    ```

2. Open MATLAB for decomposing monthly runoff using EEMD, VMD, DWT and MODWT, building ARIMA models and computing PACF of decomposed signal components. Go to the root directory. 

    ```
    % E.g., run the following code in command window of matlab.
    >> cd D:/MonthlyRunoffForecastByAutoReg/tools/
    ```

3. Open this repository with [vscode](https://code.visualstudio.com/) for other tasks. You can run code with [code runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) extension.

## Test shift-variance and sensitivity of appending data and analysis boundary effect for VMD,EEMD,SSA,DWT

* Run [vmd_shift_var_sen_test_sunspot.m](./tools/vmd_shift_var_sen_test_sunspot.m) in Matlab for VMD of [sunspots dataset](http://www.sidc.be/silso/datafiles).
* Run [vmd_shift_var_sen_test_huaxian.m](./tools/vmd_shift_var_sen_test_huaxian.m) in Matlab for VMD of Huaxian monthly runoff. 
* Run [eemd_shift_var_sen_test_huaxian.m](./tools/eemd_shift_var_sen_test_huaxian.m) in Matlab for EEMD of Huaxian Monthly runoff.
* Run [dwt_shift_var_sen_test_huaxian.m](./tools/dwt_shift_var_sen_test_huaxian.m) in Matlab for DWT of Huaxian Monthly runoff.
* Run [ssa_shift_var_sen_test_huaxian.m](./tools/ssa_shift_var_sen_test_huaxian.m) in VSCode for SSA of Huaxian Monthly runoff.

## Monthly runoff decomposition

* Run [RUN_EEMD.mlx](./tools/RUN_EEMD.mlx) in Matlab for EEMD of monthly runoff.
* Run [RUN_VMD.mlx](./tools/RUN_VMD.mlx) in Matlab for VMD of monthly runoff.
* Run [RUN_DWT.mlx](./tools/RUN_DWT.mlx) in Matlab for DWT of monthly runoff.
* Run [RUN_MODWT_WDDFF.mlx](./tools/RUN_MODWT_WDDFF.mlx) in Matlab for BCMODWT of monthly runoff.
* Run [ssa_decompose.py](./tools/ssa_decompose.py) in VSCode for SSA of monthly runoff.

## Determine the input predictors

* Run [compute_pacf.m](./tools/compute_pacf.m) in Matlab for computing PACF.
* Visualizing PACF and config optimal lags in VScode:
  |Station|Decomposition Tools|Visualization Tools|Configuration Tools|
  |---|---|---|---|
  |Huaxian|Non|[plot_pacf.py](./Huaxian/projects/plot_pacf.py)|[variables.py](./Huaxian/Projects/variables.py)|
  |Huaxian|DWT|[plot_pacf.py](./Huaxian_dwt/projects/plot_pacf.py)|[variables.py](./Huaxian_dwt/Projects/variables.py)|
  |Huaxian|EEMD|[plot_pacf.py](./Huaxian_eemd/projects/plot_pacf.py)|[variables.py](./Huaxian_eemd/Projects/variables.py)|
  |Huaxian|SSA|[plot_pacf.py](./Huaxian_ssa/projects/plot_pacf.py)|[variables.py](./Huaxian_ssa/Projects/variables.py)|
  |Huaxian|VMD|[plot_pacf.py](./Huaxian_vmd/projects/plot_pacf.py)|[variables.py](./Huaxian_vmd/Projects/variables.py)|
  |Xianyang|Non|[plot_pacf.py](./Xianyang/projects/plot_pacf.py)|[variables.py](./Xianyang/Projects/variables.py)|
  |Xianyang|DWT|[plot_pacf.py](./Xianyang_dwt/projects/plot_pacf.py)|[variables.py](./Xianyang_dwt/Projects/variables.py)|
  |Xianyang|EEMD|[plot_pacf.py](./Xianyang_eemd/projects/plot_pacf.py)|[variables.py](./Xianyang_eemd/Projects/variables.py)|
  |Xianyang|SSA|[plot_pacf.py](./Xianyang_ssa/projects/plot_pacf.py)|[variables.py](./Xianyang_ssa/Projects/variables.py)|
  |Xianyang|VMD|[plot_pacf.py](./Xianyang_vmd/projects/plot_pacf.py)|[variables.py](./Xianyang_vmd/Projects/variables.py)|
  |Zhangjiashan|Non|[plot_pacf.py](./Zhangjiashan/projects/plot_pacf.py)|[variables.py](./Zhangjiashan/Projects/variables.py)|
  |Zhangjiashan|DWT|[plot_pacf.py](./Zhangjiashan_dwt/projects/plot_pacf.py)|[variables.py](./Zhangjiashan_dwt/Projects/variables.py)|
  |Zhangjiashan|EEMD|[plot_pacf.py](./Zhangjiashan_eemd/projects/plot_pacf.py)|[variables.py](./Zhangjiashan_eemd/Projects/variables.py)|
  |Zhangjiashan|SSA|[plot_pacf.py](./Zhangjiashan_ssa/projects/plot_pacf.py)|[variables.py](./Zhangjiashan_ssa/Projects/variables.py)|
  |Zhangjiashan|VMD|[plot_pacf.py](./Zhangjiashan_vmd/projects/plot_pacf.py)|[variables.py](./Zhangjiashan_vmd/Projects/variables.py)|


## Generate training, development and testing samples [In VSCode]

|Station|Decomposition Tools|Samples generator|
|---|---|---|
|Huaxian|Non|[generate_samples.py](./Huaxian/projects/generate_samples.py)|
|Huaxian|DWT|[generate_samples.py](./Huaxian_dwt/projects/generate_samples.py)|
|Huaxian|EEMD|[generate_samples.py](./Huaxian_eemd/projects/generate_samples.py)|
|Huaxian|SSA|[generate_samples.py](./Huaxian_ssa/projects/generate_samples.py)|
|Huaxian|VMD|[generate_samples.py](./Huaxian_vmd/projects/generate_samples.py)|
|Xianyang|Non|[generate_samples.py](./Xianyang/projects/generate_samples.py)|
|Xianyang|DWT|[generate_samples.py](./Xianyang_dwt/projects/generate_samples.py)|
|Xianyang|EEMD|[generate_samples.py](./Xianyang_eemd/projects/generate_samples.py)|
|Xianyang|SSA|[generate_samples.py](./Xianyang_ssa/projects/generate_samples.py)|
|Xianyang|VMD|[generate_samples.py](./Xianyang_vmd/projects/generate_samples.py)|
|Zhangjiashan|Non|[generate_samples.py](./Zhangjiashan/projects/generate_samples.py)|
|Zhangjiashan|DWT|[generate_samples.py](./Zhangjiashan_dwt/projects/generate_samples.py)|
|Zhangjiashan|EEMD|[generate_samples.py](./Zhangjiashan_eemd/projects/generate_samples.py)|
|Zhangjiashan|SSA|[generate_samples.py](./Zhangjiashan_ssa/projects/generate_samples.py)|
|Zhangjiashan|VMD|[generate_samples.py](./Zhangjiashan_vmd/projects/generate_samples.py)|

## Autoregressive moving average [In Matlab]

* Search space of ARMA models
    |Hyper-parameters|Range|
    |---|---|
    |Autoregression Lags (p)|[1,20]|
    |Moving-average Lags (q)|[1,20]|

* Run [RUN_ARMA.mlx](./tools/RUN_ARMA.mlx) to tune ARMA models.

## Support Vector Regression [In VSCode]

The 'SVR' in [scikit-learn](https://scikit-learn.org/stable/) was used to build support vector regression (SVR) models. The 'gp_minimize' (Bayesian optimization based on Gaussian process) in [scikit-optimize](https://scikit-optimize.github.io/) was used to optimize SVR models. The SVR model optimized by Bayesian optimization are organized in [models.py](./tools/models.py).

* The default optimization ranges of hyperparameters of SVR are set as follows:

    |Hyper-parameters|Range|
    |---|---|
    |C|[0.1,200]|
    |epsilon|[10^-6,1]|
    |gamma|[10^-6,1]|

* Tune SVR models and ensemble forecast results using the following programs.

    |Stations|Decomposition Tools|Tune Files|Ensemble Files|
    |---|---|---|---|
    |Huaxian|Non|[esvr_orig_history.py](./Huaxian/projects/esvr_orig_history.py)|[ensemble_models.py](./Huaxian/projects/ensemble_models.py)|
    |Huaxian|DWT|[esvr_one_step.py](./Huaxian_dwt/projects/esvr_one_step.py) and [esvr_multi_step.py](./Huaxian_dwt/projects/esvr_multi_step.py)|[ensemble_models.py](./Huaxian/projects/ensemble_models.py)|
    |Huaxian|EEMD|[esvr_one_step.py](./Huaxian_eemd/projects/esvr_one_step.py) and [esvr_multi_step.py](./Huaxian_eemd/projects/esvr_multi_step.py)|[ensemble_models.py](./Huaxian/projects/ensemble_models.py)|
    |Huaxian|MODWT|[esvr_one_step.py](./Huaxian_modwt/projects/esvr_one_step.py) and [esvr_multi_step.py](./Huaxian_modwt/projects/esvr_multi_step.py)|[ensemble_models.py](./Huaxian/projects/ensemble_models.py)|
    |Huaxian|SSA|[esvr_one_step.py](./Huaxian_ssa/projects/esvr_one_step.py) and [esvr_multi_step.py](./Huaxian_ssa/projects/esvr_multi_step.py)|[ensemble_models.py](./Huaxian/projects/ensemble_models.py)|
    |Huaxian|VMD|[esvr_one_step.py](./Huaxian_vmd/projects/esvr_one_step.py) and [esvr_multi_step.py](./Huaxian_vmd/projects/esvr_multi_step.py)|[ensemble_models.py](./Huaxian/projects/ensemble_models.py)|
    |Xianyang|Non|[esvr_orig_history.py](./Xianyang/projects/esvr_orig_history.py)|[ensemble_models.py](./Xianyang/projects/ensemble_models.py)|
    |Xianyang|DWT|[esvr_one_step.py](./Xianyang_dwt/projects/esvr_one_step.py) and [esvr_multi_step.py](./Xianyang_dwt/projects/esvr_multi_step.py)|[ensemble_models.py](./Xianyang/projects/ensemble_models.py)|
    |Xianyang|EEMD|[esvr_one_step.py](./Xianyang_eemd/projects/esvr_one_step.py) and [esvr_multi_step.py](./Xianyang_eemd/projects/esvr_multi_step.py)|[ensemble_models.py](./Xianyang/projects/ensemble_models.py)|
    |Xianyang|MODWT|[esvr_one_step.py](./Xianyang_modwt/projects/esvr_one_step.py) and [esvr_multi_step.py](./Xianyang_modwt/projects/esvr_multi_step.py)|[ensemble_models.py](./Xianyang/projects/ensemble_models.py)|
    |Xianyang|SSA|[esvr_one_step.py](./Xianyang_ssa/projects/esvr_one_step.py) and [esvr_multi_step.py](./Xianyang_ssa/projects/esvr_multi_step.py)|[ensemble_models.py](./Xianyang/projects/ensemble_models.py)|
    |Xianyang|VMD|[esvr_one_step.py](./Xianyang_vmd/projects/esvr_one_step.py) and [esvr_multi_step.py](./Xianyang_vmd/projects/esvr_multi_step.py)|[ensemble_models.py](./Xianyang/projects/ensemble_models.py)|
    |Zhangjiashan|Non|[esvr_orig_history.py](./Zhangjiashan/projects/esvr_orig_history.py)|[ensemble_models.py](./Zhangjiashan/projects/ensemble_models.py)|
    |Zhangjiashan|DWT|[esvr_one_step.py](./Zhangjiashan_dwt/projects/esvr_one_step.py) and [esvr_multi_step.py](./Zhangjiashan_dwt/projects/esvr_multi_step.py)|[ensemble_models.py](./Zhangjiashan/projects/ensemble_models.py)|
    |Zhangjiashan|EEMD|[esvr_one_step.py](./Zhangjiashan_eemd/projects/esvr_one_step.py) and [esvr_multi_step.py](./Zhangjiashan_eemd/projects/esvr_multi_step.py)|[ensemble_models.py](./Zhangjiashan/projects/ensemble_models.py)|
    |Zhangjiashan|MODWT|[esvr_one_step.py](./Zhangjiashan_modwt/projects/esvr_one_step.py) and [esvr_multi_step.py](./Zhangjiashan_modwt/projects/esvr_multi_step.py)|[ensemble_models.py](./Zhangjiashan/projects/ensemble_models.py)|
    |Zhangjiashan|SSA|[esvr_one_step.py](./Zhangjiashan_ssa/projects/esvr_one_step.py) and [esvr_multi_step.py](./Zhangjiashan_ssa/projects/esvr_multi_step.py)|[ensemble_models.py](./Zhangjiashan/projects/ensemble_models.py)|
    |Zhangjiashan|VMD|[esvr_one_step.py](./Zhangjiashan_vmd/projects/esvr_one_step.py) and [esvr_multi_step.py](./Zhangjiashan_vmd/projects/esvr_multi_step.py)|[ensemble_models.py](./Zhangjiashan/projects/ensemble_models.py)|

  
## Results Visualization [In VSCode]

* Fig.2 Boundary effects analysis: shift-variance and sensitivity of addition of new data, run [plot_vmd_huaxian_boundary_effect.py](./boundary_effect/plot_vmd_huaxian_boundary_effect.py).
* Fig.3 PCC of validation samples generated from appended decompositions and validation decompositions, run [plot_pcc_of_val_samples.py](./results_analysis/plot_pcc_of_val_samples.py).
* Fig.5 Center frequency alasing, run [plot_aliasing.py](./results_analysis/plot_aliasing.py).
* Fig.6 PACF plot of VMD $IMF_1$ at Huaxian station, run [plot_hua_vmd_pacf.py](./results_analysis/plot_hua_vmd_pacf.py).
* Fig.7 Partial dependence of SVR objective function, run [plot_convergence_objective.py](./results_analysis/plot_convergence_objective.py).
* Fig.8 Evaluation of wavelets and decomposition levels of BCMODWT-SVR, run [plot_wavelet_lev_eva.py](./results_analysis/plot_wavelet_lev_eva.py).
* Fig.9 PCC of signal components, run [plot_pearson_corr_subsignals.py](./results_analysis/plot_pearson_corr_subsignals.py).
* Fig.10 Frequency spectrum of VMD,EEMD,SSA,DWT,BCMODWT, run [plot_frequency_spectrum.py](./results_analysis/plot_frequency_spectrum.py)
* Fig.11 Mutual information between predictors and predicted targets at Huaxian station, run [predictors_targets_mi.py](./results_analysis/predictors_targets_mi.py)
* Fig.12 Evaluate mixing-shuffling step and generating validation samples from appended decompositions for reducing boundary effect, run [plot_boundary_reduction_eva.py](./results_analysis/plot_boundary_reduction_eva.py).
* Fig.13 Evaluate PCA on TSDP models, run [plot_tsdp_pca_metrics.py](./results_analysis/plot_tsdp_pca_metrics.py).
* Fig.14 Evaluation metrics of TSDP and TSDPE models, run [plot_tsdp_tsdpe_metrics.py](./results_analysis/plot_tsdp_tsdpe_metrics.py).
* Scatter plots of TSDP models based on PACF and PCC, run [plot_comp_pacf_pcc_scatters.py](./results_analysis/plot_comp_pacf_pcc_scatters.py).
* Fig.15 Scatter plots of ARMA, SVR, TSDP and WDDFF models, run [plot_tsdp_wddff_scatters.py](./results_analysis/plot_tsdp_wddff_scatters.py).
* Fig.16 Violin plots of ARMA, SVR, TSDP and WDDFF models, run [plot_tsdp_wddff_metrics.py](./results_analysis/plot_tsdp_wddff_metrics.py).


## Cite US

* [BibTex](./hess-2019-565.txt)
* [EndNote](./hess-2019-565.ris)
  
## Reference

* McKinney, W., 2010. Data Structures for Statistical Computing in Python, pp. 51–56.
* Stéfan, v.d.W., Colbert, S.C., Varoquaux, G., 2011. The NumPy Array: A Structure for Efficient Numerical Computation. A Structure for Efficient Numerical Computation. Comput. Sci. Eng. 13 (2), 22–30.
* Dragomiretskiy, K., Zosso, D., 2014. Variational Mode Decomposition. IEEE Trans. Signal Process. 62 (3), 531–544.
* Wu, Z., Huang, N.E., 2009. Ensemble Empirical Mode Decomposition: a Noise-Assisted Data Analysis Method. Adv. Adapt. Data Anal. 01 (01), 1–41.
* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, É., 2011. Scikit-learn. Machine Learning in Python. Journal of Machine Learning Research 12, 2825–2830.
* Tim, H., MechCoder, Gilles, L., Iaroslav, S., fcharras, Zé Vinícius, cmmalone, Christopher, S., nel215, Nuno, C., Todd, Y., Stefano, C., Thomas, F., rene-rex, Kejia, (K.) S., Justus, S., carlosdanielcsantos, Hvass-Labs, Mikhail, P., SoManyUsernamesTaken, Fred, C., Loïc, E., Lilian, B., Mehdi, C., Karlson, P., Fabian, L., Christophe, C., Anna, G., Andreas, M., and Alexander, F.: Scikit-Optimize/Scikit-Optimize: V0.5.2, Zenodo, 2018.
* Hunter, J.D., 2007. Matplotlib. A 2D Graphics Environment. Computing in Science & Engineering 9, 90–95.
