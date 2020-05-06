from statsmodels.tsa.stattools import adfuller
import pandas as pd 
import matplotlib.pyplot as plt
import os
root = os.path.abspath(os.path.dirname("__file__"))

def is_stationary(timeseries,confidence=0.95):
    """
    Judge a time series is stationary or non-stationary using adf test
    ADF test whether a time series has unit root. If a time series has
    unit root, it is non-stationary. The null hypothesis is the given time
    series has unit root (i.e.,non-stationary).
    An example of adf test result:
    (-7.460873875968257, 5.3582687026296535e-11, 33, 5226, 
    {'1%': -3.4316019163653446, '5%': -2.862093216924214, '10%': -2.5670644771344784}, 
    55423.91130326275)
    The first value is adf test result, T.
    The second value is probability (P) corresponding to T.
    The third value is delay.
    The fourth value is the number of test.
    The fifth value is the standard test (ST) with different confidence level.
    If T is smaller than ST for all three confidence level, 
    then reject the null hypothesis (non-stationary),.
    """
    is_stationary = False
    result = adfuller(timeseries)
    T=result[0]
    P=result[1]
    ST1=result[4]['1%']
    ST5=result[4]['5%']
    ST10=result[4]['10%']
    plt.figure()
    plt.scatter(1,T,label='T')
    plt.axhline(y=P,xmin=0,xmax=2,c='purple',label='probability')
    plt.axhline(y=ST10,xmin=0,xmax=2,c='g',label='confidence:90%')
    plt.axhline(y=ST5,xmin=0,xmax=2,c='b',label='confidence:95%')
    plt.axhline(y=ST1,xmin=0,xmax=2,c='r',label='confidence:99%')
    plt.xlim(0,2)
    plt.legend()
    plt.show()
    
    if confidence==0.99:
        if T < ST1:
            is_stationary = True
    elif confidence==0.95:
        if T < ST5:
            is_stationary = True
    elif confidence==0.90:
        if T < ST10:
            is_stationary = True
    return is_stationary



if __name__ == "__main__":
    data = pd.read_csv(root+'/time_series/MonthlyRunoffWeiRiver.csv')
    huaxian = data['Huaxian']
    Xianyang = data['Xianyang']
    Zhangjiashan = data['Zhangjiashan']
    print(is_stationary(huaxian))
    print(is_stationary(Xianyang))
    print(is_stationary(Zhangjiashan))