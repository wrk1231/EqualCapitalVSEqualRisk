import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.style.use('ggplot')
import os
import scipy
from scipy import optimize
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class BACKTEST(object):
    def __init__(self):
        self.MY_API_KEY = 'J4PS1W9LI8IL0E97'
        self.ONEYEARDAYS = 252
        self.cov_begin_date = datetime(2005, 1, 3).date()
        self.ret_begin_date = datetime(2006, 1, 3).date()

    def get_daily_close(self, symbol, for_cov=False):
        full_data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        series_index = pd.Series(full_data.index)
        full_data.index = series_index.apply(lambda t: np.datetime64(t).astype(datetime).date()).values
        assert '4. close' in full_data.columns
        close = full_data['4. close']
        close.index.name = 'Date'
        close.name = symbol
        assert close.index[0] < datetime(2005, 1, 3).date()
        if for_cov:
            return close[self.cov_begin_date:]
        else:
            return close[self.ret_begin_date:]

class ERBACKTEST(BACKTEST):
    pass

class ECBACKTEST(BACKTEST):
    pass