import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
plt.style.use('ggplot')
import os
import scipy
from scipy import optimize
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class BACKTEST(object):

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

    def get_all(self, for_cov = False):
        data_dict = dict()
        for each_ticket in self.tickets:
            data_dict.update({each_ticket: self.get_daily_close(symbol=each_ticket, for_cov=for_cov)})
        df =  pd.concat(data_dict, axis = 1)
        df = df[self.tickets]
        return df

    def get_returns(self, data_all):
        returns = data_all.pct_change()
        returns.dropna(inplace=True)

        return returns

    def get_returns_mul_factor(self, returns):
        returns_mul_factor = returns + 1

        return returns_mul_factor

    def __init__(self, tickets=['SPY', 'TLT', 'GLD'], cov_frequency = 90, rebalance_frequency = 60, get_records = False, fee = 0.0005):

        self.MY_API_KEY = 'J4PS1W9LI8IL0E97'
        self.ONEYEARDAYS = 252
        self.cov_begin_date = datetime(2005, 1, 3).date()
        self.ret_begin_date = datetime(2006, 1, 3).date()
        self.tickets = tickets
        self.securities_num = len(self.tickets)
        self.data_all = self.get_all(for_cov=False)
        self.returns = self.get_returns(self.data_all)
        self.returns_mul_factor = self.get_returns_mul_factor(self.returns)
        self.data_all_c = self.get_all(for_cov=True)
        self.returns_c = self.get_returns(self.data_all_c)

        self.cov_frequency = cov_frequency
        self.rebalance_frequency = rebalance_frequency
        self.get_records = get_records
        self.fee = fee
        self.cov_series = self.returns_c.rolling(window=cov_frequency).cov() * self.ONEYEARDAYS







class ERBACKTEST(BACKTEST):
    def equal_risk(weight, cov):
        TRC1 = weight[0] ** 2 * cov.iloc[0, 0] + \
               weight[0] * weight[1] * cov.iloc[1, 0] + \
               weight[0] * (1 - weight[0] - weight[1]) * cov.iloc[2, 0]

        TRC2 = weight[0] * weight[1] * cov.iloc[0, 1] + \
               weight[1] ** 2 * cov.iloc[1, 1] + \
               weight[1] * (1 - weight[0] - weight[1]) * cov.iloc[2, 1]

        TRC3 = weight[0] * (1 - weight[0] - weight[1]) * cov.iloc[0, 2] + \
               weight[1] * (1 - weight[0] - weight[1]) * cov.iloc[1, 2] + \
               (1 - weight[0] - weight[1]) ** 2 * cov.iloc[2, 2]

        return np.square(TRC1 - TRC2) + np.square(TRC2 - TRC3) + np.square(TRC1 - TRC3)

    def TRC(weight, cov):
        """
        Total Risk Contribution
        """
        TRC1 = weight[0] ** 2 * cov.iloc[0, 0] + \
               weight[0] * weight[1] * cov.iloc[1, 0] + \
               weight[0] * (1 - weight[0] - weight[1]) * cov.iloc[2, 0]

        TRC2 = weight[0] * weight[1] * cov.iloc[0, 1] + \
               weight[1] ** 2 * cov.iloc[1, 1] + \
               weight[1] * (1 - weight[0] - weight[1]) * cov.iloc[2, 1]

        TRC3 = weight[0] * (1 - weight[0] - weight[1]) * cov.iloc[0, 2] + \
               weight[1] * (1 - weight[0] - weight[1]) * cov.iloc[1, 2] + \
               (1 - weight[0] - weight[1]) ** 2 * cov.iloc[2, 2]

        Var_Portfolio = np.dot(np.dot(weight, cov), weight.T)

        return np.array([TRC1, TRC2, TRC3]) / Var_Portfolio

    def weight_calculation(cov_df):
        ans = minimize(equal_risk, [0.0, 0.0], (cov_df), method='L-BFGS-B', bounds=((0, 1), (0, 1)))
        result = []
        result.extend(list(ans.x))
        result.append(1 - ans.x.sum())
        return np.array(result)


class ECBACKTEST(BACKTEST):
    pass
