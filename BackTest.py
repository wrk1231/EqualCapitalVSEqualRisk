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
        """

        :param symbol: Underlying ticket
        :param for_cov: if for covariance computation, need more historical data
        :return: pd.Series, a return time series
        """
        MY_API_KEY = 'J4PS1W9LI8IL0E97'
        ts = TimeSeries(key='MY_API_KEY', output_format='pandas')
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

    def get_all(self, for_cov=False):
        """

        :param for_cov: if for covariance computation, need more data
        :return:
        """
        data_dict = dict()
        for each_ticket in self.tickets:
            data_dict.update({each_ticket: self.get_daily_close(symbol=each_ticket, for_cov=for_cov)})
        df = pd.concat(data_dict, axis=1)
        df = df[self.tickets]
        return df

    def get_returns(self, data_all):
        """

        :param data_all: a well-defined close price time series
        :return:
        """
        returns = data_all.pct_change()
        returns.dropna(inplace=True)

        return returns

    def get_returns_mul_factor(self, returns):
        """

        :param returns: a return time series
        :return: realized return
        """
        returns_mul_factor = returns + 1

        return returns_mul_factor

    def __init__(self, tickets=['SPY', 'TLT', 'GLD'], cov_frequency=90, rebalance_frequency=60, get_records = False, fee=0.0005):

        self.ONEYEARDAYS = 252

        self.cov_begin_date = datetime(2005, 1, 3).date()
        self.ret_begin_date = datetime(2006, 1, 3).date()
        self.tickets = tickets
        self.securities_num = len(self.tickets)

        ## For return aggregation
        self.data_all = self.get_all(for_cov=False)
        self.returns = self.get_returns(self.data_all)
        self.returns_mul_factor = self.get_returns_mul_factor(self.returns)

        self.day_counts = len(self.returns)
        self.calendar = self.returns.index.values

        ## For calculating historical covariance, need more historical data to yield the result

        self.data_all_c = None
        self.returns_c = None

        self.cov_frequency = cov_frequency
        self.rebalance_frequency = rebalance_frequency

        self.backtest_result = {}
        self.risk_contribution = {}
        self.backtest_weight = {}



    def backtest(self):
        """
        To be implemented in subclass
        :return:
        """
        pass

    def MaxDD(self, result, get_pair = False):
        i = np.argmax(np.maximum.accumulate(result) - result)  # end of the period
        j = np.argmax(result[:i])  # start of period
        if get_pair == False:
            return (result[i] - result[j]) / result[j]
        else:
            return (j,i)

    def performance_analysis(self, backtest_result):
        strategy_return = backtest_result.pct_change().dropna()
        strategy_stats = pd.DataFrame()
        ##Compound Annual Growth Rate
        N = len(backtest_result) / ONEYEARDAYS
        strategy_stats['CAGR(%)'] = (np.power(backtest_result.iloc[-1] / backtest_result.iloc[0], 1 / N) - 1) * 100
        strategy_stats['Annualized Returns(%)'] = strategy_return.mean() * ONEYEARDAYS * 100
        strategy_stats['Annualized Volatility(%)'] = strategy_return.std() * np.sqrt(ONEYEARDAYS) * 100
        strategy_stats['Sharpe Ratio'] = strategy_stats['Annualized Returns(%)'] / strategy_stats[
            'Annualized Volatility(%)']
        mdd = {}
        for colName in backtest_result.columns:
            mdd.update({colName: MaxDD(backtest_result[colName].values)})

        strategy_stats['MaxDD (%)'] = pd.Series(mdd)

        return strategy_stats

    def get_cov(self):
        """
        Data API has limit for calling 5 times per minute
        :return:
        """
        self.data_all_c = self.get_all(for_cov=True)
        self.returns_c = self.get_returns(self.data_all_c)
        self.cov_series = self.returns_c.rolling(window=self.cov_frequency).cov() * self.ONEYEARDAYS

class ERBACKTEST(BACKTEST):
    def __init__(self, tickets=['SPY', 'TLT', 'GLD'], cov_frequency=90, rebalance_frequency=60, get_records=False,
                 fee=0.0005):
        super().__init__(tickets=['SPY', 'TLT', 'GLD'], cov_frequency=90, rebalance_frequency=60, get_records=False,
                         fee=0.0005)

    def equal_risk(self, weight, cov):
        """

        :param weight:
        :param cov:
        :return:
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

        return np.square(TRC1 - TRC2) + np.square(TRC2 - TRC3) + np.square(TRC1 - TRC3)

    def TRC(self, weight, cov):
        """
        Total Risk Contribution / Variance of Portfolio = partial risk contribution
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

    def weight_calculation(self, cov_df):
        ans = minimize(self.equal_risk, [0.3, 0.3], (cov_df), method='L-BFGS-B', bounds=((0, 1), (0, 1)))
        result = []
        result.extend(list(ans.x))
        result.append(1 - ans.x.sum())
        return np.array(result)

    def backtest(self, cov_frequency = None, rebalance_frequency = None, get_records = False, fee = 0.0005):
        """

        :param cov_frequency: the frequency for calculating covariance matrix
        :param rebalance_frequency:  the frequency for rebalancing our portfolio
        :param get_records: if we need to track the historical weight & partial risk contribution or not
        :param fee: the total fee rate in our rebalancing, total fee = absolute change of our position * fee rate
        :return:
        """
        ## Create new backtest equal risk result
        self.backtest_result = {}
        self.risk_contribution = pd.DataFrame()
        self.backtest_weight = pd.DataFrame()

        if cov_frequency == None:
            cov_frequency = self.cov_frequency
            self.get_cov()
            cov_series = self.cov_series
        else:
            self.get_cov()
            cov_series = self.returns_c.rolling(window=cov_frequency).cov() * self.ONEYEARDAYS

        if rebalance_frequency == None:
            rebalance_frequency = self.rebalance_frequency

        rebalance_cov = []  ## [{date : covariance},...]
        rebalance_times = self.day_counts // rebalance_frequency

        for i in range(1, rebalance_times + 1):
            ## Indexing the covariance matrix, in pair of (date, covariance matrix)
            date = self.calendar[rebalance_frequency * i - 1]
            rebalance_cov.append((date, cov_series.loc[date]))

            ## Get rebalancing date and weight
        rebalance_pair = {}  ## {date : weight}
        for i in range(len(rebalance_cov)):
            res = self.weight_calculation(rebalance_cov[i][1])
            rebalance_pair.update({rebalance_cov[i][0]: res})

        rebalance_pair = pd.Series(rebalance_pair)
        rebalance_date = rebalance_pair.index.values
        rebalance_weight = rebalance_pair.values
        new_calendar = self.calendar[self.calendar > rebalance_date[0]]

        cov_pair = {}  ## {date : covariance}
        for date in new_calendar:
            cov_pair.update({date: cov_series.loc[date]})

        current_asset = rebalance_weight[0] * 1000
        current_weight = rebalance_weight[0]
        self.backtest_weight.update({rebalance_date[0]: current_weight})
        current_cov = cov_pair.get(rebalance_date[0])
        ## Invest 1000 unit at the first day
        self.backtest_result.update({rebalance_date[0]: 1000})

        for date in new_calendar:

            if date not in rebalance_date:
                ## at the end of day, market closed (we use close level to calculate return)
                current_asset = current_asset * (self.returns_mul_factor.loc[date].values)
                current_level = current_asset.sum()
                current_weight = current_asset / current_level
                self.backtest_weight[date] = current_weight
                self.backtest_result.update({date: current_level})
                current_cov = cov_pair.get(date)
                self.risk_contribution[date] = self.TRC(current_weight, current_cov)

            elif date in rebalance_date:
                ## at rebalance day, we hold till market close, get rolling covariance matrix, then rebalance
                ## Forward bias: rebalance should be done before market close,
                ## while we calculate our rebalance weight using the close level of that day.
                ## If market has no big move near close it's OK.
                current_asset = current_asset * (self.returns_mul_factor.loc[date].values)
                current_level = current_asset.sum()
                ## Get new rebalance weight, deduct general fee
                previous_asset = current_asset
                current_asset = rebalance_pair.get(date) * current_level
                current_asset -= np.abs(previous_asset - current_asset) * fee
                current_level -= (np.abs(previous_asset - current_asset) * fee).sum()
                self.backtest_result.update({date: current_level})
                ## Calculate Total Risk Contribution
                current_weight = rebalance_pair.get(date)
                self.backtest_weight[date] = current_weight
                current_cov = cov_pair.get(date)  ## Covariance based on close of the rebalance day
                self.risk_contribution[date] = self.TRC(current_weight, current_cov)

        self.backtest_result = pd.Series(self.backtest_result)
        self.risk_contribution = self.risk_contribution.T
        self.backtest_weight = self.backtest_weight.T
        self.backtest_weight.index.name = 'Date'
        self.backtest_result.index.name = 'Date'
        self.risk_contribution.index.name = 'Date'

        if get_records == True:
            return self.backtest_result, self.risk_contribution, self.backtest_weight
        else:
            return self.backtest_result



class ECBACKTEST(BACKTEST):
    def __init__(self, tickets=['SPY', 'TLT', 'GLD'], cov_frequency=90, rebalance_frequency=60, get_records=False,
                 fee=0.0005):
        super().__init__(tickets=['SPY', 'TLT', 'GLD'], cov_frequency=90, rebalance_frequency=60, get_records=False,
                         fee=0.0005)

        self.equal_capital_weight = np.array([1,1,1])/3

    def backtest(self, cov_frequency=None, rebalance_frequency=None, get_records=False, fee=0.0005):
        """

        :param cov_frequency: the frequency for calculating covariance matrix
        :param rebalance_frequency:  the frequency for rebalancing our portfolio
        :param get_records: if we need to track the historical weight & partial risk contribution or not
        :param fee: the total fee rate in our rebalancing, total fee = absolute change of our position * fee rate
        :return:
        """
        ## Create new backtest equal risk result
        self.backtest_result = {}
        self.risk_contribution = pd.DataFrame()
        self.backtest_weight = pd.DataFrame()

        if cov_frequency == None:
            cov_frequency = self.cov_frequency
            self.get_cov()
            cov_series = self.cov_series
        else:
            self.get_cov()
            cov_series = self.returns_c.rolling(window=cov_frequency).cov() * self.ONEYEARDAYS

        if rebalance_frequency == None:
            rebalance_frequency = self.rebalance_frequency

        ## Generate rebalance calendar
        rebalance_times = self.day_counts // rebalance_frequency
        rebalance_date = []
        for i in range(1, rebalance_times + 1):
            date = self.calendar[rebalance_frequency * i - 1]
            rebalance_date.append(date)
        new_calendar = self.calendar[self.calendar > rebalance_date[0]]

        cov_pair = {}  ## {date : covariance}
        for date in new_calendar:
            cov_pair.update({date: cov_series.loc[date]})

        ## Initialization
        current_asset = self.equal_capital_weight * 1000
        self.backtest_weight.update({rebalance_date[0]: self.equal_capital_weight})
        current_cov = cov_pair.get(rebalance_date[0])
        ## Invest 1000 unit at the first day
        self.backtest_result.update({rebalance_date[0]: 1000})

        for date in new_calendar:

            if date not in rebalance_date:
                ## at the end of day, market closed (we use close level to calculate return)
                current_asset = current_asset * (self.returns_mul_factor.loc[date].values)
                current_level = current_asset.sum()
                current_weight = current_asset / current_level
                self.backtest_weight[date] = current_weight
                self.backtest_result.update({date: current_level})
                current_cov = cov_pair.get(date)
                self.risk_contribution[date] = self.TRC(current_weight, current_cov)

            elif date in rebalance_date:
                ## at rebalance day, we hold till market close, get rolling covariance matrix, then rebalance
                ## Forward bias: rebalance should be done before market close,
                ## while we calculate our rebalance weight using the close level of that day.
                ## If market has no big move near close it's OK.
                current_asset = current_asset * (self.returns_mul_factor.loc[date].values)
                current_level = current_asset.sum()
                ## Get new rebalance weight, deduct general fee
                previous_asset = current_asset
                current_asset = self.equal_capital_weight * current_level
                current_asset -= np.abs(previous_asset - current_asset) * fee
                current_level -= (np.abs(previous_asset - current_asset) * fee).sum()
                current_weight = self.equal_capital_weight
                self.backtest_result.update({date: current_level})
                self.backtest_weight[date] = current_weight
                current_cov = cov_pair.get(date)  ## Covariance based on close of the rebalance day
                self.risk_contribution[date] = self.TRC(current_weight, current_cov)

        self.backtest_result = pd.Series(self.backtest_result)
        self.risk_contribution = self.risk_contribution.T
        self.backtest_weight = self.backtest_weight.T
        self.backtest_result.index.name = 'Date'
        self.backtest_weight.index.name = 'Date'
        self.risk_contribution.index.name = 'Date'

        if get_records == True:
            return self.backtest_result, self.risk_contribution, self.backtest_weight
        else:
            return self.backtest_result

def run(argRun):
    """
    For running in the command line, need to take care of it, our data-api only allow 5 calls per minute
    :param argRun:
    :return:
    """
    if argRun.get('strategy', None) == 'EC':
        S = ECBACKTEST(tickets= argRun.get('tickets',['SPY', 'TLT', 'GLD']),
                       cov_frequency=argRun.get('cov_frequency',60),
                       rebalance_frequency=argRun.get('rebalance_frequency',30),
                       get_records= argRun.get('get_records', False),
                       fee = argRun.get('fee', 0.0005)
                       )
        res = S.backtest()
        return res

    elif argRun.get('strategy', None) == 'ER':
        S = ERBACKTEST(tickets= argRun.get('tickets',['SPY', 'TLT', 'GLD']),
                       cov_frequency=argRun.get('cov_frequency',60),
                       rebalance_frequency=argRun.get('rebalance_frequency',30),
                       get_records= argRun.get('get_records', False),
                       fee = argRun.get('fee', 0.0005)
                       )
        res = S.backtest()
        return res

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Strategy')
    parser.add_argument('-s', '--strategy', dest='strategy', choices={'ER', 'EC'},
                        help='Choose Equal Risk (ER) or Equal Capital (EC)')
    parser.add_argument('-t', '--tickets', dest='tickets', help='Assets used to construct the portfolio', nargs='*', default=['SPY', 'TLT', 'GLD'])
    parser.add_argument('--cov_frequency', dest='cov_frequency', type=int, help='Covariance Matrix Calculation Frequency')
    parser.add_argument('--rebalance_frequency', dest='rebalance_frequency', type=int, help='Rebalancing Frequency')
    parser.add_argument('--get_records', dest='get_records',  choices={1, 0}, type=bool, default=False,
                        help='If get historical weight and risk contribution or not')
    parser.add_argument('--fee', dest='fee', type=float, default=0.0005,
                        help='total cost of trades')
    args = parser.parse_args()
    argRun = vars(args)
    print(run(argRun))