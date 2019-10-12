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

