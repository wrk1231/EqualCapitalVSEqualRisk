# EqualCapitalVSEqualRisk

## Requirements:
Anaconda Environment of Python 3.7 + alpha-vantage 2.1.1

## Python module:

#### BACKTEST is the basic class

#### 2 sub-classes: ERBACKTEST and ECBACKTEST for 2 different strategies: Equal Risk and Equal Capital

#### After initialization just run backtest function

#### Better to import the module and run initialization and backtest separately, the data-api we use only supports 5 calls perminute.

#### Run in command line will cause data-api warning for too many queries in a short time

## Jupyter Notebook can be loaded in Google Colab

#### Chrome add-in 'Open in Colab' can help do that, or save to Google Drive then open as Colab Project
