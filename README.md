
# * **`WORK IN PROGRESS DUE 10/10/2017`** *

# **EUR/USD Foreign Exchange Rate Prediction**

I am using logistic regression, neural networks, and boosted trees to predict the future direction of of the EUR/USD foreign exchange rates.

LINK TO PREDICTION WEBSITE

## Table of Contents
1. [Data](#data)
2. [Target](#target)
3. [Features](#features)
4. [Modeling](#modeling)
5. [Results](#results)
6. [Web Application](#web-application)
7. [Resources](#resources)

# Data

I used the Oanda API to download historical EUR_USD candles to a PostgreSQL database. The data contains the volume, open, high, low, close mid prices (between bid / ask prices). The API only allows you to receive 5,000 records per request so I setup a script to download this information overnight. The database contains tables with the exchange price every 5 seconds, 10 seconds, 15 seconds, etc. from 2005 to today as shown by the table below.

Granularity | Description
--- | ---
S5 | 5 second candlesticks, minute alignment
S10 | 10 second candlesticks, minute alignment
S15 | 15 second candlesticks, minute alignment
S30 | 30 second candlesticks, minute alignment
M1 | 1 minute candlesticks, minute alignment
M2 | 2 minute candlesticks, hour alignment
M4 | 4 minute candlesticks, hour alignment
M5 | 5 minute candlesticks, hour alignment
M10 | 10 minute candlesticks, hour alignment
M15 | 15 minute candlesticks, hour alignment
M30 | 30 minute candlesticks, hour alignment
H1 | 1 hour candlesticks, hour alignment
H2 | 2 hour candlesticks, day alignment
H3 | 3 hour candlesticks, day alignment
H4 | 4 hour candlesticks, day alignment
H6 | 6 hour candlesticks, day alignment
H8 | 8 hour candlesticks, day alignment
H12 | 12 hour candlesticks, day alignment
D | 1 day candlesticks, day alignment
W | 1 week candlesticks, aligned to start of week
M | 1 month candlesticks, aligned to first day of the month

#### 15 Minute Candles Example
| time | volume | open | high | low | close | complete |
|---------|--------|----------|----------|----------|----------|----------|
| 6:45:00 | 473 | 1.346250 | 1.348050 | 1.345950 | 1.348050 | True |
| 7:00:00 | 481 | 1.347950 | 1.348250 | 1.347350 | 1.348150 | True |
| 7:15:00 | 303 | 1.348150 | 1.348350 | 1.347300 | 1.347900 | True |
| 7:30:00 | 290 | 1.348000 | 1.350850 | 1.348000 | 1.350750 | True |
| 7:45:00 | 373 | 1.350650 | 1.353250 | 1.350250 | 1.352800 | True |
| 8:00:00 | 290 | 1.352800 | 1.354700 | 1.352500 | 1.352500 | True |
| 8:15:00 | 219 | 1.352400 | 1.353000 | 1.351250 | 1.351570 | True |
| 8:30:00 | 206 | 1.351670 | 1.351900 | 1.350470 | 1.351700 | True |
| 8:45:00 | 186 | 1.351750 | 1.353270 | 1.351750 | 1.353070 | True |
| 9:00:00 | 462 | 1.352970 | 1.353070 | 1.351420 | 1.352820 | True |

### 1 Month Candles with Volume
![alttext](/imgs/1monthcandles.png "Forex Candles")

http://developer.oanda.com/rest-live-v20/introduction/

Future Opportunities: Incorporate more data, economic calendar, historical position ratios, bid ask spreads, commitments of traders, orderbook, tick data...

# Target

I am using a binary classification target (1 / 0) of whether or not the close price of the next candle is higher or lower than the close price of the current candle. I also calculated the log returns log(close n+1 / close n) that are used for calculating returns.

| time | volume | open | high | low | close | log_returns | log_returns_shifted | target |
|----------|--------|----------|----------|----------|----------|-------------|---------------------|--------|
| 18:15:00 | 1 | 1.356000 | 1.356000 | 1.356000 | 1.356000 | nan | 0.000000 | 1 |
| 18:30:00 | 1 | 1.356000 | 1.356000 | 1.356000 | 1.356000 | 0.000000 | 0.000516 | 1 |
| 18:45:00 | 4 | 1.356700 | 1.356800 | 1.356500 | 1.356700 | 0.000516 | 0.000147 | 1 |
| 19:00:00 | 5 | 1.356900 | 1.357000 | 1.356900 | 1.356900 | 0.000147 | -0.000959 | 0 |
| 19:15:00 | 27 | 1.356500 | 1.356900 | 1.355600 | 1.355600 | -0.000959 | 0.000221 | 1 |
| 19:30:00 | 6 | 1.355600 | 1.356500 | 1.355600 | 1.355900 | 0.000221 | -0.000074 | 0 |
| 19:45:00 | 6 | 1.356000 | 1.356000 | 1.355800 | 1.355800 | -0.000074 | 0.000000 | 1 |
| 20:00:00 | 12 | 1.355900 | 1.356100 | 1.355800 | 1.355800 | 0.000000 | 0.000000 | 1 |
| 20:15:00 | 15 | 1.355700 | 1.355900 | 1.355600 | 1.355800 | 0.000000 | 0.000148 | 1 |
| 20:30:00 | 27 | 1.355900 | 1.356000 | 1.355800 | 1.356000 | 0.000148 | -0.001033 | 0 |

### Log Returns Distribution vs. Arithmetic Returns Distribution
![alttext](/imgs/returns.png "Forex Returns")

Future Opportunities: Incorporate regression not just classification.

# Features

Below are the technical analysis features that were added to the data.

### Technical Indicators without Parameters

| Group | Short Name | Name | Parameters | Output |
|-----------------------|--------------|---------------------------------------------|------------|-----------------------|
| Momentum Indicators | BOP | Balance Of Power | [] | [real] |
| Overlap Studies | HT_TRENDLINE | Hilbert Transform - Instantaneous Trendline | [] | [real] |
| Volume Indicators | AD | Chaikin A/D Line | [] | [real] |
| Volume Indicators | OBV | On Balance Volume | [] | [real] |
| Cycle Indicators | HT_DCPERIOD | Hilbert Transform - Dominant Cycle Period | [] | [real] |
| Cycle Indicators | HT_DCPHASE | Hilbert Transform - Dominant Cycle Phase | [] | [real] |
| Cycle Indicators | HT_PHASOR | Hilbert Transform - Phasor Components | [] | [inphase, quadrature] |
| Cycle Indicators | HT_SINE | Hilbert Transform - SineWave | [] | [sine, leadsine] |
| Cycle Indicators | HT_TRENDMODE | Hilbert Transform - Trend vs Cycle Mode | [] | [integer] |
| Volatility Indicators | TRANGE | True Range | [] | [real] |

### Technical Indicators with Only Timeperiod Parameter

I used each technical indicator with inputs 5, 15, 25, 35, and 45.

| Group | Short Name | Name | Parameters | Output |
|-----------------------|---------------------|---------------------------------------------------|------------------|----------------------|
| Momentum Indicators | ADX | Average Directional Movement Index | [timeperiod: 14] | [real] |
| Momentum Indicators | ADXR | Average Directional Movement Index Rating | [timeperiod: 14] | [real] |
| Momentum Indicators | AROON | Aroon | [timeperiod: 14] | [aroondown, aroonup] |
| Momentum Indicators | AROONOSC | Aroon Oscillator | [timeperiod: 14] | [real] |
| Momentum Indicators | CCI | Commodity Channel Index | [timeperiod: 14] | [real] |
| Momentum Indicators | CMO | Chande Momentum Oscillator | [timeperiod: 14] | [real] |
| Momentum Indicators | DX | Directional Movement Index | [timeperiod: 14] | [real] |
| Momentum Indicators | MFI | Money Flow Index | [timeperiod: 14] | [real] |
| Momentum Indicators | MINUS_DI | Minus Directional Indicator | [timeperiod: 14] | [real] |
| Momentum Indicators | MINUS_DM | Minus Directional Movement | [timeperiod: 14] | [real] |
| Momentum Indicators | MOM | Momentum | [timeperiod: 10] | [real] |
| Momentum Indicators | PLUS_DI | Plus Directional Indicator | [timeperiod: 14] | [real] |
| Momentum Indicators | PLUS_DM | Plus Directional Movement | [timeperiod: 14] | [real] |
| Momentum Indicators | ROC | Rate of change : ((price/prevPrice)-1)*100 | [timeperiod: 10] | [real] |
| Momentum Indicators | ROCP | Rate of change Percentage: (price-prevPrice)/p... | [timeperiod: 10] | [real] |
| Momentum Indicators | ROCR | Rate of change ratio: (price/prevPrice) | [timeperiod: 10] | [real] |
| Momentum Indicators | ROCR100 | Rate of change ratio 100 scale: (price/prevPri... | [timeperiod: 10] | [real] |
| Momentum Indicators | RSI | Relative Strength Index | [timeperiod: 50] | [real] |
| Momentum Indicators | TRIX | 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA | [timeperiod: 30] | [real] |
| Momentum Indicators | WILLR | Williams' %R | [timeperiod: 14] | [real] |
| Overlap Studies | DEMA | Double Exponential Moving Average | [timeperiod: 30] | [real] |
| Overlap Studies | EMA | Exponential Moving Average | [timeperiod: 30] | [real] |
| Overlap Studies | KAMA | Kaufman Adaptive Moving Average | [timeperiod: 30] | [real] |
| Overlap Studies | MIDPOINT | MidPoint over period | [timeperiod: 14] | [real] |
| Overlap Studies | MIDPRICE | Midpoint Price over period | [timeperiod: 14] | [real] |
| Overlap Studies | SMA | Simple Moving Average | [timeperiod: 30] | [real] |
| Overlap Studies | TEMA | Triple Exponential Moving Average | [timeperiod: 30] | [real] |
| Overlap Studies | TRIMA | Triangular Moving Average | [timeperiod: 30] | [real] |
| Overlap Studies | WMA | Weighted Moving Average | [timeperiod: 30] | [real] |
| Volatility Indicators | ATR | Average True Range | [timeperiod: 14] | [real] |
| Volatility Indicators | NATR | Normalized Average True Range | [timeperiod: 14] | [real] |
| Statistic Functions | BETA | Beta | [timeperiod: 5] | [real] |
| Statistic Functions | CORREL | Pearson's Correlation Coefficient (r) | [timeperiod: 30] | [real] |
| Statistic Functions | LINEARREG | Linear Regression | [timeperiod: 14] | [real] |
| Statistic Functions | LINEARREG_ANGLE | Linear Regression Angle | [timeperiod: 14] | [real] |
| Statistic Functions | LINEARREG_INTERCEPT | Linear Regression Intercept | [timeperiod: 14] | [real] |
| Statistic Functions | LINEARREG_SLOPE | Linear Regression Slope | [timeperiod: 14] | [real] |
| Statistic Functions | TSF | Time Series Forecast | [timeperiod: 14] | [real] |

### Technical Indicators with more than 1 Parameter

| Group | Short Name | Name | Parameters | Output |
|---------------------|------------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| Momentum Indicators | APO | Absolute Price Oscillator | ['fastperiod: 12', 'slowperiod: 26', 'matype: 0'] | ['real'] |
| Momentum Indicators | MACD | Moving Average Convergence/Divergence | ['fastperiod: 12', 'slowperiod: 26', 'signalperiod: 9'] | ['macd', 'macdsignal', 'macdhist'] |
| Momentum Indicators | MACDEXT | MACD with controllable MA type | ['fastperiod: 12', 'fastmatype: 0', 'slowperiod: 26', 'slowmatype: 0', 'signalperiod: 9', 'signalmatype: 0'] | ['macd', 'macdsignal', 'macdhist'] |
| Momentum Indicators | MACDFIX | Moving Average Convergence/Divergence Fix 12/26 | ['signalperiod: 9'] | ['macd', 'macdsignal', 'macdhist'] |
| Momentum Indicators | PPO | Percentage Price Oscillator | ['fastperiod: 12', 'slowperiod: 26', 'matype: 0'] | ['real'] |
| Momentum Indicators | STOCH | Stochastic | ['fastk_period: 5', 'slowk_period: 3', 'slowk_matype: 0', 'slowd_period: 3', 'slowd_matype: 0'] | ['slowk', 'slowd'] |
| Momentum Indicators | STOCHF | Stochastic Fast | ['fastk_period: 5', 'fastd_period: 3', 'fastd_matype: 0'] | ['fastk', 'fastd'] |
| Momentum Indicators | STOCHRSI | Stochastic Relative Strength Index | ['timeperiod: 14', 'fastk_period: 5', 'fastd_period: 3', 'fastd_matype: 0'] | ['fastk', 'fastd'] |
| Momentum Indicators | ULTOSC | Ultimate Oscillator | ['timeperiod1: 7', 'timeperiod2: 14', 'timeperiod3: 28'] | ['real'] |
| Overlap Studies | BBANDS | Bollinger Bands | ['timeperiod: 5', 'nbdevup: 2', 'nbdevdn: 2', 'matype: 0'] | ['upperband', 'middleband', 'lowerband'] |
| Overlap Studies | MA | Moving average | ['timeperiod: 30', 'matype: 0'] | ['real'] |
| Overlap Studies | MAMA | MESA Adaptive Moving Average | ['fastlimit: 0.5', 'slowlimit: 0.05'] | ['mama', 'fama'] |
| Overlap Studies | SAR | Parabolic SAR | ['acceleration: 0.02', 'maximum: 0.2'] | ['real'] |
| Overlap Studies | SAREXT | Parabolic SAR - Extended | ['startvalue: 0', 'offsetonreverse: 0', 'accelerationinitlong: 0.02', 'accelerationlong: 0.02', 'accelerationmaxlong: 0.2', 'accelerationinitshort: 0.02', 'accelerationshort: 0.02', 'accelerationmaxshort: 0.2'] | ['real'] |
| Overlap Studies | T3 | Triple Exponential Moving Average (T3) | ['timeperiod: 5', 'vfactor: 0.7'] | ['real'] |
| Volume Indicators | ADOSC | Chaikin A/D Oscillator | ['fastperiod: 3', 'slowperiod: 10'] | ['real'] |
| Statistic Functions | STDDEV | Standard Deviation | ['timeperiod: 5', 'nbdev: 1'] | ['real'] |
| Statistic Functions | VAR | Variance | ['timeperiod: 5', 'nbdev: 1'] | ['real'] |

https://mrjbq7.github.io/ta-lib/funcs.html

Future Opportunities: More technical indicators with varying parameters.

# Exploratory Data Analysis

I calculated feature importance of the technical indicators with a variety of methods using Chi Squared Test, ANOVA F-value test, Mutual Information test, Logistic Regression with Regularization, and Tree Based Gini Information Gain feature importance.

### Chi Squared Calculation for Feature importances

| Indicator | Chi Importance |
|------------|----------------|
| STOCHRSI_1 | 8.14 |
| WILLR_5 | 7.56 |
| STOCHF_1 | 7.56 |
| WILLR_15 | 7.22 |
| WILLR_25 | 6.27 |
| WILLR_35 | 5.52 |
| SAREXT | 4.21 |
| BOP | 3.97 |
| WILLR_45 | 3.93 |
| CMO_5 | 3.59 |

### ANOVA F-value Calculation for Feature Importance

| Indicator | ANOVA Importance |
|-----------|------------------|
| RSI_5 | 46.6 |
| CMO_5 | 46.6 |
| WILLR_15 | 45.52 |
| STOCHF_1 | 44.85 |
| WILLR_5 | 44.85 |
| WILLR_25 | 38.64 |
| WILLR_35 | 33.69 |
| RSI_15 | 29.07 |
| CMO_15 | 29.07 |
| BOP | 27.02 |

### Mutual Information Calculation for Feature Importance

| Indicator | Mutual Info Importance |
|------------------------|------------------------|
| LINEARREG_ANGLE_15 | 0.024 |
| LINEARREG_SLOPE_15 | 0.024 |
| CCI_35 | 0.021 |
| CMO_45 | 0.019 |
| WILLR_25 | 0.019 |
| RSI_45 | 0.019 |
| TEMA_35 | 0.018 |
| WILLR_5 | 0.016 |
| LINEARREG_INTERCEPT_15 | 0.016 |
| ADOSC | 0.015 |


### Explained Variance of Features

![alttext](/imgs/pca_exp_var_m15.png "PCA Exp Variance")

### PCA the Features to 2 Dimensions to Observe Separability

![alttext](/imgs/pca_2_m15.png "PCA 2")

Future Opportunities: Try other dimensionality reduction algorithms (t-sne) and other feature selection methods.

# Modeling

I created data transformation and modeling pipelines that were used to gridsearch cross validated the models and preventing data leakage. The data transformation steps included scaling, selecting the best features, and dimensionality reduction. Logistic regression, neural networks, and boosted trees were used for the models. Each step in the pipeline has a variety of parameters that can be tuned. I used a powerful Amazon Web Service EC2 server to do the gridsearch parameter optimization in parallel. Both ROC_AUC and a custom % Return scoring functions were used for gridsearching.

Future Opportunities: Optimize the gridsearch scoring function to incorporate other financial metrics including alpha, beta, max drawdown, etc.

# Results

![alttext](/imgs/BADROC.png "Bad ROC")

![alttext](/imgs/calcreturns.png "Bad ROC")

### Logistic Regression

![alttext](/imgs/lr_gran_auc.png "Logistic Regression ROC")
![alttext](/imgs/lr_gran_returns.png "Logistic Regression Returns")

### Neural Network

![alttext](/imgs/nn_gran_auc.png "Neural Network ROC")
![alttext](/imgs/nn_gran_returns.png "Neural Network Returns")

### XGBoost Classifier

![alttext](/imgs/xg_gran_auc.png "XGBoost ROC")
![alttext](/imgs/xg_gran_returns.png "XGBoost Returns")

Future Opportunities: Stack classification and regression models. Tune a trading strategy based upon probabilities. Use a proper backtesting library incorporating bid / ask spreads, trading fees. Paper Trade.

# Web Application

Continuously update SQL database with live candles.
Continuously fit gridsearched models and predict the future direction for each candle granularity and save predictions.
Display results in table with fancy tradingview widgets below.

LINK TO WEBSITE HERE


# Resources

* get data
  * oanda restful api
    * eur_usd candles
      * bid / ask / mid
      * open / high / low / close
      * timestamp is at the open
    * https://www.udemy.com/python-algo-trading-fx-trading-with-oanda/
    * http://oanda-api-v20.readthedocs.io/en/latest/index.html
    * http://developer.oanda.com/rest-live-v20/introduction/
  * pickled dataframes
  * postgres
  * mongo
* target
  * sign of (next candle open price - current candle open price)
  * future open price
  * http://www.dcfnerds.com/94/arithmetic-vs-logarithmic-rates-of-return/
* features
  * exponential smoothing
  * log returns
  * technical indicators
    * TA-Lib https://github.com/mrjbq7/ta-lib
    * http://www.ta-lib.org/hdr_dw.html
    * https://mrjbq7.github.io/ta-lib/index.html
    * https://github.com/mobone/ta-machine/blob/master/indicators/sqz.py
    * wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    * tar xvzf ta-lib-0.4.0-src.tar.gz
    * ./configure --prefix=/usr
    * make
    * sudo make install
* models
  * classification vs. regression?
  * classic momentum and such
    * https://www.oreilly.com/learning/algorithmic-trading-in-less-than-100-lines-of-python-code
    * https://www.datacamp.com/community/tutorials/finance-python-trading#gs.DSdj=ds
  * machine learning
    * https://arxiv.org/pdf/1605.00003.pdf
    * http://wseas.us/e-library/conferences/2011/Penang/ACRE/ACRE-05.pdf
    * http://francescopochetti.com/stock-market-prediction-part-introduction/
  * neural nets (rnn / lstm)
    * https://github.com/GalvanizeOpenSource/Recurrent_Neural_Net_Meetup
    * http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
    * http://www.jakob-aungiers.com/articles/a/Multidimensional-LSTM-Networks-to-Predict-Bitcoin-Price
    * https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
    * https://www.otexts.org/fpp
  * probablistic programming
    * https://www.youtube.com/watch?v=coEVZNg_nlA
  * ARIMA
* predict and evaluate performance
  * backtest
    * https://github.com/quantopian/zipline
    * http://gbeced.github.io/pyalgotrade/
    * https://github.com/mementum/backtrader
  * returns, alpha, beta, sharpe, sortino, max drawdown, volatility
  * precision, recall, accuracy
  * mean squared error, root mean squared error
* other references
  * https://github.com/owocki/pytrader
