
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
6. [Paper Trading](#paper-trading)
7. [Web Application](#web-application)
8. [Resources](#resources)

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

### Technical Indicators with More Than 1 Parameter

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

# Feature Analysis

I calculated feature importance of the technical indicators with a variety of methods using Chi Squared Test, ANOVA F-value test, Mutual Information test, Logistic Regression with Regularization, and Tree Based Gini Information Gain feature importance.

### Chi Squared Calculation for Feature Importances

| Indicator | Chi 2 Feature Importance |
|--------------------|--------------------------|
| STOCHRSI_FASTK | 187.418567 |
| WILLR_5_REAL | 149.494590 |
| STOCHF_FASTK | 149.494590 |
| WILLR_15_REAL | 118.146973 |
| BOP_REAL | 97.073508 |
| WILLR_25_REAL | 94.308444 |
| CCI_5_REAL | 87.559761 |
| SAREXT_REAL | 80.269241 |
| WILLR_35_REAL | 78.232391 |
| AROON_5_AROONUP | 75.784805 |
| WILLR_45_REAL | 68.047940 |
| STOCHRSI_FASTD | 65.356457 |
| AROONOSC_5_REAL | 64.849298 |
| RSI_5_REAL | 63.942682 |
| CMO_5_REAL | 63.942682 |
| AROON_5_AROONDOWN | 60.208233 |
| STOCHF_FASTD | 52.228574 |
| STOCH_SLOWK | 52.228574 |
| MFI_5_REAL | 43.643316 |
| AROON_15_AROONUP | 36.278569 |
| PLUS_DI_5_REAL | 35.991333 |
| AROONOSC_15_REAL | 30.384222 |
| MINUS_DI_5_REAL | 28.155336 |
| AROON_15_AROONDOWN | 27.817957 |
| AROON_25_AROONUP | 27.463824 |
| STOCH_SLOWD | 25.369297 |
| AROONOSC_25_REAL | 21.301853 |
| RSI_15_REAL | 18.953485 |
| CMO_15_REAL | 18.953485 |
| AROON_35_AROONUP | 18.861512 |

### ANOVA F-value Calculation for Feature Importance

| Indicator | ANOVA Feature Importance |
|-----------------|--------------------------|
| STOCHF_FASTK | 907.387974 |
| WILLR_5_REAL | 907.387974 |
| RSI_5_REAL | 892.273184 |
| CMO_5_REAL | 892.273184 |
| WILLR_15_REAL | 742.706199 |
| BOP_REAL | 684.049823 |
| WILLR_25_REAL | 587.246205 |
| ULTOSC_REAL | 560.140155 |
| CCI_5_REAL | 556.521236 |
| STOCHRSI_FASTK | 553.365985 |
| CMO_15_REAL | 531.577331 |
| RSI_15_REAL | 531.577331 |
| CCI_15_REAL | 521.843503 |
| PLUS_DI_5_REAL | 494.773138 |
| WILLR_35_REAL | 481.797253 |
| STOCH_SLOWK | 472.382646 |
| STOCHF_FASTD | 472.382646 |
| WILLR_45_REAL | 414.600246 |
| MINUS_DI_5_REAL | 402.280988 |
| CCI_25_REAL | 401.438393 |
| CMO_25_REAL | 380.701244 |
| RSI_25_REAL | 380.701244 |
| MOM_5_REAL | 340.955823 |
| CCI_35_REAL | 328.555654 |
| ROC_5_REAL | 326.194949 |
| ROCP_5_REAL | 326.194949 |
| ROCR_5_REAL | 326.194949 |
| ROCR100_5_REAL | 326.194949 |
| MFI_5_REAL | 323.982085 |
| STOCHRSI_FASTD | 319.723070 |

### Mutual Information Calculation for Feature Importance

| Indicator | Mutual Info Feature Importance |
|------------------------|--------------------------------|
| STOCHF_FASTK | 0.006356 |
| STOCHRSI_FASTK | 0.005146 |
| RSI_5_REAL | 0.005116 |
| CMO_5_REAL | 0.005096 |
| AROON_15_AROONUP | 0.004554 |
| HT_TRENDMODE_INTEGER | 0.004179 |
| CORREL_5_REAL | 0.004031 |
| ROCR100_15_REAL | 0.004027 |
| ROC_15_REAL | 0.003884 |
| PLUS_DI_35_REAL | 0.003870 |
| ROCR_15_REAL | 0.003867 |
| MINUS_DI_15_REAL | 0.003775 |
| CCI_35_REAL | 0.003768 |
| WILLR_15_REAL | 0.003745 |
| ROCP_15_REAL | 0.003722 |
| LINEARREG_SLOPE_5_REAL | 0.003672 |
| RSI_15_REAL | 0.003459 |
| CMO_15_REAL | 0.003421 |
| BOP_REAL | 0.003329 |
| WILLR_5_REAL | 0.003328 |
| MINUS_DM_15_REAL | 0.003286 |
| ULTOSC_REAL | 0.003267 |
| WILLR_25_REAL | 0.003259 |
| PLUS_DI_5_REAL | 0.003096 |
| MACD_MACD | 0.003094 |
| AROON_45_AROONUP | 0.002972 |
| CORREL_25_REAL | 0.002475 |
| MINUS_DI_5_REAL | 0.002436 |
| ADX_15_REAL | 0.002397 |
| TRIX_5_REAL | 0.002391 |


### Explained Variance of Features

![alttext](/imgs/pca_exp_var_m15.png "PCA Exp Variance")

### PCA the Features to 2 Dimensions to Observe Separability

![alttext](/imgs/pca_2_m15.png "PCA 2")

Future Opportunities: Try other dimensionality reduction algorithms (t-sne) and other feature selection methods.

# Modeling

I created data transformation and modeling pipelines that were used to gridsearch cross validated the models and preventing data leakage. The data transformation steps included scaling, selecting the best features, and dimensionality reduction. Logistic regression, neural networks, and boosted trees were used for the models. Each step in the pipeline has a variety of parameters that can be tuned. I used a powerful Amazon Web Service EC2 server to do the gridsearch parameter optimization in parallel. Both ROC_AUC and a Custom % Return scoring function was used for gridsearching.

Future Opportunities: Optimize the gridsearch scoring function to incorporate other financial metrics including alpha, beta, max drawdown, etc.

# Results

Scaling the Features, No PCA, No Feature Selection, and Logistic Regression with Regularization for Feature Selection (L1 and L2 were very similar) on the 15 minute candles produced the best results. The MLP network that provided very similar results had 1 layer and a Logistic activation function. It was pretty much just a Logistic Regression algorithm with a fancier name.

### 15 Minute Candles

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

Future Opportunities: Stack classification and regression models. Tune a trading strategy based upon probabilities. Use a proper backtesting library incorporating bid / ask spreads, trading fees.

# Paper Trading

Currently the Logistic Regression model is being Paper Traded with the 15 Minute bars. The script is running on a free AWS EC2 instance with a PostgreSQL database to store the historical candles.

# Web Application

The web app has a script that continuously updates the SQL database with new candles for each granularity. A model and predict the future direction for each candle granularity and save predictions. Display the best features for predicting each candle. Display from papertrading.

![alttext](/imgs/webapp.png "Web App")


![alttext](/imgs/tradinglog.png "Trading Log")

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
