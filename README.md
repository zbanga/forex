
# EUR/USD Foreign Exchange Rate Prediction

Using neural networks and boosted trees to predict the direction of of the EUR/USD foreign exchange rates.

Web App: link

## Table of Contents
1. [Data](#data)
2. [Target](#target)
3. [Features](#features)
4. [Modeling](#modeling)
5. [Results](#results)
6. [Web Application](#web application)
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

#### Minute Candles Example
time | volume | open | high | low | close | complete
--- | --- | --- | --- | --- | --- | ---
2005-01-02 20:46:00 | 1 | 1.356100 | 1.356100 | 1.356100 | 1.356100 | True
2005-01-02 20:47:00 | 3 | 1.356000 | 1.356800 | 1.356000 | 1.356100 | True
2005-01-02 20:48:00 | 1 | 1.356000 | 1.356000 | 1.356000 | 1.356000 | True
2005-01-02 20:49:00 | 5 | 1.356100 | 1.356600 | 1.355800 | 1.355800 | True
2005-01-02 20:50:00 | 2 | 1.355700 | 1.355700 | 1.355600 | 1.355600 | True
2005-01-02 20:51:00 | 3 | 1.355600 | 1.355600 | 1.355200 | 1.355200 | True
2005-01-02 20:52:00 | 4 | 1.355500 | 1.356100 | 1.355100 | 1.355100 | True
2005-01-02 20:53:00 | 4 | 1.355200 | 1.355200 | 1.354600 | 1.354900 | True
2005-01-02 20:54:00 | 8 | 1.355500 | 1.355500 | 1.354100 | 1.354100 | True
2005-01-02 20:55:00 | 2 | 1.354100 | 1.355000 | 1.354100 | 1.355000 | True

### 1 Month Candles with Volume
![alttext](/imgs/1monthcandles.png "Forex Candles")

http://developer.oanda.com/rest-live-v20/introduction/

Future Opportunities: Incorporate more data, economic calendar, historical position ratios, bid ask spreads, commitments of traders, orderbook...

# Target

I am using a classification target (1 or 0) of whether or not the close price of the next candle is higher or lower than the close price of the current candle. I also calculated the log returns log(close n+1 / close n) to be used for calculating returns.

| time | volume | open | high | low | close | log_returns | log_returns_shifted | target |
|---------------------|--------|----------|----------|----------|----------|-------------|---------------------|--------|
| 0:00 | 29 | 1.137860 | 1.137960 | 1.137680 | 1.137920 | nan | 0.000097 | 1 |
| 1:00 | 16 | 1.137880 | 1.138030 | 1.137800 | 1.138030 | 0.000097 | 0.000132 | 1 |
| 2:00 | 11 | 1.138060 | 1.138210 | 1.138040 | 1.138180 | 0.000132 | 0.000000 | 1 |
| 3:00 | 15 | 1.138160 | 1.138230 | 1.138130 | 1.138180 | 0.000000 | 0.000000 | 1 |
| 4:00 | 8 | 1.138180 | 1.138200 | 1.138140 | 1.138180 | 0.000000 | 0.000123 | 1 |
| 5:00 | 4 | 1.138220 | 1.138320 | 1.138220 | 1.138320 | 0.000123 | -0.000018 | 0 |
| 6:00 | 14 | 1.138280 | 1.138300 | 1.138220 | 1.138300 | -0.000018 | -0.000018 | 0 |
| 7:00 | 10 | 1.138340 | 1.138340 | 1.138270 | 1.138280 | -0.000018 | 0.000325 | 1 |
| 8:00 | 42 | 1.138320 | 1.138720 | 1.138320 | 1.138650 | 0.000325 | 0.000026 | 1 |
| 9:00 | 12 | 1.138620 | 1.138720 | 1.138620 | 1.138680 | 0.000026 | -0.000123 | 0 |

### Log Returns Distribution vs. Arithmetic Returns Distribution
![alttext](/imgs/returns.png "Forex Returns")

Future Opportunities: Incorporate regression not just classification.

# Features

Below are the technical analysis features that were added to the data.

| Group | Short Name | Name |
|-----------------------|---------------------|---------------------------------------------------|
| Momentum Indicators | ADX | Average Directional Movement Index |
| Momentum Indicators | ADXR | Average Directional Movement Index Rating |
| Momentum Indicators | APO | Absolute Price Oscillator |
| Momentum Indicators | AROON | Aroon |
| Momentum Indicators | AROONOSC | Aroon Oscillator |
| Momentum Indicators | BOP | Balance Of Power |
| Momentum Indicators | CCI | Commodity Channel Index |
| Momentum Indicators | CMO | Chande Momentum Oscillator |
| Momentum Indicators | DX | Directional Movement Index |
| Momentum Indicators | MACD | Moving Average Convergence/Divergence |
| Momentum Indicators | MACDEXT | MACD with controllable MA type |
| Momentum Indicators | MACDFIX | Moving Average Convergence/Divergence Fix 12/26 |
| Momentum Indicators | MFI | Money Flow Index |
| Momentum Indicators | MINUS_DI | Minus Directional Indicator |
| Momentum Indicators | MINUS_DM | Minus Directional Movement |
| Momentum Indicators | MOM | Momentum |
| Momentum Indicators | PLUS_DI | Plus Directional Indicator |
| Momentum Indicators | PLUS_DM | Plus Directional Movement |
| Momentum Indicators | PPO | Percentage Price Oscillator |
| Momentum Indicators | ROC | Rate of change : ((price/prevPrice)-1)*100 |
| Momentum Indicators | ROCP | Rate of change Percentage: (price-prevPrice)/p... |
| Momentum Indicators | ROCR | Rate of change ratio: (price/prevPrice) |
| Momentum Indicators | ROCR100 | Rate of change ratio 100 scale: (price/prevPri... |
| Momentum Indicators | RSI | Relative Strength Index |
| Momentum Indicators | STOCH | Stochastic |
| Momentum Indicators | STOCHF | Stochastic Fast |
| Momentum Indicators | STOCHRSI | Stochastic Relative Strength Index |
| Momentum Indicators | TRIX | 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA |
| Momentum Indicators | ULTOSC | Ultimate Oscillator |
| Momentum Indicators | WILLR | Williams' %R |
| Overlap Studies | BBANDS | Bollinger Bands |
| Overlap Studies | DEMA | Double Exponential Moving Average |
| Overlap Studies | EMA | Exponential Moving Average |
| Overlap Studies | HT_TRENDLINE | Hilbert Transform - Instantaneous Trendline |
| Overlap Studies | KAMA | Kaufman Adaptive Moving Average |
| Overlap Studies | MA | Moving average |
| Overlap Studies | MAMA | MESA Adaptive Moving Average |
| Overlap Studies | MIDPOINT | MidPoint over period |
| Overlap Studies | MIDPRICE | Midpoint Price over period |
| Overlap Studies | SAR | Parabolic SAR |
| Overlap Studies | SAREXT | Parabolic SAR - Extended |
| Overlap Studies | SMA | Simple Moving Average |
| Overlap Studies | T3 | Triple Exponential Moving Average (T3) |
| Overlap Studies | TEMA | Triple Exponential Moving Average |
| Overlap Studies | TRIMA | Triangular Moving Average |
| Overlap Studies | WMA | Weighted Moving Average |
| Volume Indicators | AD | Chaikin A/D Line |
| Volume Indicators | ADOSC | Chaikin A/D Oscillator |
| Volume Indicators | OBV | On Balance Volume |
| Cycle Indicators | HT_DCPERIOD | Hilbert Transform - Dominant Cycle Period |
| Cycle Indicators | HT_DCPHASE | Hilbert Transform - Dominant Cycle Phase |
| Cycle Indicators | HT_PHASOR | Hilbert Transform - Phasor Components |
| Cycle Indicators | HT_SINE | Hilbert Transform - SineWave |
| Cycle Indicators | HT_TRENDMODE | Hilbert Transform - Trend vs Cycle Mode |
| Volatility Indicators | ATR | Average True Range |
| Volatility Indicators | NATR | Normalized Average True Range |
| Volatility Indicators | TRANGE | True Range |
| Statistic Functions | BETA | Beta |
| Statistic Functions | CORREL | Pearson's Correlation Coefficient (r) |
| Statistic Functions | LINEARREG | Linear Regression |
| Statistic Functions | LINEARREG_ANGLE | Linear Regression Angle |
| Statistic Functions | LINEARREG_INTERCEPT | Linear Regression Intercept |
| Statistic Functions | LINEARREG_SLOPE | Linear Regression Slope |
| Statistic Functions | STDDEV | Standard Deviation |
| Statistic Functions | TSF | Time Series Forecast |
| Statistic Functions | VAR | Variance |

https://mrjbq7.github.io/ta-lib/funcs.html

Future Opportunities: More technical indicators with varying parameters.

# Exploratory Analysis

Feature importances with SelectKBest
LogisticRegresssion and Regularization with the L1 Lasso at different rates to see which features survive!
Tree Based selection can be used to see which features give the model the highest information gain.
Use PCA to reduce the dimensions and solve for the curse of dimensionality and the colinearity between the features.

Use these tools in a pipeline to prevent data leakage and allow for gridsearching.

Future Opportunities: Try other dimensionality reduction algorithms (t-sne) and other feature selection methods.

# Modeling

Spin up the most expensive AWS EC2 instances and gridsearch cross validate (train / test split) each pipeline with a variety of parameters for each for each candle granularity and take the model with the highest score.
Customized the gridsearch scoring function to maximize returns.

Neural Network and XGBoost Classifier

Future Opportunities: Optomize the gridsearch scoring function to incorporate other financial metrics including alpha, beta, max drawdown, etc.

# Results

Look are predicted probability by models
Calculate the returns and the classification metrics including a confusion matrix, accuracy, precision, recall, roc curve.

Future Opportunities: Tune a trading strategy based upon probabilities. Use a proper backtesting library incorporating bid / ask spreads, trading fees.

# Web Application

Continuously update database with live candles.
Continuously fit gridsearched models and predict the future direction for each candle granularity.
Display results in table with tradingview widgets below.

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
