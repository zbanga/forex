

# Acquire Data

*Still* | `renders` | **nicely**

I used the Oanda API to download historical EUR_USD exchange prices to a PostgreSQL database. The API only allows you to receive 5,000 records per request so I setup a script to download this information overnight. The database contains tables with the exchange price every 5 seconds, 10 seconds, 15 seconds, etc. as shown by the table below.

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

![alttext](/imgs/forexline.png "Forex Line")


<!-- TradingView Widget BEGIN -->
<div id="tv-medium-widget-64ec8"></div>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script type="text/javascript">
new TradingView.MediumWidget({
  "container_id": "tv-medium-widget-64ec8",
  "symbols": [
    [
      "EUR_USD",
      "OANDA:EURUSD|1d"
    ]
  ],
  "gridLineColor": "#e9e9ea",
  "fontColor": "#83888D",
  "underLineColor": "#dbeffb",
  "trendLineColor": "#4bafe9",
  "width": "1000px",
  "height": "400px",
  "locale": "en"
});
</script>
<!-- TradingView Widget END -->

http://developer.oanda.com/rest-live-v20/introduction/

# Create Target




# Add Features





# Exploratory Analysis




# Model and Gridsearch




# Analyze Results






# Share Results

Live data pipeline





forex

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
