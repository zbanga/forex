'''
from
https://www.oreilly.com/learning/algorithmic-trading-in-less-than-100-lines-of-python-code

    example
                          returns  position_15  strategy_15
    time                                                   
    2016-12-08 00:00:00       NaN          NaN          NaN
    2016-12-08 00:01:00  0.000065          NaN          NaN
    2016-12-08 00:02:00  0.000093          NaN          NaN
    2016-12-08 00:03:00  0.000242          NaN          NaN
    2016-12-08 00:04:00 -0.000019          NaN          NaN
    2016-12-08 00:05:00  0.000214          NaN          NaN
    2016-12-08 00:06:00 -0.000037          NaN          NaN
    2016-12-08 00:07:00  0.000028          NaN          NaN
    2016-12-08 00:08:00  0.000019          NaN          NaN
    2016-12-08 00:09:00 -0.000009          NaN          NaN
    2016-12-08 00:10:00  0.000214          NaN          NaN
    2016-12-08 00:11:00  0.000084          NaN          NaN
    2016-12-08 00:12:00 -0.000418          NaN          NaN
    2016-12-08 00:13:00 -0.000056          NaN          NaN
    2016-12-08 00:14:00 -0.000149          NaN          NaN
    2016-12-08 00:17:00  0.000223          1.0          NaN
    2016-12-08 00:18:00 -0.000037          1.0    -0.000037 pos: long
    2016-12-08 00:19:00  0.000037          1.0     0.000037 pos: long
    2016-12-08 00:20:00 -0.000028          1.0    -0.000028 pos: long
    2016-12-08 00:21:00 -0.000009          1.0    -0.000009 pos: long
    2016-12-08 00:23:00  0.000019         -1.0     0.000019 pos: long
    2016-12-08 00:24:00  0.000037         -1.0    -0.000037 pos: short
    2016-12-08 00:25:00 -0.000028         -1.0     0.000028
    2016-12-08 00:26:00 -0.000139         -1.0     0.000139

Timestamp for candle is at the open price

'''

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def get_data_build_model(file):
    
    df = pd.read_pickle(file)
    df.set_index('time', inplace=True)
    df = df[['open']]
    df = df.loc[datetime(2016, 12, 8):datetime(2016, 12, 10), :]
    
    df['actual_up_down'] = np.sign(df['open']-df['open'].shift(1))
    
    df['log_returns'] = np.log(df['open'] / df['open'].shift(1))
    df['ari_returns'] = (df['open'] / df['open'].shift(1)) - 1 # or df['open'].pct_change()
    
    cols = []
    for momentum in [15, 30, 60, 120]:
        col = 'position_{}'.format(momentum)
        df[col] = np.sign(df['log_returns'].rolling(momentum).mean()).shift(1) #the sign of the average returns of the last x candles
        cols.append(col)
    
    strats = ['log_returns']
    
    for col in cols:
        strat = 'strategy_{}'.format(col.split('_')[1])
        df[strat] = df[col] * df['log_returns'] #shift last sign one and multiply by return
        strats.append(strat)
            
    return df, strats


def plot_stuff(df, strats):
    
    return_types = ['log_returns', 'ari_returns']
    fig, axes = plt.subplots(len(return_types), 1)
    for i, ax in enumerate(axes.reshape(-1)):
        df[return_types[i]].hist(bins=50, ax=ax)
        ax.set_title(return_types[i])
    plt.show()
    
    fig, axes = plt.subplots(len(strats), 1)
    for i, ax in enumerate(axes.reshape(-1)):
        df[strats[i]].hist(bins=50, ax=ax)
        ax.set_title(strats[i])
    plt.show()
    
    fig, axes = plt.subplots()
    cum_returns = df[strats].dropna().cumsum().apply(np.exp)-1 #you can add log returns and then transpose back with np.exp
    cum_returns.plot(ax=axes)
    plt.legend(loc='best')
    plt.show()
    
    fig, axes = plt.subplots()
    for strat in strats:
        df[strat].hist(alpha = 0.2, bins = 30, histtype="stepfilled", normed=True, label = strat)
    plt.title('Returns')
    plt.legend()
    
    pass


if __name__ == '__main__':
    #df, strats = get_data_build_model('../data/EUR_USD_D')
    df, strats = get_data_build_model('../data/EUR_USD_M1')
    plot_stuff(df, strats)
    
    
    pass