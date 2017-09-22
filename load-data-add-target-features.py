# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_data(file_name, date_start, date_end):
    df = pd.read_pickle('data/'+file_name)
    df.set_index('time', inplace=True)
    df.drop('complete', axis=1, inplace=True)
    #df_columns = list(df.columns)
    df = df.loc[date_start:date_end]
    return df

def up_down(row):
    if row >= 0:
        return 1
    elif row < 0:
        return 0
    else:
        None

def add_target():
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['ari_returns'] = (df['close'] / df['close'].shift(1)) - 1
    df['log_returns_shifted'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_label_direction'] = df['log_returns'].apply(up_down)
    df['target_label_direction_shifted'] = df['log_returns_shifted'].apply(up_down)

def add_features():
    mom_ind = talib.get_function_groups()['Momentum Indicators']
    over_stud = talib.get_function_groups()['Overlap Studies']
    volu_ind = talib.get_function_groups()['Volume Indicators']
    cyc_ind = talib.get_function_groups()['Cycle Indicators']
    vola_ind = talib.get_function_groups()['Volatility Indicators']
    talib_abstract_fun_list = mom_ind + over_stud + volu_ind + cyc_ind + vola_ind
    talib_abstract_fun_list.remove('MAVP')
    ohlcv = {
    'open': df['open'],
    'high': df['high'],
    'low': df['low'],
    'close': df['close'],
    'volume': df['volume'].astype(float)
    }
    for fun in talib_abstract_fun_list:
    res = getattr(talib.abstract, fun)(ohlcv)
    if len(res) > 10:
        df[fun] = res
    else:
        for i, val in enumerate(res):
            df[fun+'_'+str(i+1)] = val
    for per in [3,6,12,18,24,30]:
    col_name = 'MAVP_'+str(per)
    df[col_name] = talib.MAVP(df['close'].values, periods=np.array([float(per)]*df.shape[0]))
    
def split_data_x_y():
    drop_columns = ['volume', 'close', 'high', 'low', 'open', 'complete', 'log_returns', 'ari_returns', 'log_returns_shifted', 'target_label_direction', 'target_label_direction_shifted']
    predict_columns = [i for i in df.columns if i not in drop_columns]
    y = df['target_label_direction_shifted']
    x = df[predict_columns]
    return x, y

def window_cross_val():
    x.shape[0]
    
    
    pass


def scale_train_test(scaler):
    scaler = StandardScaler()
    x_ss = ss.fit_transform(x)
    x_ss = pd.DataFrame(x, columns=predict_columns)
    
    
    
def quick_line_plot(price_series, figtitle):
    fig, ax = plt.subplots(figsize=(25,10))
    ax.plot(price_series)
    ax.set_title(figtitle)
    
def quick_plot_feature(feature_names, date_start, date_end):
    df.loc[date_start:date_end].plot(y=feature_names, figsize=(25,10))
    
    

if __name__ == '__main__':
    df = get_data('EUR_USD_M1', datetime(2016,4,1), datetime(2016,6,1))
    add_target()
    add_features()
    x, y = split_data_x_y()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass