# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import re
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano
import xgboost as xgb
import gc


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
    df.dropna(inplace=True)
    y = df['target_label_direction_shifted']
    x = df[predict_columns]
    return x, y


def get_pipelines():
    pipe1 = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', LogisticRegression())])
    pipe2 = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', SVC())])
    pipe3 = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', DecisionTreeClassifier())])
    pipe4 = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', RandomForestClassifier())])
    pipe5 = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', AdaBoostClassifier())])
    pipe6 = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', GradientBoostingClassifier())])
    pipes = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6]
    return pipes

def cross_val():
    ts = TimeSeriesSplit(n_splits=3)
    scores = {}
    for pipe in pipes:
        pipe_name = pipe.steps[2][1].__class__.__name__
        print('training '.format(pipe_name))
        scores[pipe_name] = []
        for train_index, test_index in ts.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            pipe.fit(x_train, y_train)
            scores[pipe_name].append(pipe.score(x_test, y_test))
    return scores
        
            
def quick_line_plot(price_series, figtitle):
    fig, ax = plt.subplots(figsize=(25,10))
    ax.plot(price_series)
    ax.set_title(figtitle)
    
def quick_plot_feature(feature_names, date_start, date_end):
    df.loc[date_start:date_end].plot(y=feature_names, figsize=(25,10))
    
    

if __name__ == '__main__':
    df = get_data('EUR_USD_M1', datetime(2016,4,1), datetime(2016,6,1))
    print('got data')
    add_target()
    print('added targets')
    add_features()
    print('added features')
    x, y = split_data_x_y()
    print('split data')
    pipes = get_pipelines()
    print('got pipes')
    scores = cross_val()
    print('completed cross val')
    print(scores)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass