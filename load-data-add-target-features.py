
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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
import theano
import xgboost as xgb
from xgboost import XGBClassifier
import gc
import operator
plt.style.use('ggplot')

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
# =============================================================================
# def get_momentum():
#     mom_cols = []
#     for mom_time in [1, 15, 30, 60, 120]:
#         col = 'average_log_return_{}_sign'.format(mom_time)
#         df[col] = df['log_returns'].rolling(mom_time).mean().apply(up_down) #the sign of the average returns of the last x candles
#         mom_cols.append(col)
#     return mom_cols
# =============================================================================
       
def get_nn():
    model = Sequential()
    num_neurons_in_layer = 50
    num_inputs = 5 #x_train.shape[1]
    num_classes = 2 #y_train_binary.shape[1]
    model.add(Dense(input_dim=num_inputs, 
                     output_dim=num_neurons_in_layer, 
                     init='uniform', 
                     activation='tanh')) 
    model.add(Dense(input_dim=num_neurons_in_layer, 
                     output_dim=num_neurons_in_layer, 
                     init='uniform', 
                     activation='tanh'))
    model.add(Dense(input_dim=num_neurons_in_layer, 
                     output_dim=num_neurons_in_layer, 
                     init='uniform', 
                     activation='sigmoid'))
    model.add(Dense(input_dim=num_neurons_in_layer, 
                     output_dim=num_classes,
                     init='uniform', 
                     activation='softmax')) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"] ) # (keep)
    return model

def get_pipelines():
    lr = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', LogisticRegression())])
    dtc = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', DecisionTreeClassifier())])
    rfc = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', RandomForestClassifier(n_jobs=-1))])
    abc = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', AdaBoostClassifier())])
    gbc = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', GradientBoostingClassifier())])
    nnm = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', KerasClassifier(build_fn=get_nn, epochs=100, batch_size=500, verbose=1))])
    xgb = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', XGBClassifier())])
    mlp = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', MLPClassifier((100,3)))])
    svl = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', SVC(kernel="linear", C=0.025))])
    svc = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', SVC(gamma=2, C=1))])
    knc = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', KNeighborsClassifier(3))])
    #gpc = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True))])
    gnb = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', GaussianNB())])
    qda = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=5)), ('clf', QuadraticDiscriminantAnalysis())])
    pipes = {
            'lr': lr,
# =============================================================================
#             'dtc': dtc,
#             'rfc': rfc,
#             'abc': abc,
# =============================================================================
            'gbc': gbc,
# =============================================================================
            'nnm': nnm,
            'xgb': xgb,
# =============================================================================
            'mlp': mlp
# =============================================================================
#             'svl': svl,
#             'svc': svc,
#             'knc': knc,
#             #'gpc': gpc,
#             'gnb': gnb,
#             'qda': qda
# =============================================================================
            }
    return pipes

def pipe_cross_val():
    ts = TimeSeriesSplit(n_splits=3)
    prediction_df = pd.DataFrame([])
    for split_index, (train_index, test_index) in enumerate(ts.split(x)):
        print('split index: {}'.format(split_index))
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        prediction_df['y_test_{}'.format(str(split_index))] = y_test.values
        prediction_df['log_returns_{}'.format(str(split_index))] = df['log_returns'][test_index].values        
        for key, pipe in pipes.items():
            print('training {}'.format(key))
            pipe.fit(x_train, y_train)
            y_pred = pipe.predict(x_test)
            y_pred_proba = pipe.predict_proba(x_test)
            prediction_df['{}_{}_pred'.format(key, str(split_index))] = y_pred
            prediction_df['{}_{}_pred_proba'.format(key, str(split_index))] = y_pred_proba[:,1]
    return prediction_df

def calc_prediction_returns():
    prediction_cols = [col for col in prediction_df.columns if col[-4:] == 'pred']
    for pred_col in prediction_cols:
        sp_ind = re.search('_\d_', pred_col).group(0)[1]
        prediction_df[pred_col] = prediction_df[pred_col].map({1:1, 0:-1}).shift(1)
        prediction_df[pred_col+'_returns'] = prediction_df[pred_col] * prediction_df['log_returns_{}'.format(sp_ind)]
        print('{} {:.2f}%'.format(pred_col+'_returns', (np.exp(np.sum(prediction_df[pred_col+'_returns']))-1)*100))

def print_predictions_stats(mod_name, y_true, y_pred):
    print('\n', mod_name)
    up = sum(y_true==1) / len(y_true) *100
    print('up: {:.2f}%'.format(up))
    print('down: {:.2f}%'.format(1-up))
    print('accuracy: {:.2f}%'.format(accuracy_score(y_true, y_pred)*100))
    y_true = pd.Series(y_true, name='Actual')
    y_pred = pd.Series(y_pred, name='Predicted')
    print('classification report: ')
    print(classification_report(y_true, y_pred))
    print('confusion matrix: ')
    print(pd.crosstab(y_true, y_pred))

def compare_scalers_plot():
    close_prices = df['close'].values.reshape(-1,1)
    sc, mm, ma, rs = StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()
    scalers = [sc, mm, ma, rs]
    fig, axes = plt.subplots(len(scalers)+1, 1, figsize=(10,40))
    for i, ax in enumerate(axes.reshape(-1)):
        if i == 0:
            ax.hist(close_prices, bins=100)
            ax.set_title('No Scaling')
        else:
            scale = scalers[i-1]
            close_prices_scaled = scale.fit_transform(close_prices)
            ax.hist(close_prices_scaled, bins=100)
            ax.set_title(scale.__class__.__name__)
            print('{} min: {:.2f} max: {:.2f}'.format(scale.__class__.__name__, close_prices_scaled.min(), close_prices_scaled.max()))

def plot_prediction_roc(mod_name, y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr) * 100
    plt.plot(fpr, tpr, lw=1, label='{} auc: {:.2f}'.format(mod_name, roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.legend(loc='best')
    
def plot_prediction_returns(pred_returns):
    cum_returns = pred_returns.cumsum().apply(np.exp)-1 #you can add log returns and then transpose back with np.exp
    plt.plot(cum_returns)
    plt.legend(loc='best')
    
def plot_pca_2():
    pca = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=2))])
    x_new = pca.fit_transform(x)
    fig, ax = plt.subplots()
    x_new_one = x_new[y==1]
    x_new_zero = x_new[y==0]
    ax.scatter(x_new_one[:,0], x_new_one[:,1], c='green', label='up', alpha=.1)
    ax.scatter(x_new_zero[:,0], x_new_zero[:,1], c='orange', label='down', alpha=.1)
    plt.legend(loc='best')
            
def plot_line_data(price_series, figtitle):
    fig, ax = plt.subplots(figsize=(25,10))
    ax.plot(price_series)
    ax.set_title(figtitle)
    
def plot_a_feature(feature_names, date_start, date_end):
    df.loc[date_start:date_end].plot(y=feature_names, figsize=(25,10))
    
def plot_pca_elbow():
    ss = StandardScaler()
    x_ss = ss.fit_transform(x)
    pca = PCA()
    pca.fit(x_ss)
    plt.figure(1)
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_)
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    
def plot_pred_proba_hist(y_pred_proba):
    fig, ax = plt.subplots(figsize=(25,10))
    ax.hist(y_pred_proba, bins=100)
    
    

if __name__ == '__main__':
    df = get_data('EUR_USD_M1', datetime(2016,4,1), datetime(2016,6,1))
    print('got data')
    add_target()
    print('added targets')
    add_features()
    print('added features')
    x, y = split_data_x_y()
    pipes = get_pipelines()
    print('got pipes')
    prediction_df = pipe_cross_val()
    
    for mod in pipes.keys():
        plot_prediction_roc(mod, prediction_df['y_test_2'], prediction_df['{}_2_pred_proba'.format(mod)])
        print_predictions_stats(mod, prediction_df['y_test_2'], prediction_df['{}_2_pred'.format(mod)])
    plt.show()
    
    calc_prediction_returns()
    
    return_cols = [col for col in prediction_df.columns if col[-7:] == 'returns' and col[:2] == 'lr']
    for return_col in return_cols:
        plot_prediction_returns(prediction_df[return_col])
    plt.show()
    
    compare_scalers_plot()
    plt.show()
    
    plot_pca_2()
    plt.show()
    
    plot_pca_elbow()
    plt.show()
    
    
    
    
    pass