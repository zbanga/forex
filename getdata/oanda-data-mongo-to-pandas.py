#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:27:37 2017

@author: applesauce

params
http://developer.oanda.com/rest-live-v20/instrument-ep/

EUR_USD - Euro
GBP_USD - Cable
USD_JPY - Gopher
USD_CHF - Swissie

AUD_USD - Aussie
USD_CAD - Loonie
NZD_USD - Kiwi

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import os
import sys
from pymongo import MongoClient

db_cilent = MongoClient()
db = db_cilent['forex']
cols = db.collection_names()
print(cols)

columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']

cols = ['USD_CHF_S30', 'USD_CAD_S15','USD_CAD_S5', 'USD_JPY_S30', 'USD_CAD_S10', 'USD_CHF_S10', 'USD_CHF_S5', 'USD_JPY_S5', 'USD_JPY_S10', 'USD_CAD_S30', 'USD_CHF_S15', 'GBP_USD_S30', 'GBP_USD_S5', 'USD_JPY_S15', 'GBP_USD_S15', 'GBP_USD_S10']
cols = ['EUR_USD_S5']

for col in cols:
    i=0
    df = pd.DataFrame(columns=columns)
    for item in db[col].find():
        key = list(item.keys())[1]
        data = list(item[key])
        df_next = pd.DataFrame(data, columns=columns)
        df = df.append(df_next, ignore_index=True)
        i+=1
        print(i, df.shape)
    df.drop_duplicates('time', keep='first', inplace=True)
    print(col, df.shape)
    df.to_pickle('data/'+col)

#
# df = pd.DataFrame(columns=columns)
# for item in db[col].find():
#     key = list(item.keys())[1]
#     data = list(item[key])
#     df_next = pd.DataFrame(data, columns=columns)
#     df = df.append(df_next, ignore_index=True)
# df = pd.DataFrame(list(list(db[col].find())[0].values())[1], columns=columns)
    #
    # i=0
    # df = pd.DataFrame(columns=columns)
    # for item in db[col].find():
    #     key = list(item.keys())[1]
    #     data = list(item[key])
    #     df_next = pd.DataFrame(data, columns=columns)
    #     df = df.append(df_next, ignore_index=True)
    #     i+=1
    #     print(i, df.shape)
