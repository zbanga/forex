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
from os import listdir
from os.path import isfile, join


def mongo_to_pandas_pickle():
    db_cilent = MongoClient()
    db = db_cilent['forex']
    cols = db.collection_names()
    print(cols)

    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    cols = ['EUR_USD_S5']

    for col in cols:
        i=0
        df = pd.DataFrame(columns=columns)
        for item in db[col].find():
            key = list(item.keys())[1]
            data = list(item[key])
            df = pd.DataFrame(data, columns=columns)
            i+=1
            print(i, df.shape)
            df.to_pickle('data/data-s5/'+col+'_'+str(i))


def pickle_to_one():
    mypath = '/home/applesauce/galvanize/immersive/forex/forex/data/data-s5'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    i=0
    dfs = []
    stop = int(len(onlyfiles)/2) #3371
    print(stop)
    for fil in onlyfiles[3371:]:
        df_next = pd.read_pickle('data/data-s5/'+fil)
        dfs.append(df_next)
        i+=1
        print(i)
    df = pd.concat(dfs)
    print('cat')
    df.to_pickle('data/EUR_USD_S5_2')
    print('done')










if __name__ == '__main__':
    pickle_to_one()
