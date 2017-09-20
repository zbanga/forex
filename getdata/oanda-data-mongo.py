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


accountID = os.environ['oanda_demo_id']
access_token = os.environ['oanda_demo_api']

client = oandapyV20.API(access_token=access_token)

granularities = ['H6']

 #['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3','H4', 'H6', 'H8', 'H12', 'D']
granularities = granularities[::-1]

instru = sys.argv[1]

db_cilent = MongoClient()
db = db_cilent['forex']

for gran in granularities:

    i=0
    hit_today = False
    last_timestamp = '2005-01-01T00:00:00.000000000Z'
    table = db[instru+'_'+gran]

    while not hit_today:

        params = {'price': 'M', 'granularity': gran, 'count': 5000,
                  'from': last_timestamp,
                  'alignmentTimezone': 'America/New_York'}
        r = instruments.InstrumentsCandles(instrument=instru,params=params)
        client.request(r)
        resp = r.response
        i+=1
        data = []
        for can in resp['candles']:
            data.append([can['time'], can['volume'], can['mid']['c'], can['mid']['h'], can['mid']['l'], can['mid']['o'], can['complete']])
        table.insert({str(i): data})
        last_timestamp = data[-1][0]
        last_month = last_timestamp[:7]
        print(last_month)
        if last_month == '2017-09':
            hit_today = True
