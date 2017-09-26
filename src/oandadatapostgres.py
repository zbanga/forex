# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import os
import sys
import glob
import psycopg2 as pg2
from sqlalchemy import create_engine
import time
from datetime import datetime, timezone

'''
psycopg2
1.In a new terminal, type conda install psycopg2
2.By default, psycopg2 looks for postgres in the wrong place, so we'll create a symbolic
link pointing it to the correct postgres server. Enter this command:
sudo ln -s /var/run/postgresql/.s.PGSQL.5432 /tmp/.s.PGSQL.5432
Now, when using the psycopg2.connect() function in python, you only need to
specify the database keyword, and not user or host
'''

def create_db():
    '''
    create forex database
    '''
    conn = pg2.connect(dbname='applesauce')
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE DATABASE forex;')
    cur.close()
    conn.close()

def add_table(table_name):
    '''
    drop table if exists and add table_name to database
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query1 = 'DROP TABLE IF EXISTS {}'.format(table_name)
    query2 = '''
    CREATE TABLE {} (
        time text,
        volume real,
        close real,
        high real,
        low real,
        open real,
        complete bool
    );
    '''.format(table_name)
    cur.execute(query1)
    cur.execute(query2)
    cur.close()
    conn.close()

def data_to_table(table_name, data):
    '''
    insert candles into table_name
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = 'INSERT INTO {}(time, volume, close, high, low, open, complete) VALUES (%s, %s, %s, %s, %s, %s, %s)'.format(table_name)
    cur.executemany(query, data)
    cur.close()
    conn.close()

def get_last_timestamp(table_name):
    '''
    return last timestamp from table_name
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = 'SELECT time FROM {} ORDER BY time DESC LIMIT 1;'.format(table_name)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data[0][0]

def time_in_table(table_name, time_stamp):
    '''
    check if time_stamp candle in table_name
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = "SELECT EXISTS(SELECT 1 FROM {} WHERE time='{}');".format(table_name, time_stamp)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data[0][0]

def get_data(instru, gran, last_timestamp = '2017-09-04T15:41:00.000000000Z'):
    '''
    get initial data to databse from 2005 to today
    '2000-01-01T00:00:00.000000000Z'
    '''
    accountID = os.environ['oanda_demo_id']
    access_token = os.environ['oanda_demo_api']
    client = oandapyV20.API(access_token=access_token)
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    i=0
    hit_today = False
    table_name = instru.lower()+'_'+gran.lower()
    while not hit_today:
        params = {'price': 'M', 'granularity': gran,
                  'count': 5000,
                  'from': last_timestamp,
                  'includeFirst': False,
                  'alignmentTimezone': 'America/New_York'}
        r = instruments.InstrumentsCandles(instrument=instru,params=params)
        client.request(r)
        resp = r.response
        i+=1
        print(r.status_code, i)
        data = []
        for can in resp['candles']:
            data.append((can['time'], can['volume'], can['mid']['c'], can['mid']['h'], can['mid']['l'], can['mid']['o'], can['complete']))
        data_to_table(table_name, data)
        last_timestamp = data[-1][0]
        last_month = data[-1][0][:10]
        print(last_timestamp)
        print(last_month)
        if last_month == '2017-09-26':
            hit_today = True

def get_data_continuous(instru, gran):
    '''
    continuously update table with new candles
    '''
    accountID = os.environ['oanda_demo_id']
    access_token = os.environ['oanda_demo_api']
    client = oandapyV20.API(access_token=access_token)
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    i=0
    hit_today = False
    table_name = instru.lower()+'_'+gran.lower()
    while True:
        last_timestamp = get_last_timestamp(table_name)
        print('last time stamp in table: {}'.format(last_timestamp))
        params = {'price': 'M', 'granularity': gran,
                  'count': 5000,
                  'from': last_timestamp,
                  'includeFirst': False,
                  'alignmentTimezone': 'America/New_York'}
        r = instruments.InstrumentsCandles(instrument=instru,params=params)
        client.request(r)
        resp = r.response
        i+=1
        print('oanda status: {} loop: {}'.format(r.status_code, i))
        data = []
        for can in resp['candles']:
            if can['complete'] == True and time_in_table(table_name, can['time']) == False:
                data.append((can['time'], can['volume'], can['mid']['c'], can['mid']['h'], can['mid']['l'], can['mid']['o'], can['complete']))
                data_to_table(table_name, data)
                print('added to table: {}'.format(data))
                data = []
        time.sleep(10)

def return_data_table(table_name):
    '''
    get all data from table
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = 'SELECT * FROM {};'.format(table_name)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def return_data_table_gt_time(table_name, time_stamp):
    '''
    get all data from table
    ex eur_usd_m1, '2017-09-26T15:41:00.000000000Z'
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = "SELECT * FROM {} WHERE time > '{}';".format(table_name, time_stamp)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data


def clean_data(data):
    '''
    take data dump and convert to df
    '''
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    df = pd.DataFrame(data, columns=columns)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['volume'] = df.volume.astype(float)
    df['close'] = df.close.astype(float)
    df['high'] = df.high.astype(float)
    df['low'] = df.low.astype(float)
    df['open'] = df.open.astype(float)
    df.set_index('time', inplace=True)
    df.drop('complete', axis=1, inplace=True)
    return df




if __name__ == '__main__':

    get_data_continuous('EUR_USD', 'M1')

    #
    # granularities = ['S5', 'S10', 'S15', 'S30', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3','H4', 'H6', 'H8', 'H12', 'D']
    # granularities = granularities[::-1]
    # for gran in granularities:
    # gran = 'M1'
    # print(gran)
    # # add_table('EUR_USD_'+gran)
    # get_data(instru='EUR_USD', gran=gran)












    pass
