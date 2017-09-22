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
    conn = pg2.connect(dbname='applesauce')
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE DATABASE forex;')
    cur.close()
    conn.close()
    pass

def add_table(table_name):
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query1 = 'DROP TABLE IF EXISTS {}'.format(table_name)
    query2 = '''
    CREATE TABLE {} (
        time timestamptz, 
        volume int,
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
    pass

    
def data_to_table(table_name, data):
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = 'INSERT INTO {}(time, volume, close, high, low, open, complete) VALUES (%s, %s, %s, %s, %s, %s, %s)'.format(table_name) 
    cur.executemany(query, data)
    cur.close()
    conn.close()
    
    pass

def get_data(instru):
    accountID = os.environ['oanda_demo_id']
    access_token = os.environ['oanda_demo_api']
    client = oandapyV20.API(access_token=access_token)
    
    granularities = ['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3','H4', 'H6', 'H8', 'H12', 'D']
    granularities = granularities[::-1]
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    
    
    for gran in granularities:
    
        i=0
        hit_today = False
        last_timestamp = '2005-01-01T00:00:00.000000000Z'
        table_name = instru.lower()+'_'+gran.lower()
        add_table(table_name)
    
        while not hit_today:
    
            params = {'price': 'M', 'granularity': gran, 'count': 5000,
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
            last_month = data[-1][0][:7]
            print(last_timestamp)
            print(last_month)
            
            if last_month == '2017-09':
                hit_today = True
            
            pass


def return_data_db(table_name):
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = 'SELECT * FROM {};'.format(table_name)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def clean_data(file_path_name):
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    df = pd.DataFrame(data, columns=columns)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['volume'] = df.volume.astype(int)
    df['close'] = df.close.astype(float)
    df['high'] = df.high.astype(float)
    df['low'] = df.low.astype(float)
    df['open'] = df.open.astype(float)
    df['complete'] = df.complete.astype(bool)
    return df
    
        


if __name__ == '__main__':
    
    data = return_data_db('eur_usd_d')
    df = clean_data(df)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass
    