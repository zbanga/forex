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

'''
psycopg2
1.In a new terminal, type conda install psycopg2
2.By default, psycopg2 looks for postgres in the wrong place, so we'll create a symbolic
link pointing it to the correct postgres server. Enter this command:
sudo ln -s /var/run/postgresql/.s.PGSQL.5432 /tmp/.s.PGSQL.5432
Now, when using the psycopg2.connect() function in python, you only need to
specify the database keyword, and not user or host

setup db
$
sudo adduser evdbuser
sudo -u postgres psql template1

psql>
CREATE DATABASE eventsdb;
CREATE USER evdbuser WITH PASSWORD 'p2FX68';
GRANT ALL PRIVILEGES ON DATABASE eventsdb TO evdbuser;

$
psql -d eventsdb -U evdbuser -h localhost

drop table events;
CREATE TABLE events(
    eventid             SERIAL PRIMARY KEY NOT NULL,
    prediction          VARCHAR(20),
    pred_prob           INT,
    disposition         VARCHAR(20),
    approx_payout_date  TIMESTAMP    NOT NULL,
    body_length         INT          NOT NULL,
    channels            INT          NOT NULL,
    country             VARCHAR(2)   NOT NULL,
    currency            VARCHAR(3)   NOT NULL,
    delivery_method     REAL         NOT NULL,
    description         VARCHAR(500) NOT NULL,
    email_domain        VARCHAR(50)  NOT NULL,
    event_created       TIMESTAMP    NOT NULL,
    event_end           TIMESTAMP    NOT NULL,
    event_published     TIMESTAMP    NOT NULL,
    event_start         TIMESTAMP    NOT NULL,
    fb_published        INT          NOT NULL,
    gts                 REAL         NOT NULL,
    has_analytics       BOOLEAN      NOT NULL,
    has_header          VARCHAR(100) NOT NULL,
    has_logo            BOOLEAN      NOT NULL,


data_layer: https://github.com/sfischbuch/dsi-fraud-detection-case-study/blob/master/eventdb.py

data_pump: https://github.com/sfischbuch/dsi-fraud-detection-case-study/blob/master/event_run.py
'''

def get_data():
    accountID = os.environ['oanda_demo_id']
    access_token = os.environ['oanda_demo_api']
    
    client = oandapyV20.API(access_token=access_token)
    
    granularities = ['D']
    
     #['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3','H4', 'H6', 'H8', 'H12', 'D']
    
    granularities = granularities[::-1]
    
    instru = 'EUR_USD'
    
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    
    for gran in granularities:
    
        i=0
        hit_today = False
        df = pd.DataFrame(columns=columns)
        df_next = pd.DataFrame(columns=columns)
        last_timestamp = '2005-01-01T00:00:00.000000000Z'
    
        while not hit_today:
    
            params = {'price': 'M', 'granularity': gran, 'count': 5000,
                      'from': last_timestamp,
                      'alignmentTimezone': 'America/New_York'}
            r = instruments.InstrumentsCandles(instrument=instru,params=params)
            client.request(r)
            resp = r.response
            i+=1
            print(r.status_code, i)
            data = []
            for can in resp['candles']:
                data.append([can['time'], can['volume'], can['mid']['c'], can['mid']['h'], can['mid']['l'], can['mid']['o'], can['complete']])
            df_next = pd.DataFrame(data, columns=columns)
            df = df.append(df_next, ignore_index=True)
            last_timestamp = list(df.time)[-1]
            last_month = list(df.time)[-1][:7]
            if last_month == '2017-09':
                hit_today = True
    
        df.drop_duplicates('time', keep='first', inplace=True)
        save_name = instru+'_'+gran
        print(save_name, df.shape)
        df.to_pickle('data/'+save_name)
    
def clean_data(df):
    df['time'] = pd.to_datetime(df['time'])
    df['volume'] = df.volume.astype(int)
    df['close'] = df.close.astype(float)
    df['high'] = df.high.astype(float)
    df['low'] = df.low.astype(float)
    df['open'] = df.open.astype(float)
    df['complete'] = df.complete.astype(bool)
    return df

def create_db():
    conn = pg2.connect(dbname='applesauce')
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE DATABASE forex;')
    cur.close()
    conn.close()
    pass

def add_table(instrument_time):
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = '''
    CREATE TABLE {} (
        time timestamptz, 
        volume int,
        close real,
        high real,
        low real,
        open real,
        complete bool
    );
    '''.format(instrument_time)
    cur.execute(query)
    cur.close()
    conn.close()
    pass
    

if __name__ == '__main__':
    
    all_files = glob.glob('../data/*')
    all_files.sort(key=os.path.getmtime)
    df = pd.read_pickle(file_path_name)
    df = clean_data(df)
    df.to_pickle(file_path_name)
    
    pass
    