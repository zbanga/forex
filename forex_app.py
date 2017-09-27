from flask import Flask, request, render_template
import json
import requests
import socket
from datetime import datetime
import pickle
import time
import pandas as pd
import numpy as np
from src.oandadatapostgres import return_data_table_gt_time

'''
SSH 22
HTTP 80
Custom TCP Rule 8080
RabitMQ
https://www.tradingview.com/widget/
'''

app = Flask(__name__)

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/tables.html')
def tables():
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    data = return_data_table_gt_time('eur_usd_m1', '2017-09-26T00:00:00.000000000Z')
    return render_template('tables.html', data=data)

@app.route('/flot.html')
def flot():
    return render_template('flot.html')

@app.route('/morris.html')
def morris():
    return render_template('morris.html')

@app.route('/forms.html')
def forms():
    return render_template('forms.html')

@app.route('/panels-wells.html')
def panelswells():
    return render_template('panels-wells.html')

@app.route('/buttons.html')
def buttons():
    return render_template('buttons.html')

@app.route('/notifications.html')
def notifications():
    return render_template('notifications.html')

@app.route('/typography.html')
def typography():
    return render_template('typography.html')

@app.route('/icons.html')
def icons():
    return render_template('icons.html')

@app.route('/liveprediction.html')
def grid():
    columns=['time', 'open', 'high', 'low', 'close', 'volume', 'table_name','y_pred', 'y_pred_proba']
    data = pick = pickle.load(open('picklehistory/live_results_df.pkl', 'rb'))
    data = data.values
    return render_template('liveprediction.html', data=data)

@app.route('/blank.html')
def blank():
    return render_template('blank.html')


@app.route('/login.html')
def login():
    return render_template('login.html')

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8080, debug=True)


    pass
