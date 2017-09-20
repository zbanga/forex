#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:20:22 2017

@author: applesauce
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_price(df):
    fig, ax = plt.subplots()
    ax.plot(df['open'])
    ax.set_title(file_name)
    plt.show()


if __name__ == '__main__':
    file_name = 'EUR_USD_M1'
    df = pd.read_pickle('../data/'+file_name)
    df.sort_values('time', inplace=True)
    df.set_index('time', inplace=True)
    df = df.loc[datetime(2015, 1, 1):, :]
    plot_price(df)
    pass