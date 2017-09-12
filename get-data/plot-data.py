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


file_name = sys.argv[1]
df = pd.read_pickle('data/'+file_name)
df['time'] = pd.to_datetime(df['time'])
df['close'] = df.close.astype(float)
df.sort_values('time', inplace=True)
fig, ax = plt.subplots()
ax.plot(df.time, df.close)
ax.set_title(file_name)
plt.show()
