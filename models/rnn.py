# -*- coding: utf-8 -*-


'''
RNNs
vanishing gradient problem
Long Short Term Memory get rid of vanishing gradient problem
There are 4 neurons / gates in 
Sliding windows of 50...
3 dimensional array
    1. total windows = 4k
    2. window size = 50
    3. dimensionality of the data (1d just price). Can add many inputs (price, volatility, volume, indicator)
        might need dimensionality reduction
after 50 it is predicting on the predictions 
make windows and shuffle windows

https://youtu.be/l4X-kZjl1gs?t=29m34s


'''

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# sin wave





