# -*- coding: utf-8 -*-

import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
#warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(filename, seq_len, normalise_window):
    '''
    4171 rows of stock data, seq_len = 50, normalise_window = True
    result = 'windows' of stock data at each price. shape: (4121, 51)
    normalize will divide each price by the first stock price in the window
    train is 90% of windows that are randomly shuffled
    x_train: all rows/windows and all but the last price. shape: (3709, 50)
    y_train: all rows/windows and just the last price at index 51. shape: (3709,)
    test: other 10% of windows
    x_test: all rows/windows and all but the last price. shape: (412,50)
    y_test: all rows/windows and just the last price at index 51. shape: (412,)
    reshape x_train and x_test to (N,W,F)
    3 dimensions in LSTM (N = Number of training sequences,W = sequence length,F = number of features)
    '''
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
    print(x_train.shape, x_test.shape)
    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    '''
    for each window, divide each price my the first price in the window and subtract 1
    aka percent change from 0
    '''
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    '''
    example [1,50,100,1]
    input layer: size 1 of sequence 50
    layer 1: 50 neurons
    layer 2: 100 neurons
    layer 3: 1 neuron with linear activation function for prediction
    '''
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    #model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    #model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    '''
    Input:
        model: keras net
        data: x_test
    Output:
        predicted:
    -----------------
    for each window of 50 previous true data to only the next timestep
    predicted shape (412,)
    -----------------
    '''
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    '''
    Input:
        model: keras net
        data: x_test
        window_size: 50
    Output:
        predicted: 
    ------------------
    for each window predict the next price
    shift the window 1 and add the price to the end
    predict on new window
    ------------------
    '''
    curr_frame = data[0] # (50,1)
    predicted = [] #(len x_test)
    for i in range(len(data)):
        pred_data = curr_frame[newaxis,:,:] #(1,50,1)
        prediction = model.predict(pred_data) #(1,1)
        predicted.append(prediction[0,0])
        curr_frame = curr_frame[1:] #(49,1)
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0) #insert prediction on end (50,1)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    '''
    Input:
        model: keras net
        data: x_test
        window_size: 50
        prediction_len: 50
    Output:
        predicted: 
    Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    for 412/50 = 8 individual predictions
    predict next 50 timesteps adjusting and using predictions to predict with
    
    
    '''
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            pred_data = curr_frame[newaxis,:,:]
            prediction = model.predict(pred_data)
            predicted.append(prediction[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

















if __name__ == '__main__':
    pass