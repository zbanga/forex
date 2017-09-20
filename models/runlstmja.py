# -*- coding: utf-8 -*-

'''
watch -n 1 nvidia-smi


'''


import lstmja
import time
import matplotlib.pyplot as plt
import numpy as np

def plot_results(predicted_data, true_data, figtitle):
    '''
    use when predicting just one analysis window
    '''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.title(figtitle)
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len, figtitle):
    '''
    use when predicting multiple analyses windows in data
    '''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        if i != 0:
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
    plt.title(figtitle)
    plt.show()

#Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 10  # suggest 100 for sine wave, 10 for stock
    seq_len = 50 # suggest using 25 for sine wave, 50 for stock

    print('> Loading data... ')

    # choose either the sine wave data or stock data
    #x_train, y_train, x_test, y_test = lstmja.load_data('../data/sinwave.csv', seq_len, False) # data is sine wave
    x_train, y_train, x_test, y_test = lstmja.load_data('../data/sp500.csv', seq_len, True) # data is a stock, normalize data is True
    
    print('> Data Loaded. Compiling...')

    model = lstmja.build_model([1, seq_len, 100, 1]) # 1 input layer, layer 1 has seq_len neurons, layer 2 has 100 neurons, 1 output

    model.fit(
        x_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)

    print('> Completed.')
    print('Training duration (s) : ', time.time() - global_start_time)
    
    predict_point_by_point = lstmja.predict_point_by_point(model, x_test)
    plot_results(predict_point_by_point, y_test, '50 Actual and Predict 1 Next Price') #Always use 50 Act to Predict 1 Next
    predict_full = lstmja.predict_sequence_full(model, x_test, seq_len)
    plot_results(predict_full, y_test, '50 Actual and Predict All Next Prices') #Use 50 first and predict all next using predictions to predict on
    predict_sequences = lstmja.predict_sequences_multiple(model, x_test, seq_len, seq_len) #predict 
    plot_results_multiple(predict_sequences, y_test, seq_len, '50 Actual and Predict 50 Next Prices') # prediction, true data, prediction length)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass