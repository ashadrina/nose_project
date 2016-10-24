%matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import accuracy_score

from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn

import generate_data

RNN_LAYERS = 1
DENSE_LAYERS = [120, 60]
TRAINING_STEPS = 1000
BATCH_SIZE = 1

def main():
    if len (sys.argv) == 3:
        data_file = sys.argv[1]
        labels_file = sys.argv[2]

    print (data_file, labels_file)
    data = generate_data.load_data(data_file)
    lat_labels = generate_data.load_labels(labels_file)
    print (len(set(lat_labels)))


    mlb = LabelBinarizer()
    bin_labels = mlb.fit_transform(lat_labels) 
    
    print ("initial data: ", np.array(data).shape)
    ress = generate_data.run_svd(data)
    new_ress = generate_data.reduce_data(ress)
    print ("reduced data: ", np.array(new_ress).shape)
    
    X = []
    for dat in new_ress:
        X.extend(dat)
    #X = np.array(X)
    
    X_data = generate_data.load_data("data_val.txt")
    lat_labels_test = generate_data.load_labels("labels_val.txt")
    print ("initial data: ", np.array(X_data).shape)
    val_ress = run_svd(X_data)
    new_val_ress = generate_data.reduce_data(val_ress)
    print ("reduced data: ", np.array(new_val_ress).shape)
    
    X_1= []
    for dat in new_val_ress:
        X_1.extend(dat)
    #X_1 = np.array(X_1)
    
    X_1.extend(X)
    X_1 = np.array(X_1)
    ll = []
    for l in lat_labels:
        ll.append([l])
    lat_labels_test.extend(ll)
    mlb1 = MultiLabelBinarizer()
    bin_y =  mlb1.fit_transform(lat_labels_test) 
    bin_y_lar =  mlb1.inverse_transform(bin_y) 

    X_data_new = generate_data.load_data("data_test.txt")
    print ("initial data: ", np.array(X_data_new).shape)
    val_ress_new = generate_data.run_svd(X_data_new)
    new_val_ress_new = generate_data.reduce_data(val_ress_new)
    print ("reduced data: ", np.array(new_val_ress_new).shape)
    
    X_new = []
    for dat in new_val_ress_new:
        X_new.extend(dat)

    model = learn.TensorFlowEstimator(model_fn=lstm_model(RNN_LAYERS, DENSE_LAYERS),
                                     n_classes=16,
                                     steps=TRAINING_STEPS,
                                     batch_size=BATCH_SIZE)
    model.fit(X_1, bin_y)
    score = accuracy_score(X_new, model.predict(mlb1))
    print('Accuracy: {0:f}'.format(score))
    
    
def lstm_model(rnn_layers, dense_layers, learning_rate=0.03, optimizer='Adagrad'):

    def _lstm_model(X, y):
        cell = tf.nn.rnn_cell.BasicLSTMCell(120, state_is_tuple=True)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([cell]*rnn_layers, state_is_tuple=True)
        x_ = tf.unpack(X, axis=1, num=120)
        output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = tf.contrib.layers.stack(output[-1],
                                        tf.contrib.layers.fully_connected,
                                        dense_layers)
        return learn.models.linear_regression(output, y)

    return _lstm_model
    
if __name__ == "__main__":
    main()