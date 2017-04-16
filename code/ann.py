import os 
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import copy 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

from sklearn import preprocessing

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


def load_data(in_file):
    input_f = open(in_file, "r")
    matrix = []
    for line in input_f:
        channels = [] #get channels
        for l in line.split("|"):
            samples = l.split(";")
            channels.append([float(i) for i in samples])
        matrix.append(channels)
        del channels
    input_f.close()
    return matrix

def load_labels(in_file):
    input_f = open(in_file, "r")
    labels = []
    for line in input_f:
        if ";" in line:
            labels.append(line.replace("\n","").split(";"))
        else:
            labels.append(line.replace("\n",""))
    input_f.close()
    return labels
    
def load_dataset():
    X_train_data = load_data("data/data_train.txt")
    y_train_lat_labels = load_labels("data/labels_train.txt")
    print ("initial data: ", np.array(X_train_data).shape)

    X_test_data = load_data("data/data_test.txt")
    y_test_lat_labels = load_labels("data/labels_test.txt")
    y_test_lat_labels = ["_with_".join(i) for i in y_test_lat_labels]    
    print ("initial data: ", np.array(X_test_data).shape)
    
    ##########################################
    X_train_big = []
    X_train_big.extend(X_train_data)
    X_train_big.extend(X_test_data)
    X_train_big = np.array(X_train_big)

    y_train_lat_big = []
    y_train_lat_big.extend(y_train_lat_labels)
    y_train_lat_big.extend(y_test_lat_labels)
    
    #y_train_lat_big_list = []
    #for i in y_train_lat_big:
    #    y_train_lat_big_list.append([i])

    #mlb = MultiLabelBinarizer()
    #y_train_big =  mlb.fit_transform(y_train_lat_big_list) 

    X_new_data = load_data("data/data_new.txt")
    print ("initial data: ", np.array(X_new_data).shape)
    #return X_train_big, y_train_big, X_new_data, mlb
    return X_train_big, y_train_lat_big, X_new_data

##########################################
    
def normalize_data(data):
    norm_matrix = []
    for block in data:
        #current_max = np.amax(block)
        norm_col = []
        for col in block:
            current_mean = np.mean(col)
            surrent_std = np.std(col)
            norm_col.append([(float(i) - current_mean)//surrent_std for i in col])
        norm_matrix.append(norm_col)
    return norm_matrix

def detrend(x):
    import numpy as np
    import scipy.signal as sps
    import matplotlib.pyplot as plt

    x = np.asarray(x)    
    #plt.plot(x, label='original')

    # detect and remove jumps
    jmps = np.where(np.diff(x) < -0.5)[0]  # find large, rapid drops in amplitdue
    for j in jmps:
        x[j+1:] += x[j] - x[j+1]    
    #plt.plot(x, label='unrolled')

    # detrend with a low-pass
    order = 20
    x -= sps.filtfilt([1] * order, [order], x)  # this is a very simple moving average filter
    #plt.plot(x, label='detrended')

    #plt.legend(loc='best')
    #plt.show()
    return x
    
def patch_detrend(X_train):
    X_res = []
    for matr in X_train:
        matr_res = []
        for ch in matr:
            matr_res.append(detrend(ch))
        X_res.append(matr_res)
    return X_res
 
def fit_polynom(X_train, N):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    X_train_new = []
    for matr in X_train:
        matr_new = []
        for i in range(len(matr)):
            vec = matr[i]
            L = len(vec)            
            T = 1/250.0       
            t = np.linspace(1,L,L)*T   
            xx = np.asarray(t)
            yy = np.asarray(vec)
            z = np.asarray(np.polyfit(xx, yy, N))
            ff = np.poly1d(z)
            x_new = np.linspace(xx[0], xx[-1], len(xx))
            y_new = ff(x_new)
            matr_new.append(y_new)
        X_train_new.append(matr_new)
    return X_train_new
    
########################################## 

def create_model_fully_connected():
    model = Sequential()
    model.add(Dense(968, input_dim=968, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7)) 
    model.add(Dense(968, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(121, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model
  
def parameter_est_fully_connected(model, X_train, y_train):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = np_utils.to_categorical(encoded_Y)
    batch_size = [4, 10, 20, 40]
    epochs = [10, 30, 50, 70, 100]
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    print (np.array(X_train).shape, np.array(dummy_y).shape)
    grid_result = grid.fit(X_train, dummy_y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
def run_model_fully_connected(model, X_train, y_train, X_new):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = np_utils.to_categorical(encoded_Y)    

    print ("Training model...")
    model.fit(X_train, dummy_y, nb_epoch=100, batch_size=5, verbose=2, shuffle=False)
    y_score = model.predict(X_train)
    print ("Training model... DONE")
    
    y_test_true = load_labels("true_labels.txt")
    y_new_proba = []
    for x,y_tr in zip(X_new, y_test_true):
        probabilities = model.predict_proba(x.reshape(1,968))
        y_new_proba.append(probabilities[0])
        probabilities = [float(p) for p in probabilities[0]]
        r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(encoder.classes_,probabilities)]
        sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
        #print (sorted_by_second_1)
        #print (y_tr)
        #print ("-----------------------------------")

    from sklearn.preprocessing import MultiLabelBinarizer
  
    ytrain = []
    for i in y_train:
        ytrain.append([i])   
        
    yscore = []
    for i in y_score:
        yscore.append([i])    
                     

    mlb = MultiLabelBinarizer()
    ytrain =  mlb.fit_transform(ytrain) 
    yscore =  mlb.fit_transform(yscore) 
   # y_new_proba =  mlb.fit_transform(y_new_proba) 
   # print (y_new_proba)
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
    y_test_true =  mlb.fit_transform(y_test_true_labels) 
    y_test_true = list(y_test_true)[:-1]   
  #  print (y_new_proba)

    from sklearn.metrics import coverage_error
    err1 = coverage_error(ytrain, yscore)
    print ("You should predict top ",err1, " labels for train")
    err2 = coverage_error(y_test_true, y_new_proba)
    print ("You should predict top ",err2, " labels for toys")

    from sklearn.metrics import label_ranking_average_precision_score
    rap1 = label_ranking_average_precision_score(ytrain, yscore) 
    print ("label_ranking_average_precision_score on train", rap1)
    rap2 = label_ranking_average_precision_score(y_test_true, y_new_proba) 
    print ("label_ranking_average_precision_score on toys", rap2)

##########################################
    
def get_lookback(a, lsize):
    a=a.tolist()
    new_a = []
    for item in a:
        st=a.index(item)
        if st-lsize < 0:
            arr = []
            for i in range(0, lsize):
                arr.append(a[st-lsize+i])
        else:
            arr=a[st-lsize:st]
        new_a.append(arr)
    return np.asarray(new_a)

def create_model_lstm():
    look_back = 3
    b_size = 5
    n_classes = 20
    epoches = 100
    
    #define model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, batch_input_shape=(b_size, look_back, 968), stateful=True))  # returns a sequence of vectors
    model.add(BatchNormalization())
    model.add(Dropout(0.7)) 
    model.add(LSTM(128, batch_input_shape=(b_size, look_back, 968), stateful=True, dropout_W=0.2, dropout_U=0.2))  # return a single vector 
    model.add(Dropout(0.3)) 
    model.add(Dense(n_classes, activation='sigmoid'))

    print ("Compiling model...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print ("Compiling model... DONE")    
    return model
    
def lstm_rnn(X_train, y_train, X_test):
    look_back = 3
    b_size = 5
    n_classes = 20
    epoches = 100

    print ("Preparing data...")
    X_train = get_lookback(X_train, look_back)
    X_test =  get_lookback(X_test, look_back)

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = np_utils.to_categorical(encoded_Y, n_classes)
    
    print (X_train.shape, dummy_y.shape, X_test.shape)

    model = create_model_lstm()
        
    print ("Training model...")
    for i in range(epoches):
        model.fit(X_train, dummy_y, nb_epoch=1, batch_size=b_size, verbose=2, shuffle=False)
        model.reset_states()
    print ("Training model... DONE")

    #model performance estimation
    train_loss, train_acc = model.evaluate(X_train, dummy_y, batch_size=b_size, verbose=0)
    model.reset_states()
    print ("Train score: "+str(train_acc)+", loss: "+str(train_loss))

    #predictions for training and test
    y_score = model.predict_classes(X_train, batch_size=b_size)
    
    y_test_true = load_labels("true_labels.txt")
    y_new_proba = model.predict_proba(X_test,  batch_size=b_size)
    for x,y_tr in zip(y_new_proba, y_test_true):
        probabilities = [float(p) for p in x]
        r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(encoder.classes_,probabilities)]
        sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
        print (sorted_by_second_1)
        print (y_tr)
        print ("-----------------------------------")
    
    from sklearn.preprocessing import MultiLabelBinarizer
  
    ytrain = []
    for i in y_train:
        ytrain.append([i])   
        
    yscore = []
    for i in y_score:
        yscore.append([i])    
                     
    mlb = MultiLabelBinarizer()
    ytrain =  mlb.fit_transform(ytrain) 
    yscore =  mlb.fit_transform(yscore) 
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
    y_test_true =  mlb.fit_transform(y_test_true_labels) 
    y_test_true = list(y_test_true)[:-1]   

    from sklearn.metrics import coverage_error
    err1 = coverage_error(ytrain, yscore)
    print ("You should predict top ",err1, " labels for train")
    err2 = coverage_error(y_test_true, y_new_proba)
    print ("You should predict top ",err2, " labels for toys")

    from sklearn.metrics import label_ranking_average_precision_score
    rap1 = label_ranking_average_precision_score(ytrain, yscore) 
    print ("label_ranking_average_precision_score on train", rap1)
    rap2 = label_ranking_average_precision_score(y_test_true, y_new_proba) 
    print ("label_ranking_average_precision_score on toys", rap2)


    
def main():
    #X_train, y_train, X_new, mlb = load_dataset()
    X_train, y_train, X_new = load_dataset()

    X_train = normalize_data(X_train)    
    X_train = patch_detrend(X_train)    

    X_train = np.array(X_train)
    nsamples00, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples00,nx*ny))        

    X_new = normalize_data(X_new)    
    X_new = patch_detrend(X_new)    

    X_new = np.array(X_new)
    nsamples22, nx, ny = X_new.shape
    X_new = X_new.reshape((nsamples22,nx*ny))        

    X_train = preprocessing.scale(X_train)
    X_new = preprocessing.scale(X_new)
    
    #model = KerasClassifier(build_fn=create_model_fully_connected, verbose=0)
    #parameter_est_fully_connected(model, X_train, y_train)
    #run_model_fully_connected(model, X_train, y_train, X_new)
    
    lstm_rnn(X_train, y_train, X_new)
    
if __name__ == "__main__":
    main()
