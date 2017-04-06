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

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from numpy.linalg import svd
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA

from sklearn import preprocessing
from keras.models import Model
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense
from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy 

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

def normalize_data(data):
    norm_matrix = []
    for block in data:
        norm_col = []
        for col in block:
            current_mean = np.mean(col)
            surrent_std = np.std(col)
            norm_col.append([(float(i) - current_mean)//surrent_std for i in col])
        norm_matrix.append(norm_col)
    return norm_matrix

def patch_detrend(X_train):
    X_res = []
    for matr in X_train:
        matr_res = []
        for ch in matr:
            matr_res.append(detrend(ch))
        X_res.append(matr_res)
    return X_res
   
def load_train():    
  #  X_train_0 = np.array(load_data("data/data_train.txt"))
    X_train_0 = np.array(load_data("data/data_train_over.txt"))
  #  y_train = load_labels("data/labels_train.txt")
    y_train = load_labels("data/labels_train_over.txt")
    print ("initial train data: ", X_train_0.shape)
    from sklearn.cross_validation import train_test_split as tts
    X_train, X_test, y_train, y_test = tts(X_train_0, y_train, train_size=0.6, random_state=42)
#    X_train_poly = fit_polynom(X_train_0, 3)
 #   X_train = normalize_data(X_train)
 #   X_test = np.array(load_data("data/data_test.txt"))
 #   y_test = load_labels("data/labels_test.txt")
 #   print ("initial test data: ", np.array(X_test_0).shape)
 #   X_test_poly = fit_polynom(X_test_0, 3)
  #  X_test = normalize_data(X_teX_trainst_poly)
    return X_train, y_train, X_test, y_test
 
def load_testing_2():
    X_train_data = load_data("data/data_train.txt")
    y_train_lat_labels = load_labels("data/labels_train.txt")
   # print (len(set(y_train_lat_labels)))
    print ("initial data: ", np.array(X_train_data).shape)

    mlb = LabelBinarizer()
    y_train_bin_labels = mlb.fit_transform(y_train_lat_labels) 

    X_train_poly = fit_polynom(X_train_data, 3)
    X_train_2 = normalize_data(X_train_poly)

    X_test_data = load_data("data/data_test.txt")
    y_test_lat_labels = load_labels("data/labels_test.txt")
    y_test_lat_labels_2 = load_labels("data/labels_test.txt")
    print ("initial data: ", np.array(X_test_data).shape)
    
    X_test_poly = fit_polynom(X_test_data, 3)
    X_test_2 = normalize_data(X_test_poly)

    mlb1 = MultiLabelBinarizer()
    y_test_lat_labels.append(y_train_lat_labels)
    y_test_bin_labels = mlb1.fit_transform(y_test_lat_labels)
    y_test = y_test_bin_labels[:-1]
    
    ###################################
    X_train_big = []
    X_train_big.extend(X_train_2)
    X_train_big.extend(X_test_2)
    X_train_big = np.array(X_train_big)

    ll = [] #y_train
    for l in y_train_lat_labels:
        ll.append([l])

    y_train_lat_big = []
    y_train_lat_big.extend(ll)
    y_train_lat_big.extend(y_test_lat_labels_2)

    mlb2 = MultiLabelBinarizer()
    y_train_big =  mlb2.fit_transform(y_train_lat_big) 

    X_new_data = load_data("data/data_new.txt")
    print ("initial data: ", np.array(X_new_data).shape)
    
    X_new_poly = fit_polynom(X_new_data, 8)
    X_new_data_2 = normalize_data(X_new_poly)    
   
    X = []
    x = []
    pca = PCA(n_components=4)
    for matr in X_train_big:
        X.append(pca.fit_transform(matr))    
    for matr in X_new_data_2:
        x.append(pca.fit_transform(matr))
    return X, y_train_big, x, mlb1

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

def create_model():
    model = Sequential()
    model.add(Dense(480, input_dim=968, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(20, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
  
def parameter_est(model, X_train, y_train):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = np_utils.to_categorical(encoded_Y)
    batch_size = [10, 13, 15, 17, 20, 23, 25, 27, 30]
    epochs = [10, 13, 19, 23, 33, 50, 100]
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    print (np.array(X_train).shape, np.array(dummy_y).shape)
    grid_result = grid.fit(X_train, dummy_y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
def run_model(model, X_train, y_train, X_test, y_test):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = np_utils.to_categorical(encoded_Y)

    b_size = 11
    print ("Training model...")
    model.fit(X_train, dummy_y, nb_epoch=13, batch_size=b_size, verbose=2, shuffle=False)
    print ("Training model... DONE")

    train_loss, train_acc = model.evaluate(X_train, dummy_y, batch_size=b_size, verbose=0)
    print('Train Score: %.2f, loss %.2f' % (train_acc, train_loss))

    probabilities = model.predict(X_test)
    probabilities[probabilities>= 0.5] = 1
    probabilities[probabilities<0.5] = 0
    print (probabilities)
    print(encoder.inverse_transform(probabilities))

    
def main():
    #learn on compouns, test on mixtures
    X_train, y_train, X_test, y_test = load_train()
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    nsamples, nx, ny = X_train.shape
    X_train_2d = X_train.reshape((nsamples,nx*ny))    
    nsamples2, nx2, ny2 = X_test.shape
    X_test_2d = X_test.reshape((nsamples2,nx2*ny2))
    #(X_train_2d, y_train, X_test_2d, y_test)

    #model = create_model()    
    model = KerasClassifier(build_fn=create_model, verbose=0)
    parameter_est(model, X_train_2d, y_train)
    #run_model(model, X_train_2d, y_train, X_test_2d, y_test)
    #############################################
    #print ("#############################################")
    ##learn on compounds and mixtures and try toys
    #X_train_full, y_labels_train_full, X_new,mlb = load_testing_2()
    #X_train_full = np.array(X_train_full)
    #X_new = np.array(X_new)
    #nsamples, nx, ny = X_train_full.shape
    #X_train_full_2d = X_train_full.reshape((nsamples,nx*ny))    
    #nsamples2, nx2, ny2 = X_new.shape
    #X_test_full_2d = X_new.reshape((nsamples2,nx2*ny2))
    
    #X_train_full_2d = preprocessing.scale(X_train_full_2d)
    #X_test_full_2d = preprocessing.scale(X_test_full_2d)
    
    ##(X_train_full_2d, y_labels_train_full, X_test_full_2d,mlb) 
    
if __name__ == "__main__":
    main()
