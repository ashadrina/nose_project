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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# from keras.preprocessing import sequence
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Masking, TimeDistributedDense
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM
# from keras.utils.np_utils import to_categorical
# from keras.utils.visualize_util import plot

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
        current_max = np.amax(block)
        norm_col = []
        for col in block:
            norm_col.append([float(i)//current_max for i in col])
        norm_matrix.append(norm_col)
    return norm_matrix
   
def load_train():    
    #training - compounds
    X_train_0 = np.array(load_data("data/data_train.txt"))
    lat_labels_train = load_labels("data/labels_train.txt")
    print ("initial train data: ", X_train_0.shape)
    X_train_poly = fit_polynom(X_train_0, 5)
    X_train = normalize_data(X_train_poly)
    mlb = LabelBinarizer()
    y_train = mlb.fit_transform(lat_labels_train) 
    
    #testing - mixtures
    X_test_0 = np.array(load_data("data/data_test.txt"))
    lat_labels_list = load_labels("data/labels_test.txt")
    print ("initial test data: ", np.array(X_test_0).shape)
    X_test_poly = fit_polynom(X_test_0, 5)
    X_test = normalize_data(X_test_poly)
    mlb1 = MultiLabelBinarizer()
    lat_labels_list.append(lat_labels_train)
    y_test_bin_labels = mlb1.fit_transform(lat_labels_list)
    y_test = y_test_bin_labels[:-1]

    return X_train, y_train, X_test, y_test
  
def load_testing():
    X_train_full_0 = load_data("data/data_train.txt")
    X_train_full_0.extend(load_data("data/data_test.txt"))
    X_train_full_0 = np.array(X_train_full_0)
    
    lat_labels_train_full = list(load_labels("data/labels_train.txt"))
    lat_labels_train_list = list(load_labels("data/labels_test.txt"))
    for item in lat_labels_train_list:
        lat_labels_train_full.append("WITH".join(item))
        #print (item)
        #lat_labels_train_full.append(item)
        
    X_train_poly = fit_polynom(X_train_full_0, 5)
    X_train_full = normalize_data(X_train_poly)
    #print ("initial TRAIN data: ", np.array(X_train_full).shape, np.array(lat_labels_train_full).shape)
    mlb = LabelBinarizer()
    y_train_full = mlb.fit_transform(lat_labels_train_full) 
    
    #testing is new data
    X_new_0 = np.array(load_data("data/data_new.txt"))
    X_new_poly = fit_polynom(X_new_0, 5)
    X_new = normalize_data(X_new_poly)
    print ("initial NEW data: ", np.array(X_new).shape)
    
    return X_train_full, lat_labels_train_full, X_new, mlb
   
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

# def build_model(input_dim, output_classes):
    # model = Sequential()
    # model.add(Dense(input_dim=input_dim, output_dim=12, activation='sigmoid'))
    # model.add(Dropout(0.5))
    # model.add(Dense(output_dim=output_classes, activation='softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='adadelta')
    # return model


def run_svd(data):
	new_data = []
	for matr in data:
		U, s, V = svd(np.array(matr), full_matrices=True)
		new_data.append(s)	
	return new_data	
    
def knn_cl_training(X_train, y_train, X_test, y_test):
    knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
    err_train = np.mean(y_train != knn.predict(X_train))
    err_test  = np.mean(y_test  != knn.predict(X_test))
    print ("knn accuracy: ", 1 - err_train, 1 - err_test)
 
def knn_cl_testing(X_train, y_train, X_test):
    knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
    err_train = np.mean(y_train != knn.predict(X_train))
    print ("knn train accuracy: ", 1 - err_train)		
    y_new = knn.predict(X_test)
    txt_outfile = open("knn_new_labels.txt", 'w')
    for y in y_new:
        print (y)
        if y:
            txt_outfile.write(y+"\n")
        else:
            txt_outfile.write("?\n")
    txt_outfile.close()
 
def main():
    #learn on compouns, test on mixtures
    X_train, y_train, X_test, y_test = load_train()
    train_ress = run_svd(X_train)
    test_ress = run_svd(X_test)
    knn_cl_training(train_ress, y_train, test_ress, y_test)
    #############################################
    #print ("#############################################")
    #learn on compounds and mixtures and try toys
    X_train_full, y_labels_train_full, X_new,mlb = load_testing()
    train_ress = run_svd(X_train_full)
    test_ress = run_svd(X_new)
    knn_cl_testing(train_ress, y_labels_train_full, test_ress) 
    
if __name__ == "__main__":
    main()