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

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

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
    X_train_poly = fit_polynom(X_train_0, 3)
    X_train = normalize_data(X_train_poly)
    mlb = LabelBinarizer()
    y_train = mlb.fit_transform(lat_labels_train) 
    
    #testing - mixtures
    X_test_0 = np.array(load_data("data/data_test.txt"))
    lat_labels_list = load_labels("data/labels_test.txt")
    print ("initial test data: ", np.array(X_test_0).shape)
    X_test_poly = fit_polynom(X_test_0, 3)
    X_test = normalize_data(X_test_poly)
    mlb1 = MultiLabelBinarizer()
    lat_labels_list.append(lat_labels_train)
    y_test_bin_labels = mlb1.fit_transform(lat_labels_list)
    y_test = y_test_bin_labels[:-1]

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
 
def model_evaluation(X_train, y_train):
    processors=1
    scoring='log_loss'
    
    kfold = KFold(n=len(X_train), n_folds=2, random_state=7)

    # Prepare some basic models
    models = []
    models.append(('LogRegr', OneVsRestClassifier(LogisticRegression())))
    models.append(('Knn', OneVsRestClassifier(KNeighborsClassifier())))
    models.append(('DecTree', OneVsRestClassifier(DecisionTreeClassifier())))
    models.append(('NaiveBayess', OneVsRestClassifier(GaussianNB())))
    models.append(('RFC', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_features=10))))
    models.append(('SVM', OneVsRestClassifier(SVC(probability=True))))
    models.append(('ExtraTrees', OneVsRestClassifier(ExtraTreesClassifier())))
    models.append(('AdaBoost', OneVsRestClassifier(AdaBoostClassifier())))
    models.append(('Bagging', OneVsRestClassifier(BaggingClassifier())))

    # Evaluate each model in turn
    results = []
    names = []

    for name, model in models:
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=processors)
        results.append(cv_results)
        names.append(name)
        print("{0}: ({1:.3f}) +/- ({2:.3f})".format(name, cv_results.mean(), cv_results.std()))
 
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
    
    model_evaluation(X_train_2d, y_train)

    # #############################################
    # print ("#############################################")
    # #learn on compounds and mixtures and try toys
    # X_train_full, y_labels_train_full, X_new,mlb = load_testing_2()
    # X_train_full = np.array(X_train_full)
    # X_new = np.array(X_new)
    # nsamples, nx, ny = X_train_full.shape
    # X_train_full_2d = X_train_full.reshape((nsamples,nx*ny))    
    # nsamples2, nx2, ny2 = X_new.shape
    # X_test_full_2d = X_new.reshape((nsamples2,nx2*ny2))
    
    # X_train_full_2d = preprocessing.scale(X_train_full_2d)
    # X_test_full_2d = preprocessing.scale(X_test_full_2d)
    # #(X_train_full_2d, y_labels_train_full, X_test_full_2d,mlb) 

    
if __name__ == "__main__":
    main()
