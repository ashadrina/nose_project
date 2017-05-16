# -*- coding: utf-8 -*-
import os 
import sys
import numpy as np
import pandas as pd
import scipy
import copy 
import collections

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from numpy.linalg import svd

import warnings

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
            labels.append(set(line.replace("\n","").split(";")))
        else:
            labels.append(line.replace("\n",""))
    input_f.close()
    return labels
    
def load_labels_with_proba(in_file):
    input_f = open(in_file, "r")
    labels = []
    for line in input_f:
        pairs = line.replace("\n","").split(";")
        pairs2 = []
        for item in pairs:
            pairs2.append(tuple(item.split(":")))
        pairs2 = [(tuple(filter(None, p))) for p in pairs2]
        labels.append(pairs2)
    input_f.close()
    return labels

def create_mlb():
    y_train_lat_labels = load_labels("data/labels_train.txt")
    y_test_lat_labels = load_labels("data/labels_test.txt")
    y_test_lat_labels = ["_with_".join(i) for i in y_test_lat_labels]
    
    y_train_lat_big = []
    y_train_lat_big.extend(y_train_lat_labels)
    y_train_lat_big.extend(y_test_lat_labels)
    
    y_train_lat_big_list = []
    for i in y_train_lat_big:
        if type(i) is list:
            y_train_lat_big_list.append(i)
        else:
            y_train_lat_big_list.append([i])

    mlb = MultiLabelBinarizer()
    y_train_big =  mlb.fit_transform(y_train_lat_big_list) 

    return mlb

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

def detrend(x, order, FLAG):
    import numpy as np
    import scipy.signal as sps
    import matplotlib.pyplot as plt

    x = np.asarray(x)   
    if FLAG:
        plt.plot(x, label='original')

    # detect and remove jumps
    jmps = np.where(np.diff(x) < -0.5)[0]  # find large, rapid drops in amplitdue
    for j in jmps:
        x[j+1:] += x[j] - x[j+1] 
    if FLAG:        
        plt.plot(x, label='unrolled')

    # detrend with a low-pass
    x -= sps.filtfilt([1] * order, [order], x)  # this is a very simple moving average filter
    if FLAG:
        plt.plot(x, label='detrended')

        plt.legend(loc='best')
        plt.show()
    return x
    
def patch_detrend(X_train):
    order = 5
    START = 0#1
    print ("order: ", order)
    X_res = []
    for matr in X_train:
        matr_res = []
        for ch in matr:
            if START == 1:
                matr_res.append(detrend(ch, order, 1))
                START = 0
            else:
                matr_res.append(detrend(ch, order, 0))
        X_res.append(matr_res)
    return X_res

##################################
def model_evaluation(X_train, y_train): 
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    from sklearn.cross_validation import KFold
    kfold = KFold(n=len(X_train), shuffle=True, n_folds=7, random_state=42)
    return kfold
    
def estimation_svm(X_train, y_train, kfold, group, y_train_lat):    
    print ("parameter estimation for SVM")  
    from sklearn.svm import SVC  
    from sklearn.grid_search import GridSearchCV
    from sklearn.multiclass import OneVsRestClassifier
    Gammas = [0.001, 0.01, 1, 10]
    kernels = ['rbf','linear', 'sigmoid']
    decision_function = ['ovo', 'ovr']
    coefs = [0.0, 0.5, 1.0, 10.0, 100.0]
    class_weight = ['balanced'] #, None]
    Cs = [0.001, 0.01, 1, 10]
        
    svc_grid = GridSearchCV(estimator = OneVsRestClassifier(SVC(probability=True)), 
        param_grid = {'estimator__C': Cs, 'estimator__gamma': Gammas, 'estimator__kernel': kernels, 
        'estimator__decision_function_shape': decision_function,
        'estimator__coef0': coefs, 'estimator__class_weight': class_weight },
        cv = kfold, n_jobs = 1)
 
    svc_grid.fit(X_train, y_train)

    print ("best score:", svc_grid.best_score_)
    print ("best params: ", svc_grid.best_params_)
      
    y_train2 = load_labels(y_train_lat)
    best = {}
    for k, v in svc_grid.best_params_.items():
        r = k.replace("estimator__","")
        best[r] = v
    
    clf = SVC(**best)
    clf.fit(X_train, y_train2)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train2 != y_score)
    print ("svm train accuracy: ", 1 - err_train)
    
    from sklearn.metrics import confusion_matrix
    cnf_matrix_train = confusion_matrix(y_score, y_train2)    
    np.set_printoptions(precision=2)
    class_names = list(set(y_train2))
    plt.figure()
    plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')
    plt.savefig("output/system/"+group+"_confusion.png", dpi=100)
    return best
    
def svm_mlb(X_train, y_train, X_new, mlb, params):
    from sklearn.svm import SVC, NuSVC
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

    print ("SVM")
    clf = OneVsRestClassifier( SVC(probability=True, **params) )
    clf.fit(X_train, y_train)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("svm train accuracy: ", 1 - err_train)
    y_new_proba = clf.predict_proba(X_new)
    return (y_score, y_new_proba)

def estimation_rf(X_train, y_train, kfold, group, y_train_lat):    
    print ("parameter estimation for RF")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn.multiclass import OneVsRestClassifier
    n_est = [7, 15, 25, 50, 101, 500]
    criterion = ['gini', 'entropy']
    class_weight = ['balanced']

    rf_grid = GridSearchCV(OneVsRestClassifier(estimator = RandomForestClassifier()), 
        param_grid = {'estimator__n_estimators': n_est, 'estimator__criterion' : criterion, 'estimator__class_weight' : class_weight  },
        cv = kfold, n_jobs = 1)
 
    rf_grid.fit(X_train, y_train)

    print ("best score:", rf_grid.best_score_)
    print ("best params: ", rf_grid.best_params_)
    
    y_train2 = load_labels(y_train_lat)
    best = {}
    for k, v in rf_grid.best_params_.items():
        r = k.replace("estimator__","")
        best[r] = v
    
    clf = RandomForestClassifier(**best)
    clf.fit(X_train, y_train2)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train2 != y_score)
    print ("rf train accuracy: ", 1 - err_train)
    
    from sklearn.metrics import confusion_matrix
    cnf_matrix_train = confusion_matrix(y_score, y_train2)    
    np.set_printoptions(precision=2)
    class_names = list(set(y_train2))
    plt.figure()
    plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')
    plt.savefig("output/system/"+group+"_confusion.png", dpi=100)
    return best
    
def rf_mlb(X_train, y_train, X_new, mlb, params):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

    print ("RF")
    clf = OneVsRestClassifier( RandomForestClassifier(**params) )
    clf.fit(X_train, y_train)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("rf train accuracy: ", 1 - err_train)
    y_new_proba = clf.predict_proba(X_new)
    return (y_score, y_new_proba)
        
def estimation_lr(X_train, y_train, kfold, group, y_train_lat):
    print ("parameter estimation for Logistic Regression")
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    C = [0.001, 0.01, 0.1, 1, 5, 10, 50, 100] 
    solver = ['newton-cg', 'lbfgs']#, 'sag']
    penalty = [ 'l2'] #['l1', 'l2']
    class_weight = ['balanced']
    multi_class  = ['multinomial', 'ovr']

    lr_grid = GridSearchCV(
        estimator = OneVsRestClassifier( LogisticRegression()),
        param_grid = { 'estimator__penalty': penalty, 'estimator__C': C, 'estimator__solver': solver, 
            'estimator__multi_class': multi_class , 'estimator__class_weight': class_weight}, 
        cv = kfold,  n_jobs = 1)

    lr_grid.fit(X_train, y_train)

    print("best score: ", lr_grid.best_score_)
    print("best params: ", lr_grid.best_params_)
    
    y_train2 = load_labels(y_train_lat)
    best = {}
    for k, v in lr_grid.best_params_.items():
        r = k.replace("estimator__","")
        best[r] = v
    
    clf = LogisticRegression(**best)
    clf.fit(X_train, y_train2)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train2 != y_score)
    print ("LR train accuracy: ", 1 - err_train)
    
    from sklearn.metrics import confusion_matrix
    cnf_matrix_train = confusion_matrix(y_score, y_train2)    
    np.set_printoptions(precision=2)
    class_names = list(set(y_train2))
    plt.figure()
    plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')
    plt.savefig("output/system/"+group+"_confusion.png", dpi=100)
    return best
    
def lr_mlb(X_train, y_train, X_new, mlb, params):
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    clf = OneVsRestClassifier( LogisticRegression(**params) )
    clf.fit(X_train, y_train)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("LR train accuracy: ", 1 - err_train)
    y_new_proba = clf.predict_proba(X_new)
    return (y_score, y_new_proba)
##################################
def compute_av_metrics(y_train, y_score, y_new_proba, mlb, y_true_path):
    y_test_true_labels = load_labels(y_true_path)
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    if "true_labels_other" in y_true_path:
        y_test_true =  mlb.fit_transform(y_test_true_labels) 
        y_test_true = list(y_test_true)[:-1]      
    else:
        y_test_true =  mlb.fit_transform(y_test_true_labels) 

    from sklearn.metrics import coverage_error
    err1 = coverage_error(y_train, y_score)
    print ("You should predict top ",err1, " labels for train")
    err2 = coverage_error(y_test_true, y_new_proba)
    print ("You should predict top ",err2, " labels for toys")

    from sklearn.metrics import label_ranking_average_precision_score
    rap1 = label_ranking_average_precision_score(y_train, y_score) 
    print ("label_ranking_average_precision_score on train", rap1)
    rap2 = label_ranking_average_precision_score(y_test_true, y_new_proba) 
    print ("label_ranking_average_precision_score on toys", rap2)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def hamming_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len( set_true.intersection(set_pred) ) / float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def compute_final_metrics(y_true, y_pred):
    print('Hamming score: {0}'.format(hamming_score(y_true, y_pred))) 
    # Subset accuracy: 
    #1 if the prediction for one sample fully matches the gold
    # 0 otherwise.
    import sklearn.metrics
    print('Subset accuracy: {0}'.format(sklearn.metrics.accuracy_score(y_true, y_pred)))
    # Hamming loss (smaller is better)
    print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred))) 
    
def compute_ind_metrics(pred, true, mlb):
    y_new_predict = []
    for items in pred:
        c = []
        it = (filter(None, items))
        for item in it:
            c.append(item[0])
        
        y = []
        if len(set(c)) >= 8:
            common = [x for x in c if x != "other"]
            y.extend(common[:8])
        elif len(set(c)) >= 8 and len(set(c)) != 1:
            common = [x for x in c if x != "other"]
            y.extend(common)
        else:
            y.extend(list(set(c)))
        y_new_predict.append(y)
        
    y_new_true_labels = load_labels(true)
    y_new_true_labels = [list(filter(None, lab)) for lab in y_new_true_labels]
    if "true_labels_other" in true:
        y_test_true =  mlb.fit_transform(y_new_true_labels) 
        y_new_true_labels = list(y_test_true)[:-1]      
    else:
        y_new_true_labels =  mlb.fit_transform(y_new_true_labels) 
        
    y_new_predict =  mlb.transform(y_new_predict) 

    y_new_predict = np.array(y_new_predict)
    y_new_true_labels = np.array(y_new_true_labels)

    compute_final_metrics(y_new_true_labels, y_new_predict)

def save_res(y_new, group):
    txt_outfile = open("output/system/"+group+".txt", 'w')
    for l in y_new:
        if len(l) != 1:
            r = ";".join(l)
        else:
            r = str(l[0])+";"
        txt_outfile.write(str(r)+"\n")
    txt_outfile.close()
        
def save_group(y_new, group):
    txt_outfile = open("output/system/"+group+".txt", 'w')
    for l in y_new:
        if len(l) > 1:
            pairs = []
            for rr in l:
                pairs.append(str(rr[0])+":"+str(rr[1]))
            r = ";".join(pairs)
        else:
            r = ":".join(l[0])
            r += ";"
        txt_outfile.write(str(r)+"\n")
    txt_outfile.close()
##################################
def main():
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    mlb = create_mlb()
    
    datasets = [["data/new/data_all.txt", "data/new/labels_dof_vs_other.txt", "data/data_new.txt", "data/new/true_labels_dof_vs_other.txt", 0.4, "Group1"], 
       ["data/new/data_all.txt", "data/new/labels_av_vs_other.txt", "data/data_new.txt", "data/new/true_labels_av_vs_other.txt", 0.25, "Group2"],
        ["data/new/data_all.txt", "data/new/labels_only_other2.txt", "data/data_new.txt", "data/new/true_labels_other.txt", 0.2, "Group3"]]
    
    #detect only DOF -> detect only  classes 2-4 -> detect rare classes
    
    for dataset in datasets:
        X_train_path = dataset[0]
        y_train_path = dataset[1]
        X_new_path = dataset[2]
        y_true_path = dataset[3]
        thr = dataset[4]
        group = dataset[5]

        print ("Learning on: ", group)
        
        X_train  = load_data(X_train_path)
        y_train  = load_labels(y_train_path)
        X_new  = load_data(X_new_path)
        
        y_train_lat_list = []
        for i in y_train:
            y_train_lat_list.append([i])
        y_train =  mlb.fit_transform(y_train_lat_list) 
             
        ###################################  
        X_train = normalize_data(X_train)    
        X_train = patch_detrend(X_train)
        X_train = np.array(X_train)
        nsamples00, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples00,nx*ny))        
        ###################################  
        X_new = normalize_data(X_new)    
        X_new = patch_detrend(X_new)    
        X_new = np.array(X_new)
        nsamples22, nx, ny = X_new.shape
        X_new = X_new.reshape((nsamples22,nx*ny))      
        ###################################
        X_train = preprocessing.scale(X_train)
        X_new = preprocessing.scale(X_new)
        print (np.array(X_train).shape)
        print (np.array(X_new).shape)
        ###################################
        kfold = model_evaluation(X_train, y_train)
        
        params = estimation_svm(X_train, y_train, kfold, group, y_train_path)
        (y_score, y_new_proba) = svm_mlb(X_train, y_train, X_new, mlb, params)

        #params = estimation_rf(X_train, y_train, kfold, group, y_train_path)
        #(y_score, y_new_proba) = rf_mlb(X_train, y_train, X_new, mlb, params)

       # params = estimation_lr(X_train, y_train, kfold, group, y_train_path)
       # (y_score, y_new_proba) = lr_mlb(X_train, y_train, X_new, mlb, params)
       
        compute_av_metrics(y_train, y_score, y_new_proba, mlb, y_true_path)
       
        y_solution = []
        for y_pred,i in zip(y_new_proba,range(len(X_new))):
            r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb.classes_,y_pred)]
            sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
            fin = []
            for item in sorted_by_second_1:
                if float(item[1]) > float(thr):
                    fin.append(item)
            y_solution.append(fin)
       
        save_group(y_solution, group)
        compute_ind_metrics(y_solution, y_true_path, mlb)
        print ("--------------------------------------------------")

    G1  = load_labels_with_proba("output/system/Group1.txt")
    G2  = load_labels_with_proba("output/system/Group2.txt")
    G3  = load_labels_with_proba("output/system/Group3.txt")       
    y_new_true_labels = load_labels("data/new/true_labels.txt")
     
    y_new_predict = []
    for g1,g2,g3 in zip(G1,G2,G3):
        common = []
        common.extend(g1)
        common.extend(g2)
        common.extend(g3)
        common = list( filter(None, common) )
                
        c = []
        sorted_by_second_1 = sorted(common, key=lambda tup: tup[1], reverse=True)
        for item in sorted_by_second_1:
            c.append(item[0])
            
        if len(set(c)) >= 8:
            common = [x for x in c if x != "other"]
            y_new_predict.append(common[:8])
        elif len(set(c)) >= 8 and len(set(c)) != 1:
            common = [x for x in c if x != "other"]
            y_new_predict.append(common)
        else:
            y_new_predict.append(set(c))

    y_new_predict = [list(filter(None, lab)) for lab in y_new_predict]
    save_res(y_new_predict, "FINAL")
    
    y_new_true_labels = [list(filter(None, lab)) for lab in y_new_true_labels]
    y_new_true_labels.append(['benzin'])
    y_new_true_labels =  mlb.fit_transform(y_new_true_labels) 
    y_new_true_labels = list(y_new_true_labels)[:-1]
    y_new_predict =  mlb.transform(y_new_predict) 
    
    y_new_predict = np.array(y_new_predict)
    y_new_true_labels = np.array(y_new_true_labels)
    
    compute_final_metrics(y_new_true_labels, y_new_predict)


if __name__ == "__main__":
    main()