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
from sklearn import preprocessing

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
    
    y_train_lat_big_list = []
    for i in y_train_lat_big:
        y_train_lat_big_list.append([i])

    mlb = MultiLabelBinarizer()
    y_train_big =  mlb.fit_transform(y_train_lat_big_list) 

    X_new_data = load_data("data/data_new.txt")
    print ("initial data: ", np.array(X_new_data).shape)
    return X_train_big, y_train_big, X_new_data, mlb

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
 
def model_evaluation(X_train, y_train): 
    kfold = KFold(n=len(X_train), shuffle=True, n_folds=7, random_state=42)

    # # Prepare some basic models
    # models = []
    # models.append(('LogRegr', OneVsRestClassifier(LogisticRegression())))
    # models.append(('Knn', OneVsRestClassifier(KNeighborsClassifier())))
    # models.append(('NaiveBayess', OneVsRestClassifier(GaussianNB())))
    # models.append(('RFC', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_features=10))))
    # models.append(('SVM', OneVsRestClassifier(SVC(probability=True))))
    # models.append(('ExtraTrees', OneVsRestClassifier(ExtraTreesClassifier())))
    # models.append(('AdaBoost', OneVsRestClassifier(AdaBoostClassifier())))

    # Evaluate each model in turn
    # for name, model in models:
        # cv_results = cross_val_score(model, X_train, y_train, cv=kfold)#, scoring=scoring, n_jobs=1)
        # print("{0}: ({1:.3f}) +/- ({2:.3f})".format(name, cv_results.mean(), cv_results.std()))
        
    return kfold

def estimation_log_reg(X_train, y_train, kfold):
    print ("parameter estimation for Logistic Regression")
    from sklearn.grid_search import GridSearchCV
    lr_grid = GridSearchCV(
        estimator = LogisticRegression(random_state=5),
        param_grid = { 'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 1, 10, 100] }, 
        cv = kfold, 
        n_jobs = 1)

    lr_grid.fit(X_train, y_train)

    print("best score: ", lr_grid.best_score_)
    print("best params: ", lr_grid.best_params_)

def estimation_svm(X_train, y_train, kfold):
    print ("parameter estimation for SVM")    
    from sklearn.grid_search import GridSearchCV
    #Cs = 10.**np.arange(-5, 5)
    Cs = [0.001, 0.01, 1, 10, 100]
 #   Gammas = 10.**np.arange(-5, 5)
    Gammas = [0.001, 0.01, 1, 10, 100]
    kernels = ['rbf','linear','poly']

    svc_grid = GridSearchCV(estimator = SVC(random_state=7), 
        param_grid = {'C': Cs, 'gamma': Gammas, 'kernel':kernels},
        cv = kfold, 
        n_jobs = 1)
 
    svc_grid.fit(X_train, y_train)

    best_cv_err = 1 - svc_grid.best_score_
    best_C = svc_grid.best_estimator_.C
    best_Gamma = svc_grid.best_estimator_.gamma
    best_kernel = svc_grid.best_estimator_.kernel

    print ("best score", best_cv_err)
    print ("C = "+best_C+", gamma = "+best_Gamma+", kernel = "+best_kernel)
   
 
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

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

 
def main():
    X_train, y_train, X_new, mlb = load_dataset()

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

    print (np.array(X_train).shape)
    print (np.array(X_new).shape)

    y_train_lat_labels = load_labels("data/labels_train.txt")
    y_test_lat_labels = load_labels("data/labels_test.txt")
    y_test_lat_labels = ["_with_".join(i) for i in y_test_lat_labels]    

    y_train = []
    y_train.extend(y_train_lat_labels)
    y_train.extend(y_test_lat_labels)
    
    ########################
    
    y_train_over = load_labels("data/imbalanced/labels_train_over.txt")
    X_train_over = load_data("data/imbalanced/data_train_over.txt")
    X_train_over = np.array(X_train_over)
    nsamples00, nx, ny = X_train_over.shape
    X_train_over = X_train_over.reshape((nsamples00,nx*ny))   
    print (np.array(X_train_over).shape)
    
    kfold = model_evaluation(X_train_over, y_train_over)  
    estimation_log_reg(X_train_over, y_train_over, kfold)
    estimation_svm(X_train_over, y_train_over, kfold)
    
    # from sklearn.metrics import confusion_matrix
    # cnf_matrix_train = confusion_matrix(y_train, y_score)    
    
    # np.set_printoptions(precision=2)
    # class_names = list(set(y_train))

    # plt.figure()
    # plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')

    # plt.show()
    

    
if __name__ == "__main__":
    main()
