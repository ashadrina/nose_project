import os 
import sys
import numpy as np
import copy 
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
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
    
def oversampling_data(X, y):
    from imblearn.over_sampling import ADASYN 
    RANDOM_STATE = 42
    ada = ADASYN(random_state=RANDOM_STATE)
    X_res, y_res = ada.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res, y_res
    
def undersampling_data(X, y):
    from imblearn.under_sampling import ClusterCentroids  
    RANDOM_STATE = 42
    ada = ClusterCentroids(random_state=RANDOM_STATE)
    X_res, y_res = ada.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res, y_res 

def viz_svm_2d(X_init,y, voc):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn import svm
    
    pca = PCA(n_components=2)
    X = pca.fit_transform(X_init)
    
    C = 1.0  # SVM regularization parameter
    h = 4  # step size in the mesh

    print ("linear")
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    print ("rbf")
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    print ("poly")
    bagging_rf = svm.LinearSVC(C=C).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 15, X[:, 0].max() + 15
    y_min, y_max = X[:, 1].min() - 15, X[:, 1].max() + 15
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    print ("ready to plot")
    for i, clf in enumerate((svc, bagging_rf, rbf_svc, poly_svc)):
        print ("doiing "+titles[i]+" graph")
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
        print ("doing "+titles[i]+"- DONE")
    plt.suptitle(voc)
    plt.show() 

def detrend(x):
    import numpy as np
    import scipy.signal as sps
    import matplotlib.pyplot as plt
    x = np.asarray(x)    
    jmps = np.where(np.diff(x) < -0.5)[0]  # find large, rapid drops in amplitdue
    for j in jmps:
        x[j+1:] += x[j] - x[j+1]    
    order = 20
    x -= sps.filtfilt([1] * order, [order], x)  # this is a very simple moving average filter
    return x
    
def patch_detrend(X_train):
    X_res = []
    for matr in X_train:
        matr_res = []
        for ch in matr:
            matr_res.append(detrend(ch))
        X_res.append(matr_res)
    return X_res

def data_to_file(X_train, y_train):
    X = X_train.reshape(X_train.shape[0], 121,8)
    X_to_write = []
    for matr in X:
        new_matr = []
        matr = list(zip(*matr)) 
        for item in matr:
            new_matr.append(";".join([str("%.2f" % i) for i in item]))
        X_to_write.append("|".join(new_matr))
        
    txt_outfile = open("data/imbalanced/data_train_over_2.txt", 'w')
#    txt_outfile = open("data/imbalanced/data_train_under.txt", 'w')
    for res in X_to_write: 
        txt_outfile.write(res+"\n")
    txt_outfile.close()
    
    txt_outfile = open("data/imbalanced/labels_train_over_2.txt", 'w')
#    txt_outfile = open("data/imbalanced/labels_train_under.txt", 'w')
    for res in y_train: 
        txt_outfile.write(res+"\n")
    txt_outfile.close()
    
def voc_to_file(X_train, y, voc):
    X = X_train.reshape(X_train.shape[0], 121,8)
    to_write = []
    for matr,y in zip(X,y):
        if y == "1":
            new_matr = []
            matr = list(zip(*matr)) 
            for item in matr:
                new_matr.append(";".join([str("%.2f" % i) for i in item]))
            to_write.append("|".join(new_matr))

    txt_outfile = open("data/pairs2/oversampling2/"+voc+".txt", 'w')
    #txt_outfile = open("data/pairs2/undersampling/"+voc+".txt", 'w')
    for res in to_write: 
        txt_outfile.write(res+"\n")
    txt_outfile.close()

def labels_to_file(y):
    txt_outfile = open("data/labels_train_over_2.txt", 'w')
 #   txt_outfile = open("data/labels_train_under.txt", 'w')
    for res in y: 
        txt_outfile.write(res+"\n")
    txt_outfile.close()    

def form_dataset(path):
    X = []
    y = []
    for dirname, dirnames, filenames in os.walk('data/pairs2/oversampling2'):
    #for dirname, dirnames, filenames in os.walk('data/pairs2/undersampling'):
        for filename in filenames:
           if "_labels" not in filename and "data" not in filename:
                voc = filename.split(".")[0]
                print (voc)
                X_part = load_data("data/pairs2/oversampling2/"+filename)
                #X_part = load_data("data/pairs2/undersampling/"+filename)
                y_part = [voc]*len(X_part)
                X.extend(X_part)
                y.extend(y_part)
                
    X = np.array(X)
    y = np.array(y)
    
    print (X.shape, y.shape)
    return X,y

def main():
    for dirname, dirnames, filenames in os.walk('data/pairs2/'):
        for filename in filenames:
           if "_labels" in filename:
                nm = filename.split("_")
                if len(nm) == 2:
                    voc = filename.split("_")[0]
                else:
                    voc = "_".join(filename.split("_")[:-1])
                print (voc)
                X_train = load_data("data/pairs2/data_all.txt")
                y = load_labels("data/pairs2/"+voc+"_labels.txt")
                print ("initial data: ", np.array(X_train).shape, np.array(y).shape)
                X_train = normalize_data(X_train)
                X_train = patch_detrend(X_train) 
                X_train = np.array(X_train)
                nsamples00, nx, ny = X_train.shape
                X_train = X_train.reshape((nsamples00,nx*ny))   
                print ("processed data: ", np.array(X_train).shape, np.array(y).shape)     
                #X_res, y_res = oversampling_data(X_train, y)
                X_res, y_res = undersampling_data(X_train, y)
                X_res = preprocessing.scale(X_res)
                #viz_svm_2d(X_res, y_res, voc)
                voc_to_file(X_res, y_res, voc)

    X_train, y_train = form_dataset("data/pairs2/")
    print ("classes: ", len(set(y_train)))
    data_to_file(X_train, y_train)
    
if __name__ == "__main__":
    main()
