# -*- coding: utf-8 -*-

import os 
import sys
import numpy as np
import pandas as pd
import scipy
#from scipy import stats
import copy 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.decomposition import PCA

#from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn import preprocessing
from numpy.linalg import svd
from sklearn.metrics import roc_curve
#from sklearn.cross_validation import train_test_split

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
    order = 15
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

##################################
def svm_mlb(X_train, y_train, X_new, mlb):
    from sklearn.svm import SVC, NuSVC
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

    print ("------------------------------------------")
    print ("--------------------SVM-------------------")
    print ("------------------------------------------")
    
    clf = OneVsRestClassifier( NuSVC(cache_size=200, class_weight='balanced', coef0=0.0, 
         decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
         max_iter=-1, nu=0.001, probability=True, random_state=None,   shrinking=True, tol=0.001, verbose=False) ) 
    
    clf.fit(X_train, y_train)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("svm train accuracy: ", 1 - err_train)

    #print (clf.get_params())
    #print (clf.decision_function(X_new))
    #print (clf.decision_function(X_train))
    
    compute_ind_metrics(y_train, y_score, X_new, clf, mlb, "svm")        
        
def bagging_rf_mlb(X_train, y_train, X_new, mlb):
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

    print ("------------------------------------------")
    print ("------------Bagging with RF---------------")
    print ("------------------------------------------")
    
    clf = OneVsRestClassifier(BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, 
                class_weight=None, criterion='gini',
                max_depth=None, max_features=10, max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=15, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False), max_samples=0.2, n_estimators=55))

    clf.fit(X_train, y_train)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("bagging train accuracy: ", 1 - err_train)
      
    compute_ind_metrics(y_train, y_score, X_new, clf, mlb, "bagging_rf")
  
def rf_mlb(X_train, y_train, X_new, mlb):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

    print ("------------------------------------------")
    print ("--------------------RF-------------------")
    print ("------------------------------------------")
    
    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=5000))
    clf.fit(X_train, y_train)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("random forest train accuracy: ", 1 - err_train)

    compute_ind_metrics(y_train, y_score, X_new, clf, mlb, "rf")
      
def mlp_mlb(X_train, y_train, X_new, mlb):
    from sklearn.neural_network import MLPClassifier

    print ("------------------------------------------")
    print ("--------------------MLP-------------------")
    print ("------------------------------------------")

    alphas = np.arange(97.4, 97.5, 0.1)
    names = []
    for i in alphas:
        names.append('alpha ' + str(i))

    classifiers = []
    for i in alphas:
        classifiers.append(MLPClassifier(alpha=i, random_state=1))
        
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        y_score =  clf.predict(X_train)
        print ("score: ", score)
        print ("alpha: ", name)

        compute_ind_metrics(y_train, y_score, X_new,  clf,  mlb, "mlp")
        break
##################################
def predicted_vs_real_label(y_pred, mlb, folder_name):
    y_true = load_labels("true_labels.txt")
    y_true = [[x.replace('dioktilftalat','dof') for x in l] for l in y_true]
    nums = range(1,76)
   # folder_name = "graphs/proba/real/"+folder_name
    folder_name = "graphs/proba/outliers/"+folder_name

    for x_test,y_tr,num in zip(y_pred,y_true, nums):
        y_tr = filter(None, y_tr)
        pred = dict([(c.replace('dioktilftalat','dof'),float("{:.3f}".format(yy))) for c,yy in zip(mlb.classes_,x_test)])      
        for pr in pred2[:10]:
            first10.append(pr[0])        
            
        for pr in pred2[-10:]:
            last10.append(pr[0])
        
        vocs = set(pred.keys())
        true = dict([(key, 0.0) for key in vocs])
        for tr in y_tr:
            true[tr] = 1.0

        df_pred = pd.DataFrame.from_dict(pred, orient='index')
        df_pred.columns=["Predicted"]
        df_true = pd.DataFrame.from_dict(true, orient='index')
        df_true.columns=["True"]
        df_result = pd.concat([df_true, df_pred], axis=1)
        df_result =  df_result.sort('Predicted', ascending=False)
              
        #print (num)
        #print (df_result)             
        #print ("--------------------------------")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        df_result["Predicted"].plot(kind='bar', ax=ax, color = 'teal', label="pred", alpha = 0.7, rot=90)
        df_result["True"].plot(kind='bar',ax=ax, color = 'red',  label="true",  alpha = 0.2)
        ax.set_xticklabels(df_result.index)
        plt.legend()
        plt.tight_layout()
        plt.title(num)
        # plt.show()

        try:
         os.stat(folder_name)
        except:
         os.mkdir(folder_name) 
        plt.savefig(folder_name+"/"+str(num)+".png", dpi=100)
        plt.close('all')

def compute_av_metrics(y_train, y_score, y_new_proba, mlb):
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
    y_test_true =  mlb.fit_transform(y_test_true_labels) 
    y_test_true = list(y_test_true)[:-1]      

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
  
def compute_ind_metrics(y_train, y_score, X_new, clf, mlb, algo):
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
    y_test_true =  mlb.fit_transform(y_test_true_labels) 
    y_test_true = list(y_test_true)[:-1]      
    
    from sklearn.metrics import coverage_error
    err1 = coverage_error(y_train, y_score)
    print ("You should predict top ",err1, " labels for train")
    from sklearn.metrics import label_ranking_average_precision_score
    rap1 = label_ranking_average_precision_score(y_train, y_score)
    print ("label_ranking_average_precision_score on train", rap1)

    true = {}
    pred = {}
    y_new = []
    for i in range(len(X_new)):
        print (i+1)
        y_new_proba = clf.predict_proba([X_new[i]])  
        y_new.append(y_new_proba)
        err2 = coverage_error([y_test_true[i]], y_new_proba)
        print ("Compounds in real solution: ", len(y_test_true_labels[i]))
        print ("You should predict top ",err2, " labels for toys")
        rap2 = label_ranking_average_precision_score([y_test_true[i]], y_new_proba)
        print ("label_ranking_average_precision_score on toys", rap2)
        print ("--------------------------------------------------------------")
        true[i] = float(len(y_test_true_labels[i]))
        pred[i] = err2
        
    df_pred = pd.DataFrame.from_dict(pred, orient='index')
    df_pred.columns=["Predicted"]
    df_true = pd.DataFrame.from_dict(true, orient='index')
    df_true.columns=["True"]
    df_result = pd.concat([df_true, df_pred], axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df_result["Predicted"].plot(kind='line', ax=ax, color = 'teal', label="pred", alpha = 0.7, rot=90, lw=3)
    df_result["True"].plot(kind='line',ax=ax, color = 'red',  label="true",  alpha = 0.7, lw=3, ls='dotted')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.title("Coverage error VS true labell number")
    #plt.show()
    #plt.savefig("graphs/metrics/outliers/"+algo+".png", dpi=100)
    #plt.savefig("graphs/metrics/real/"+algo+".png", dpi=100)
    
   # predicted_vs_real_label(y_new, mlb, algo)
   # compute_area_between_curves(df_result)
    df_result.to_csv('pandas.csv', header=None, index=None, sep=';', mode='w')

    
def compute_area_between_curves(df_result):
    y1 = df_result["Predicted"].values
    y2 = df_result["True"].values
    x = df_result.index
    z = y1-y2
    dx = x[1:] - x[:-1]
    cross_test = np.sign(z[:-1] * z[1:])
    x_intersect = x[:-1] - dx / (z[1:] - z[:-1]) * z[:-1]
    dx_intersect = - dx / (z[1:] - z[:-1]) * z[:-1]
    areas_pos = abs(z[:-1] + z[1:]) * 0.5 * dx # signs of both z are same
    areas_neg = 0.5 * dx_intersect * abs(z[:-1]) + 0.5 * (dx - dx_intersect) * abs(z[1:])
    areas = np.where(cross_test < 0, areas_neg, areas_pos)
    total_area = np.sum(areas)
    print ("Area between curves: ", total_area)
     

def k_means_new(data):
    from time import time
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import metrics
    from sklearn.cluster import KMeans
    
    labels = load_labels("data/labels_new.txt")
    
    n_samples, n_features = data.shape
    
    print("n_samples %d, \t n_features %d"
      % ( n_samples, n_features))


    print(79 * '_')
    print('% 9s' % 'init'
          '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')


    def bench_k_means(estimator, name, data):
        t0 = time()
        estimator.fit(data)
        print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(labels, estimator.labels_),
                 metrics.completeness_score(labels, estimator.labels_),
                 metrics.v_measure_score(labels, estimator.labels_),
                 metrics.adjusted_rand_score(labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean')))#,
                                         # sample_size=sample_size)))

    bench_k_means(KMeans(init='k-means++', n_clusters=8, n_init=10), name="k-means++", data=data)
    bench_k_means(KMeans(init='random', n_clusters=8, n_init=10),  name="random", data=data)
    
    kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10).fit(data)
    for i, c in zip(range(1,76), kmeans.labels_):
       print (i, " - ", c)
   
   # # Step size of the mesh. Decrease to increase the quality of the VQ.
    # h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    # from sklearn.decomposition import KernelPCA, PCA
    # kpca = KernelPCA(n_components=2, kernel="cosine", fit_inverse_transform=True)
    # reduced_data = kpca.fit_transform(data)
    # kmeans = KMeans(init='k-means++', n_clusters=6, n_init=10).fit(reduced_data)
    
    # #reduced_data = np.array(data)
    # # Plot the decision boundary. For that, we will assign a color to each
    # x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    # y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # # Obtain labels for each point in mesh. Use last trained model.
    # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(Z, interpolation='nearest',
               # extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               # cmap=plt.cm.Paired,
               # aspect='auto', origin='lower')

    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # # Plot the centroids as a white X
    # centroids = kmeans.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1],
                # marker='x', s=169, linewidths=3,
                # color='w', zorder=10)
    # plt.title('K-means clustering on the digits dataset\n'
              # 'Centroids are marked with white cross')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
        
     
def main():
    X_train, y_train, X_new, mlb = load_dataset()

    X_train = load_data("data/data_all_outliers.txt")
    y_train_lat = load_labels("data/labels_all_outliers.txt")

    y_train_lat_list = []
    for i in y_train_lat:
        y_train_lat_list.append([i])

    y_train =  mlb.fit_transform(y_train_lat_list) 
 
    X_train = normalize_data(X_train)    
    X_train = patch_detrend(X_train)    
       
    X_train = np.array(X_train)
    nsamples00, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples00,nx*ny))        
    
   # from sklearn.decomposition import KernelPCA, PCA
    #kpca = KernelPCA(kernel="cosine", fit_inverse_transform=True)
    
    #from sklearn.manifold import TSNE
    #tsne = TSNE(random_state=0)
   # X_train = tsne.fit_transform(X_train)
    #KernelPCA(n_components=None, kernel='linear', gamma=None, degree=3, coef0=1, 
    #kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
#    tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1)

  #  X_train = kpca.fit_transform(X_train)
  #  X_train = kpca.inverse_transform(X_train)
    
    ###################################  
    X_new = load_data("data/data_new.txt")
    X_new = normalize_data(X_new)    
    X_new = patch_detrend(X_new)    
  
    X_new = np.array(X_new)
    nsamples22, nx, ny = X_new.shape
    X_new = X_new.reshape((nsamples22,nx*ny))      

   # X_new = kpca.transform(X_new)    
 #  X_new = kpca.inverse_transform(X_new)    
 #   X_new = tsne.fit_transform(X_new)    
    ###################################
    X_train = preprocessing.scale(X_train)
    X_new = preprocessing.scale(X_new)

    print (np.array(X_train).shape)
    print (np.array(X_new).shape)
    
    #svm_mlb(X_train, y_train, X_new, mlb) 
    #print ("------------------------------------------")
    #bagging_rf_mlb(X_train, y_train, X_new, mlb) 
    #print ("------------------------------------------")
    #rf_mlb(X_train, y_train, X_new, mlb)
    #print ("------------------------------------------")
    #mlp_mlb(X_train, y_train, X_new, mlb) 
    #print ("------------------------------------------")
    
    known = pd.read_csv("pandas.csv",index_col=None, header=0, sep=';')
    known = known.drop('Pred', 1)
    #print (set(known["True"].values))
    print ("Metrics data")
    k_means_new(known)
    #known.plot(kind='scatter', x='True', y='Pred')
    #plt.show()
    
    print ("New data")
    X_new = np.array(load_data("data/data_new.txt"))
    X_new = np.array(X_new)
    nsamples22, nx, ny = X_new.shape
    X_new = X_new.reshape((nsamples22,nx*ny))      
#    k_means_new(X_new)
    
    from sklearn.decomposition import KernelPCA, PCA
    kpca = KernelPCA(n_components=1, kernel="cosine", fit_inverse_transform=True)
    reduced_data = kpca.fit_transform(X_new)
    k_means_new(reduced_data)
    
    # p = pd.DataFrame(data=reduced_data, index=None, columns=["One", "Two"])
    # p.plot(kind='scatter', x='One', y='Two')
    # plt.show()
    

if __name__ == "__main__":
    main()
