import os 
import sys
import numpy as np
from scipy import stats

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from transliterate import translit #perform transliteration from russian labels to latin

from numpy.linalg import svd #data reducion

import collections

from sklearn.svm import LinearSVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  ExtraTreesClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.lda import LDA

from sklearn.datasets import make_multilabel_classification as make_ml_clf

#visualization
import matplotlib.pyplot as plt
 
def load_data(in_file):
	input_f = open(in_file, "r")
	matrix = []
	for line in input_f:
		channels = [] #get channels
		for l in line.split("|"):
			samples = l.split(";")
			channels.append([float(i) for i in samples])
		matrix.append(channels)	
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
	
def labels_to_int(labels):
	new_labels = []
	for label in labels:
		new_labels.append(int((label.tolist()).index(1)+1))
	return new_labels	
	
def normalize_data(data):
    norm_matrix = []
    for block in data:
        current_max = np.amax(block)
        norm_col = []
        for col in block:
            norm_col.append([float(i)//current_max for i in col])
        norm_matrix.append(norm_col)
    return norm_matrix

def reduce_data(data):		
	new_data = []
	for matr in data:
		new_data.append(matr[:1])
	return new_data	

#########################################	
def run_svd(data):
	new_data = []
	for matr in data:
		U, s, V = svd(np.array(matr), full_matrices=True)
		s_s = [s[0]]
		s_s_1 = [0.0] * 7
		s_s.extend(s_s_1)
		S = np.zeros((8, 121), dtype=float)
		S[:8, :8] = np.diag(s_s)
		new_data.append(S)	
	return new_data	


def data_generation(X,y,thr):	
	X_new = []
	y_new = []
	c = collections.Counter(labels_to_int(y))
		
	for xx,yy, yyy in zip(X,y,labels_to_int(y)):
		nr = thr - int(c[yyy])
		for i in range(nr):
#			print (i, " -> ", i/10)
			X_new.append(xx)
			X_new.append((np.array(xx)+i/10).tolist())
			y_new.append(yy)
			y_new.append(yy)
		for i in range(nr):
			X_new.append(xx)
			X_new.append((np.array(xx)+i/10).tolist())
			y_new.append(yy)
			y_new.append(yy)	
	
	#print (np.array(X_new).shape)
	#print (np.array(y_new).shape)
	return np.array(X_new),np.array(y_new)
	
def label_generation(y,thr):	
	y_new = []
	c = collections.Counter(y)
	for yy in y:
		nr = thr - int(c[yy])
		for i in range(nr):
			y_new.append(yy)
			y_new.append(yy)
		for i in range(nr):
			y_new.append(yy)
			y_new.append(yy)	
	return np.array(y_new)
	
#########################################
def testing(X_train,y_train):
	print ("TRAIN, TEST - INITIAL + GENERATED DATA (+\-) \ 10")
	
	X_data = load_data("data_val.txt")
	lat_labels_test = load_labels("labels_val.txt")
	
	mlb1 = MultiLabelBinarizer()
	lat_labels_test.append(lat_labels)
	bin_labels_test = mlb1.fit_transform(lat_labels_test)
	bin_labels_test = bin_labels_test[:-1]

	print ("initial data: ", np.array(X_data).shape)
	val_ress = run_svd(X_data)
	new_val_ress = reduce_data(val_ress)
	print ("reduced data: ", np.array(new_val_ress).shape)
	
	X_test = []
	for dat in new_val_ress:
		X_test.extend(dat)
	X_test = np.array(X_test)

	print ("Learning...")
	run_training(X_train, y_train, X_test, bin_labels_test)
	
#########################################
def run_training(X_train, y_train, X_test, y_test):
	def svm_cl_training(X_train, y_train, X_test, y_test):
		svc = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
	#	svc = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
		err_train = np.mean(y_train != svc.predict(X_train))
		err_test  = np.mean(y_test  != svc.predict(X_test))
		print ("svm accuracy: ", 1 - err_train, 1 - err_test)
		
	def knn_cl_training(X_train, y_train, X_test, y_test):
		knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
		#knn = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
		err_train = np.mean(y_train != knn.predict(X_train))
		err_test  = np.mean(y_test  != knn.predict(X_test))
		print ("knn accuracy: ", 1 - err_train, 1 - err_test)

	def rf_cl_training(X_train, y_train, X_test, y_test):
		rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000)).fit(X_train, y_train)
		#rf = OneVsOneClassifier(RandomForestClassifier(n_estimators=1000)).fit(X_train, y_train)
		err_train = np.mean(y_train != rf.predict(X_train))
		err_test  = np.mean(y_test  != rf.predict(X_test))
		print ("rf accuracy: ", 1 - err_train, 1 - err_test)

	def bayes_cl_training(X_train, y_train, X_test, y_test):
		gnb = OneVsRestClassifier(GaussianNB()).fit(X_train, y_train)
		#gnb = OneVsOneClassifier(GaussianNB()).fit(X_train, y_train)
		err_train = np.mean(y_train != gnb.predict(X_train))
		err_test  = np.mean(y_test  != gnb.predict(X_test))
		print ("gnb accuracy: ", 1 - err_train, 1 - err_test)	
		#print ( gnb.predict(X_test))
		
	svm_cl_training(X_train, y_train, X_test, y_test)
	knn_cl_training(X_train, y_train, X_test, y_test)
	rf_cl_training(X_train, y_train, X_test, y_test)
	bayes_cl_training(X_train, y_train, X_test, y_test)
		
#########################################
def run_testing(X_train, y_train, X_test, mlb1, folder_name):
	def svm_cl_testing(X_train, y_train, X_test):
		svc = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
		err_train = np.mean(y_train != svc.predict(X_train))
		print ("svm train accuracy: ", 1 - err_train)
		y_new = svc.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/svm_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()
		
	def knn_cl_testing(X_train, y_train, X_test):
		knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
		err_train = np.mean(y_train != knn.predict(X_train))
		print ("knn train accuracy: ", 1 - err_train)		
		y_new = knn.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/knn_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()
		
	def rf_cl_testing(X_train, y_train, X_test):
		rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000)).fit(X_train, y_train)
		err_train = np.mean(y_train != rf.predict(X_train))
		print ("rf train accuracy: ", 1 - err_train)
		y_new = rf.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/rf_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()
		
		
	def bayes_cl_testing(X_train, y_train, X_test):
		gnb = OneVsRestClassifier(GaussianNB()).fit(X_train, y_train)
		err_train = np.mean(y_train != gnb.predict(X_train))
		print ("gnb train accuracy: ", 1 - err_train)
		y_new = gnb.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/gnb_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()
################				
	def ada_cl_testing(X_train, y_train, X_test):
		ada = OneVsRestClassifier(AdaBoostClassifier()).fit(X_train, y_train)
		err_train = np.mean(y_train != ada.predict(X_train))
		print ("ada train accuracy: ", 1 - err_train)
		y_new = ada.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/ada_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()
				
	def mlp_cl_testing(X_train, y_train, X_test):
		mlp = OneVsRestClassifier(MLPClassifier()).fit(X_train, y_train)
		err_train = np.mean(y_train != mlp.predict(X_train))
		print ("mlp train accuracy: ", 1 - err_train)
		y_new = mlp.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/mlp_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()
				
	def gpc_cl_testing(X_train, y_train, X_test):
		gpc = OneVsRestClassifier(GaussianProcessClassifier()).fit(X_train, y_train)
		err_train = np.mean(y_train != gpc.predict(X_train))
		print ("gpc train accuracy: ", 1 - err_train)
		y_new = gpc.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/gpc_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()	
		
	def dt_cl_testing(X_train, y_train, X_test):
		dt = OneVsRestClassifier(DecisionTreeClassifier()).fit(X_train, y_train)
		err_train = np.mean(y_train != dt.predict(X_train))
		print ("dt train accuracy: ", 1 - err_train)
		y_new = dt.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/dt_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()
		
	def et_cl_testing(X_train, y_train, X_test):
		et = OneVsRestClassifier(ExtraTreesClassifier()).fit(X_train, y_train)
		err_train = np.mean(y_train != et.predict(X_train))
		print ("et train accuracy: ", 1 - err_train)
		y_new = et.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/et_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()	
		
	def gb_cl_testing(X_train, y_train, X_test):
		gbt = OneVsRestClassifier(GradientBoostingClassifier()).fit(X_train, y_train)
		err_train = np.mean(y_train != gbt.predict(X_train))
		print ("gbt train accuracy: ", 1 - err_train)
		y_new = gbt.predict(X_test)
		y_labels = mlb1.inverse_transform(y_new)
		txt_outfile = open(folder_name+"/gbt_new_labels.txt", 'w')
		for y in y_labels:
			if y:
				txt_outfile.write(";".join(y)+"\n")
			else:
				txt_outfile.write("?\n")
		txt_outfile.close()	
		


	svm_cl_testing(X_train, y_train, X_test)
	knn_cl_testing(X_train, y_train, X_test)
	rf_cl_testing(X_train, y_train, X_test)
	bayes_cl_testing(X_train, y_train, X_test)
	ada_cl_testing(X_train, y_train, X_test)
	#mlp_cl_testing(X_train, y_train, X_test)
	#gpc_cl_testing(X_train, y_train, X_test)
	#dt_cl_testing(X_train, y_train, X_test)
	#et_cl_testing(X_train, y_train, X_test)
	#gb_cl_testing(X_train, y_train, X_test)

def run_with_generation(NN, foldername):
    data = load_data("data/data_train.txt")
    lat_labels = load_labels("data/labels_train.txt")
    print (len(set(lat_labels)))
    print ("initial data: ", np.array(data).shape)

    ##transform labels
    mlb = LabelBinarizer()
    bin_labels = mlb.fit_transform(lat_labels) 

    X_new, y_train = data_generation(data,bin_labels,NN)
    lat_labels_generated =  label_generation(lat_labels,NN)
    print ("generated data: ", X_new.shape, lat_labels_generated.shape)	
    X_new_2 = normalize_data(X_new)
    ress = run_svd(X_new_2)
    new_ress = reduce_data(ress)
    print ("reduced data: ", np.array(new_ress).shape)

    X_train = []
    for dat in new_ress:
        X_train.extend(dat)
    X_train = np.array(X_train)

    data_v = load_data("data/data_test.txt")
    lat_labels_init = load_labels("data/labels_test.txt")
    lat_labels_val = load_labels("data/labels_test.txt")
    print ("initial data: ", np.array(data_v).shape)
    data_v_2 = normalize_data(data_v)
    val_ress = run_svd(data_v_2)
    new_val_ress = reduce_data(val_ress)
    print ("reduced data: ", np.array(new_val_ress).shape)

    X_val= []
    for dat in new_val_ress:
        X_val.extend(dat)
    X_val = np.array(X_val)

    mlb1 = MultiLabelBinarizer()
    lat_labels_val.append(lat_labels)
    bin_labels_val = mlb1.fit_transform(lat_labels_val)
    y_val = bin_labels_val[:-1]

    print (X_train.shape, np.array(y_train).shape, X_val.shape, y_val.shape)	
    run_training(X_train, y_train, X_val, y_val)

    ###########################
    X_train_big = []
    X_train_big.extend(X_train)
    X_train_big.extend(X_val)
    X_train_big = np.array(X_train_big)

    ll = [] #y_train
    for l in lat_labels_generated:
        ll.append([l])

    y_train_lat = []
    y_train_lat.extend(ll)
    y_train_lat.extend(lat_labels_init)

    mlb1 = MultiLabelBinarizer()
    y_train_big =  mlb1.fit_transform(y_train_lat) 
    #	y_train_lat_2 =  mlb1.inverse_transform(y_train_big) 

    data_new = load_data("data/data_new.txt")
    print ("initial data: ", np.array(data_new).shape)
    data_new_2 = normalize_data(data_new)
    val_ress_new = run_svd(data_new_2)
    new_val_ress_new = reduce_data(val_ress_new)
    print ("reduced data: ", np.array(new_val_ress_new).shape)

    X_new = []
    for dat in new_val_ress_new:
        X_new.extend(dat)
    X_new = np.array(X_new)

    print (X_train_big.shape, np.array(y_train_big).shape, X_new.shape)	
    run_testing(X_train_big, y_train_big, X_new, mlb1, foldername)

def run_without_generation(foldername):
    X_train_data = load_data("data/data_train.txt")
    y_train_lat_labels = load_labels("data/labels_train.txt")
    print (len(set(y_train_lat_labels)))
    print ("initial data: ", np.array(X_train_data).shape)

    ##transform labels
    mlb = LabelBinarizer()
    y_train_bin_labels = mlb.fit_transform(y_train_lat_labels) 

    X_train_2 = normalize_data(X_train_data)
    X_t_ress = run_svd(X_train_2)
    X_train_ress = reduce_data(X_t_ress)
    print ("reduced data: ", np.array(X_train_ress).shape)

    X_train_0 = []
    for dat in X_train_ress:
        X_train_0.extend(dat)
    X_train_0 = np.array(X_train_0)

    X_test_data = load_data("data/data_test.txt")
    y_test_lat_labels = load_labels("data/labels_test.txt")
    y_test_lat_labels_2 = load_labels("data/labels_test.txt")
    print ("initial data: ", np.array(X_test_data).shape)
    
    X_test_2 = normalize_data(X_test_data)
    X_t_ress = run_svd(X_test_2)
    X_test_ress = reduce_data(X_t_ress)
    print ("reduced data: ", np.array(X_test_ress).shape)

    X_test_0 = []
    for dat in X_test_ress:
        X_test_0.extend(dat)
    X_test_0 = np.array(X_test_0)

    mlb1 = MultiLabelBinarizer()
    y_test_lat_labels.append(y_train_lat_labels)
    y_test_bin_labels = mlb1.fit_transform(y_test_lat_labels)
    y_test = y_test_bin_labels[:-1]

    print (X_train_0.shape, np.array(y_train_bin_labels).shape, X_test_0.shape, y_test.shape)	
    run_training(X_train_0, y_train_bin_labels, X_test_0, y_test)

    # ###########################
    X_train_big = []
    X_train_big.extend(X_train_0)
    X_train_big.extend(X_test_0)
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
    
    X_new_data_2 = normalize_data(X_new_data)
    X_n_ress = run_svd(X_new_data_2)
    X_new_ress = reduce_data(X_n_ress)
    print ("reduced data: ", np.array(X_new_ress).shape)

    X_new = []
    for dat in X_new_ress:
        X_new.extend(dat)
    X_new = np.array(X_new)

    print (X_train_big.shape, np.array(y_train_big).shape, X_new.shape)	
    run_testing(X_train_big, y_train_big, X_new, mlb2, foldername)
	
###########################	
# python clustering.py train_data2.txt
#########################	##

def main():
    out = "20012017"
    try:
        os.stat(out)
    except:
        os.mkdir(out) 
    run_with_generation(100, out)
    #run_without_generation(out)
    
if __name__ == "__main__":
	main()		
