import os 
import sys
import numpy as np
from scipy import stats

from sklearn.preprocessing import LabelBinarizer
from transliterate import translit #perform transliteration from russian labels to latin

from numpy.linalg import svd #data reducion

import collections

from sklearn.svm import LinearSVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.lda import LDA
 
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
		labels.append(line.replace("\n",""))
	input_f.close()
	return labels
	
def labels_to_int(labels):
	new_labels = []
	for label in labels:
		if ";" in label:
			new_labels.append([int((label[0].tolist()).index(1)+1), int((label[0].tolist()).index(1)+1)])
		else:
			new_labels.append(int((label.tolist()).index(1)+1))
	return new_labels	

def label_matching(bin_labels,labels):
	pairs = []
	for y,l in zip(bin_labels,labels):
		pairs.append((l,y))
	return pairs	
	
def normalize_data(data):
	norm_matrix = []
	for block in data:
		norm_col = []
		for col in block:
			norm_col.append([float(i)/sum(col) for i in col])
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

def generate_data(X,y):	
	X_new = []
	y_new = []
	c = collections.Counter(y)
	
	for xx,yy in zip(X,y):
		nr = 10 - int(c[yy])
		for i in range(nr):
			y_new.append(yy)
		for i in range(nr):
			X_new.append([x-i for x in xx])
			y_new.append(yy)
			
	print (np.array(X_new).shape)
	print (np.array(y_new).shape)
	return np.array(X_new),np.array(y_new)
	
def generate_data_2(X,y,thr):	
	X_new = []
	y_new = []
	c = collections.Counter(y)
		
	for xx,yy in zip(X,y):
		nr = thr - int(c[yy])
		for i in range(nr):
			X_new.append(xx)
			X_new.append([x+i for x in xx])
			y_new.append(yy)
			y_new.append(yy)
		for i in range(nr):
			X_new.append(xx)
			X_new.append([x-i for x in xx])
			y_new.append(yy)
			y_new.append(yy)
	
	print (np.array(X_new).shape)
	print (np.array(y_new).shape)
	return np.array(X_new),np.array(y_new)
	
def generate_data_3(X,y,thr):	
	X_new = []
	y_new = []
	c = collections.Counter(y)
		
	for xx,yy in zip(X,y):
		nr = thr - int(c[yy])
		for i in range(nr):
			X_new.append(xx)
			X_new.append([x+i for x in xx])
			y_new.append(yy)
			y_new.append(yy)
		for i in range(nr):
			X_new.append(xx)
			X_new.append([x-i for x in xx])
			y_new.append(yy)
			y_new.append(yy)	
		for i in np.linspace(0.0,1.0,num=nr):
			X_new.append(xx)
			X_new.append([x*i for x in xx])
			y_new.append(yy)
			y_new.append(yy)	

	
	print (np.array(X_new).shape)
	print (np.array(y_new).shape)
	return np.array(X_new),np.array(y_new)
		
#########################################
def training(X,y):
	print ("INITIAL DATA")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
	print ("Learning...")
	svm_cl(X_train, y_train, X_test, y_test)
	knn_cl(X_train, y_train, X_test, y_test)
	rf_cl(X_train, y_train, X_test, y_test)
	bayes_cl(X_train, y_train, X_test, y_test)	
	
	print ("=======================")
	print ("TRAIN - INITIAL DATA, TEST - GENERATED DATA")
	X_new,y_new = generate_data(X,y)	
	print ("Learning...")
	svm_cl(X, y, X_new, y_new)
	knn_cl(X, y, X_new, y_new)
	rf_cl(X, y, X_new, y_new)
	bayes_cl(X, y, X_new, y_new)
		
	print ("=======================")
	print ("TRAIN, TEST - INITIAL + GENERATED DATA (+\-) \ 10")
	X_new_2,y_new_2 = generate_data_2(X,y,10)	
	X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_new_2, y_new_2, test_size = 0.3, random_state = 1)
	print ("Learning...")
	svm_cl(X_train_2, y_train_2, X_test_2, y_test_2)
	knn_cl(X_train_2, y_train_2, X_test_2, y_test_2)
	rf_cl(X_train_2, y_train_2, X_test_2, y_test_2)
	bayes_cl(X_train_2, y_train_2, X_test_2, y_test_2)			
	
	print ("=======================")
	print ("TRAIN, TEST - INITIAL + GENERATED DATA (+\-) \ 20")
	X_new_2,y_new_2 = generate_data_2(X,y,20)	
	X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_new_2, y_new_2, test_size = 0.3, random_state = 1)
	print ("Learning...")
	svm_cl(X_train_2, y_train_2, X_test_2, y_test_2)
	knn_cl(X_train_2, y_train_2, X_test_2, y_test_2)
	rf_cl(X_train_2, y_train_2, X_test_2, y_test_2)
	bayes_cl(X_train_2, y_train_2, X_test_2, y_test_2)		
	
	print ("=======================")
	print ("TRAIN, TEST - INITIAL + GENERATED DATA (+\-\*) \ 10")
	X_new_3,y_new_3 = generate_data_3(X,y,10)	
	X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_new_3, y_new_3, test_size = 0.3, random_state = 1)
	print ("Learning...")
	svm_cl(X_train_3, y_train_3, X_test_3, y_test_3)
	knn_cl(X_train_3, y_train_3, X_test_3, y_test_3)
	rf_cl(X_train_3, y_train_3, X_test_3, y_test_3)
	bayes_cl(X_train_3, y_train_3, X_test_3, y_test_3)	
#########################################
	
def svm_cl(X_train, y_train, X_test, y_test):
	svc = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
	err_train = np.mean(y_train != svc.predict(X_train))
	err_test  = np.mean(y_test  != svc.predict(X_test))
	print ("svm accuracy: ", 1 - err_train, 1 - err_test)
	
def knn_cl(X_train, y_train, X_test, y_test):
	knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
	err_train = np.mean(y_train != knn.predict(X_train))
	err_test  = np.mean(y_test  != knn.predict(X_test))
	print ("knn accuracy: ", 1 - err_train, 1 - err_test)

def rf_cl(X_train, y_train, X_test, y_test):
	rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000)).fit(X_train, y_train)
	err_train = np.mean(y_train != rf.predict(X_train))
	err_test  = np.mean(y_test  != rf.predict(X_test))
	print ("rf accuracy: ", 1 - err_train, 1 - err_test)

def bayes_cl(X_train, y_train, X_test, y_test):
	gnb = OneVsRestClassifier(GaussianNB()).fit(X_train, y_train)
	err_train = np.mean(y_train != gnb.predict(X_train))
	err_test  = np.mean(y_test  != gnb.predict(X_test))
	print ("nb accuracy: ", 1 - err_train, 1 - err_test)
	
###########################	
# python clustering.py train_data2.txt
#########################	##

def main():
	if len (sys.argv) == 3:
		data_file = sys.argv[1]
		labels_file = sys.argv[2]

	print (data_file, labels_file)
	data = load_data(data_file)
	lat_labels = load_labels(labels_file)

	#norm_data = normalize_data(data)
	
	#transform labels
	lb = LabelBinarizer()
	bin_labels = lb.fit_transform(lat_labels)
	int_labels = labels_to_int(bin_labels)

	print ("initial data: ", np.array(data).shape)
	ress = run_svd(data)
	new_ress = reduce_data(ress)
	print ("reduced data: ", np.array(new_ress).shape)
	
	X = []
	for dat in new_ress:
		X.extend(dat)
	X = np.array(X)
	y = int_labels
	
	#training(X,y)
	
	print ("=======================")
	print ("TRAIN, TEST - INITIAL + GENERATED DATA (+\-) \ 10")
	#X_train,y_train = generate_data(X,y,10)	
	X_data = load_data("data_val.txt")
	lat_labels_test = load_labels("labels_val.txt")
	
	print ("initial data: ", np.array(X_test).shape)
	val_ress = run_svd(X_test)
	new_val_ress = reduce_data(val_ress)
	print ("reduced data: ", np.array(new_val_ress).shape)
	
	X_test = []
	for dat in new_val_ress:
		X_test.extend(dat)
	X_test = np.array(X)
	y_test = int_labels
	print (np.array(X_test).shape)
	print (y_test)

	# print ("Learning...")
	# svm_cl(X_train, y_train, X_test, y_test)
	# knn_cl(X_train, y_train, X_test, y_test)
	# rf_cl(X_train, y_train, X_test, y_test)
	# bayes_cl(X_train, y_train, X_test, y_test)			
	
if __name__ == "__main__":
	main()		
