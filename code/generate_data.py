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

from sklearn.lda import LDA

from sklearn.datasets import make_multilabel_classification as make_ml_clf

 
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

def generate_data(X,y,thr):	
	X_new = []
	y_new = []
	c = collections.Counter(labels_to_int(y))

	for xx,yy, yyy in zip(X,y,labels_to_int(y)):
		nr = thr - int(c[yyy])
		for i in range(nr):
			X_new.append([x+i for x in xx])
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
	c = collections.Counter(labels_to_int(y))
		
	for xx,yy, yyy in zip(X,y,labels_to_int(y)):
		nr = thr - int(c[yyy])
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
	c = collections.Counter(labels_to_int(y))
		
	for xx,yy, yyy in zip(X,y,labels_to_int(y)):
		nr = thr - int(c[yyy])
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
	# print ("INITIAL DATA")
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
	# print ("Learning...")
	# run_training(X_train, y_train, X_test, y_test)	
	
	print ("=======================")
	print ("TRAIN - INITIAL DATA, TEST - GENERATED DATA")
	X_new,y_new = generate_data(X,y,10)	
	print ("Learning...")
	run_training(X, y, X_new, y_new)
		
	print ("=======================")
	print ("TRAIN, TEST - INITIAL + GENERATED DATA (+\-) \ 10")
	X_new_2,y_new_2 = generate_data_2(X,y,10)	
	X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_new_2, y_new_2, test_size = 0.3, random_state = 1)
	print ("Learning...")
	run_training(X_train_2, y_train_2, X_test_2, y_test_2)
	
	print ("=======================")
	print ("TRAIN, TEST - INITIAL + GENERATED DATA (+\-) \ 20")
	X_new_20,y_new_20 = generate_data_2(X,y,20)	
	X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(X_new_20, y_new_20, test_size = 0.3, random_state = 1)
	print ("Learning...")
	run_training(X_train_20, y_train_20, X_test_20, y_test_20)

	print ("=======================")
	print ("TRAIN, TEST - INITIAL + GENERATED DATA (+\-\*) \ 10")
	X_new_3,y_new_3 = generate_data_3(X,y,10)	
	X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_new_3, y_new_3, test_size = 0.3, random_state = 1)
	print ("Learning...")
	run_training(X_train_3, y_train_3, X_test_3, y_test_3)
	
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
def run_testing(X_train, y_train, X_test):
	def svm_cl_testing(X_train, y_train, X_test):
		svc = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
		err_train = np.mean(y_train != svc.predict(X_train))
		y_new = svc.predict(X_test)
		txt_outfile = open("svm_new_labels.txt", 'w')
		for y in y_new:
			txt_outfile.write(y+"\n")
		txt_outfile.close()
		
	def knn_cl_testing(X_train, y_train, X_test):
		knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
		err_train = np.mean(y_train != knn.predict(X_train))
		y_new = knn.predict(X_test)
		txt_outfile = open("knn_new_labels.txt", 'w')
		for y in y_new:
			txt_outfile.write(y+"\n")
		txt_outfile.close()
		
	def rf_cl_testing(X_train, y_train, X_test):
		rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000)).fit(X_train, y_train)
		err_train = np.mean(y_train != rf.predict(X_train))
		y_new = rf.predict(X_test)
		txt_outfile = open("rf_new_labels.txt", 'w')
		for y in y_new:
			txt_outfile.write(y+"\n")
		txt_outfile.close()
		
	def bayes_cl_testing(X_train, y_train, X_test):
		gnb = OneVsRestClassifier(GaussianNB()).fit(X_train, y_train)
		err_train = np.mean(y_train != gnb.predict(X_train))
		y_new = gnb.predict(X_test)
		txt_outfile = open("gnb_new_labels.txt", 'w')
		for y in y_new:
			txt_outfile.write(y+"\n")
		txt_outfile.close()
	
	svm_cl_testing(X_train, y_train, X_test)
	knn_cl_testing(X_train, y_train, X_test)
	rf_cl_testing(X_train, y_train, X_test)
	bayes_cl_testing(X_train, y_train, X_test)
	
	
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
	print (len(set(lat_labels)))
		
	##transform labels

	mlb = LabelBinarizer()
	bin_labels = mlb.fit_transform(lat_labels) 
	int_labels = labels_to_int(bin_labels)
	
	print ("initial data: ", np.array(data).shape)
	ress = run_svd(data)
	new_ress = reduce_data(ress)
	print ("reduced data: ", np.array(new_ress).shape)
	
	X = []
	for dat in new_ress:
		X.extend(dat)
	X = np.array(X)
	
	#training(X,bin_labels)
	
	print ("=======================")
	print ("TRAIN, TEST - INITIAL + GENERATED DATA (+\-) \ 10")
	X_train,y_train = generate_data(X,bin_labels,10)	
	
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

if __name__ == "__main__":
	main()		
