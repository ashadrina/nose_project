import os 
import sys
import numpy as np
from scipy import stats

from sklearn.preprocessing import LabelBinarizer
from transliterate import translit #perform transliteration from russian labels to latin

from numpy.linalg import svd #data reducion

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
	
	
#########################################
	
def svm_cl(X_train, y_train, X_test, y_test):
	svc = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
	err_train = np.mean(y_train != svc.predict(X_train))
	err_test  = np.mean(y_test  != svc.predict(X_test))
	print ("svm error: ", err_train, err_test)
	
def knn_cl(X_train, y_train, X_test, y_test):
	knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
	err_train = np.mean(y_train != knn.predict(X_train))
	err_test  = np.mean(y_test  != knn.predict(X_test))
	print ("knn error: ", err_train, err_test)

def rf_cl(X_train, y_train, X_test, y_test):
	rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100)).fit(X_train, y_train)
	err_train = np.mean(y_train != rf.predict(X_train))
	err_test  = np.mean(y_test  != rf.predict(X_test))
	print ("rf error: ", err_train, err_test)

def bayes_cl(X_train, y_train, X_test, y_test):
	gnb = OneVsRestClassifier(GaussianNB()).fit(X_train, y_train)
	err_train = np.mean(y_train != gnb.predict(X_train))
	err_test  = np.mean(y_test  != gnb.predict(X_test))
	print ("nb error: ", err_train, err_test)
	
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
	
	clf = LDA()
	clf.fit(X, y)
		
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
	print ("Learning...")
	svm_cl(X_train, y_train, X_test, y_test)
	# knn_cl(X_train, y_train, X_test, y_test)
	# rf_cl(X_train, y_train, X_test, y_test)
	# bayes_cl(X_train, y_train, X_test, y_test)
	
	
	
if __name__ == "__main__":
	main()		
