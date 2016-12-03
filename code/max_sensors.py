import os 
import sys
import numpy as np
from scipy import stats

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from transliterate import translit #perform transliteration from russian labels to latin

from numpy.linalg import svd #data reducion
from itertools import cycle


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

#####################################
	
def get_avg(data):
	mean_matrix = []
	for block in data:
		mean_block = []
		for col in block:
			mean_block.append(np.mean(col))
		mean_matrix.append(mean_block)
	return mean_matrix

def get_min(data):
	min_matrix = []
	for block in data:
		min_block = []
		for col in block:
			min_block.append(min(col))
		min_matrix.append(min_block)
	return min_matrix

def get_max(data):
	max_matrix = []
	for block in data:
		max_block = []
		for col in block:
			max_block.append(max(col))
		max_matrix.append(max_block)
	return max_matrix	

##########################

def data_to_file(OUT_FILE, lat_labels, maxlist):
    txt_outfile = open(OUT_FILE, 'w')	
    for lab, ll in zip(lat_labels, maxlist):
        rr = [lab, ":", " ".join([str(l) for l in ll]), "\n"]
        res = ' '.join(str(i) for i in rr)
        txt_outfile.write(res)
    txt_outfile.close()

###########################	
# python clustering.py 
#########################	##

def main():
    X_train = np.array(load_data("data_train.txt"))
    lat_labels = np.array(load_labels("rus/labels_train.txt"))
    print (len(set(lat_labels)))
    print ("initial data: ", np.array(X_train).shape)
   
    print ("Train: ", X_train.shape, np.array(lat_labels).shape)   
    trainmax = get_max(X_train)
    data_to_file("max/train.txt", lat_labels, trainmax)
    
    X_test = np.array(load_data("data_val.txt"))
    lat_labels_test = np.array(load_labels("rus/labels_val.txt"))
    print ("initial data: ", np.array(X_test).shape)

    print ("Test: ", X_test.shape, lat_labels_test.shape)	    
    testmax = get_max(X_test)
    data_to_file("max/test.txt", lat_labels_test, testmax)
    
    X_new = np.array(load_data("data_test.txt"))
    lat_labels_new = np.array(load_labels("test_names.txt"))
    print ("initial data: ", np.array(X_new).shape, np.array(lat_labels_new).shape)

    print ("New: ", X_new.shape)	
    newmax = get_max(X_new)
    data_to_file("max/new.txt", lat_labels_new, newmax)
	
if __name__ == "__main__":
	main()		