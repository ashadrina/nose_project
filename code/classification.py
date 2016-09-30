import os 
import sys
import numpy as np
from scipy import stats

from transliterate import translit

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA
from numpy.linalg import svd
 
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
		labels.append(line.replace("\n",""))
	input_f.close()
	return labels
	
def transliterate_labels(labels):	
	lat_labels = []
	for label in labels:
		lat_labels.append(cyrillic2latin(label.replace("\n","")))
	return lat_labels

def cyrillic2latin(input):
	symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
           u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA")

	tr = {ord(a): ord(b) for a, b in zip(*symbols)}
	return input.translate(tr)		
	
def labels_to_int(labels):
	new_labels = []
	for l in labels:
		new_labels.append(int((l.tolist()).index(1)+1))
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

def run_svd(data):
	for matr in data:
		U, s, V = svd(np.array(matr), full_matrices=True)
		plt.plot(s)

	plt.title("Singular values for all data")	
	plt.savefig("sk_n_svd.png")
	
###########################	
# python clustering.py train_data2.txt
#########################	##

def main():
	if len (sys.argv) == 3:
		data_file = sys.argv[1]
		labels_file = sys.argv[2]

	print (data_file, labels_file)
	data = load_data(data_file)
	labels = load_labels(labels_file)

	lat_labels = transliterate_labels(labels)
	norm_data = normalize_data(data)
	
	#transform labels
	lb = LabelBinarizer()
	bin_labels = lb.fit_transform(lat_labels)
	int_labels = labels_to_int(bin_labels)

	ndata = normalize_data(data)
	normed_matrix = []
	for matr in data:
		normed_matrix.append(normalize(np.array(matr), axis=1, norm='l1'))
	run_svd(normed_matrix)

	
	
	
if __name__ == "__main__":
	main()		