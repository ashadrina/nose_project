import os 
import sys
import numpy as np
from scipy import stats

from transliterate import translit

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.linalg import svd
 
from pylab import *
from pybrain.structure.modules import KohonenMap
from mvpa2.suite import *
from scipy import random
import time 

 
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

#########################################	
	
def kmeans_clustering_2d(data,labels,init_labels):		
	avg_data = get_avg(data)
	min_data = get_min(data)
	max_data = get_max(data)
	
	labels_pairs = label_matching(labels,init_labels)
	
	#datasets = {'average': avg_data, 'min': min_data,  'max': max_data}
	datasets = {'average': avg_data,   'max': max_data}
			  
	fignum = 1		  
	for name,X in datasets.items():
		X = np.array(X)
		kmeans = KMeans(n_clusters=18)
		kmeans.fit(X)

		fig = plt.figure(fignum, figsize=(4, 3))
		plt.clf()
		ax = fig.add_subplot(111)

		plt.cla()		
		k_labels = kmeans.labels_
		ax.scatter(X[:, 0], X[:, 1], c=k_labels.astype(np.float))
		plt.legend()
		fignum = fignum + 1
		
		# # Plot the ground truth
		fig = plt.figure(fignum, figsize=(4, 3))
		plt.clf()
		ax = fig.add_subplot(111)
		plt.cla()
	
		ax.scatter(X[:, 0], X[:, 1], c=labels)

		for label, x, y in zip(init_labels, X[:, 0], X[:, 1]):
			plt.annotate(
				label, 
				xy = (x, y), xytext = (-20, 20),
				textcoords = 'offset points', ha = 'right', va = 'bottom',
				bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5),
				arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
		
		plt.show()
		

def kohonen_2(data, names):
#used pymvpa
	print ("kohonen:")
	#names = df['fragment_ind'].tolist()
	#names = ["mm", "MM", "mM", "Mm", "mM", "mm", "MM", "mM", "mm", "Mm", "MM", "mm", "Mm", "mm", "mM", "mm", "mm", "mm", "MM", "mm", "MM", "mM", "Mm", "mm", "Mm", "mm", "mM", "mm", "mm", "mm", "MM", "mm", "MM", "mM", "Mm", "mM", "mm", "MM", "mM", "mm", "Mm", "MM", "Mm", "mm", "Mm", "mm", "mM", "mm"]
	
	#del df['fragment_ind']
#	data = array(df.values)
	print ("Clustering elements: ",len(names))
	print ("Clustering dataset: "+str(len(data))+" elements")

	#data = array( [[0., 0., 0.],  [0., 0., 1.],  [0., 1., 0.],  [1., 0., 0.],  [1., 0., 1.],  [1., 1., 0.],  [1., 1., 1.],  [.66, .66, .66]])
	#names = 	['black',	'blue','green',	 'red', 'violet',	 'yellow', 'white', 'lightgrey'] 	# store the names of the data for visualization later on

	som = SimpleSOMMapper((10, 20), 1000, learning_rate=0.05)
	#reduced_data = PCA(n_components=3).fit_transform(data)

	som.train(data)
	#print (som.K.shape)	
	mapped = som(data)
	#plt.title('Color SOM')
	# SOM's kshape is (rows x columns), while matplotlib wants (X x Y)
	#plt.matshow(mapped)
	xyz = list(zip(*mapped))
	min_list = list(map(min, xyz))
	max_list = list(map(max, xyz))
	
	plt.xlim(min_list[1]-5, max_list[1]+5)
	plt.ylim(min_list[0]-5, max_list[0]+5)
	for i, m in enumerate(mapped):
		#print (m[1], m[0], names[i])
		if (names[i] == 'toluol'):
			r, = plt.plot(m[1], m[0], 'ro', color="red")
		elif (names[i] == 'fenol'):	
			g, = plt.plot(m[1], m[0], 'ro', color="green")
		elif (names[i] == 'etilazetat'):	
			b, = plt.plot(m[1], m[0], 'ro', color="blue")
		else:
			y, = plt.plot(m[1], m[0], 'ro', color="yellow")

		#plt.text(m[1], m[0], names[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
	plt.legend([r, g, b, y], ["toluol", "fenol", "etilazetat", "other"])
	plt.show()
	
def run_svd(data):
	new_data = []
	for matr in data:
		U, s, V = svd(np.array(matr), full_matrices=True)
		s_s = [s[0], s[1]]
		s_s_1 = [0.0] * 6
		s_s.extend(s_s_1)
		#for ss in s:
		#	if ss < 10:
		#		s_s.append(0.0)
		#	else:
		#		s_s.append(ss)
		S = np.zeros((8, 121), dtype=float)	
		S[:8, :8] = np.diag(s_s)
		new_data.append(S)	
	return new_data	
	
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
	print (lat_labels)
	#norm_data = normalize_data(data)
	
	#transform labels
	lb = LabelBinarizer()
	bin_labels = lb.fit_transform(lat_labels)
	int_labels = labels_to_int(bin_labels)


	ress = run_svd(data)
	kohonen_2(np.array(data), lat_labels)
	#kmeans_clustering_2d(ress,int_labels,lat_labels)
	
if __name__ == "__main__":
	main()		