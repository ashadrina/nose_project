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

def kmeans_clustering(data,labels,init_labels):		
	avg_data = get_avg(data)
	min_data = get_min(data)
	max_data = get_max(data)
	
	labels_pairs = label_matching(labels,init_labels)
	
	datasets = {'average': avg_data, 'min': min_data,  'max': max_data}
			  
	fignum = 1		  
	for name,X in datasets.items():
		X = np.array(X)
		kmeans = KMeans(n_clusters=len(set(labels)))
		kmeans.fit(X)

		fig = plt.figure(fignum, figsize=(4, 3))
		plt.clf()
		ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

		plt.cla()
		
		labels = kmeans.labels_

		ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float))

		ax.w_xaxis.set_ticklabels([])
		ax.w_yaxis.set_ticklabels([])
		ax.w_zaxis.set_ticklabels([])
		ax.set_xlabel('sensor1')
		ax.set_ylabel('sensor2')
		ax.set_zlabel('sensor3')
		fignum = fignum + 1
		
		# # Plot the ground truth
		# fig = plt.figure(fignum, figsize=(4, 3))
		# plt.clf()
		# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

		# plt.cla()

		# for name, label in labels_pairs:
			# ax.text3D(X[labels == label, 0].mean(),
					  # X[labels == label, 1].mean() + 1.5,
					  # X[labels == label, 2].mean(), name,
					  # horizontalalignment='center',
					  # bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
		# #Reorder the labels to have colors matching the cluster results
		# labels = np.choose(labels, [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]).astype(np.float)
		# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)

		ax.w_xaxis.set_ticklabels([])
		ax.w_yaxis.set_ticklabels([])
		ax.w_zaxis.set_ticklabels([])
		ax.set_xlabel('sensor1')
		ax.set_ylabel('sensor2')
		ax.set_zlabel('sensor3')
		plt.show()


def run_svd(data):
	new_data = []
	for matr in data:
		U, s, V = svd(np.array(matr), full_matrices=True)
		print (U.shape, s.shape, V.shape)
		s_s = []
		for ss in s:
			if ss < 10:
				s_s.append(0.0)
			else:
				s_s.append(ss)
		new_block = U*np.array(s_s)*V
	new_data.append(new_block)	
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
	#norm_data = normalize_data(data)
	
	#transform labels
	lb = LabelBinarizer()
	bin_labels = lb.fit_transform(lat_labels)
	int_labels = labels_to_int(bin_labels)
		
	#kmeans_clustering(data,int_labels,lat_labels)
	s = run_svd(data)
	print (s)
	#kmeans_clustering(data,int_labels,lat_labels)
	
if __name__ == "__main__":
	main()		