import os 
import sys
import numpy as np
from scipy import stats

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from transliterate import translit #perform transliteration from russian labels to latin

from numpy.linalg import svd #data reducion

from sklearn.cluster import KMeans #n clusters NOT required

from sklearn.cluster import DBSCAN #n clusters NOT required
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.cluster import MeanShift, estimate_bandwidth #n clusters NOT required

from sklearn.cluster import AffinityPropagation #n clusters NOT required
from itertools import cycle

#visualization
import matplotlib.pyplot as plt
import matplotlib.cm  as cmx
from mpl_toolkits.mplot3d import Axes3D


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
	
def labels_to_int(lat_labels):
	int_to_lat_labels = {}
	for k,v in zip(range(len(lat_labels)),lat_labels):
		int_to_lat_labels[k] = v
	#print (int_to_lat_labels)
	
	lat_to_int_labels = {}
	for k in lat_labels:
		k = ';'.join(k)
		lat_to_int_labels[k] = ""
		
	for k,v in zip(lat_to_int_labels.keys(),range(len(lat_to_int_labels.keys()))):
		lat_to_int_labels[k] = v

	#print (lat_to_int_labels)
	return int_to_lat_labels,lat_to_int_labels
	
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
		s_s = [s[0],s[1]]
		print (s_s)
		s_s_1 = [0.0] * 6
		s_s.extend(s_s_1)
		S = np.zeros((8, 121), dtype=float)	
		S[:8, :8] = np.diag(s_s)
		new_data.append(S)	
	return new_data	
	

def aff_prop(X,lat_labels):	
	int_to_lat_labels,lat_to_int_labels = labels_to_int(lat_labels)
	int_labels = []
	
	for ll in lat_labels:
		ll = ";".join(list(ll))
		int_labels.append(lat_to_int_labels[ll])

	af = AffinityPropagation(preference=-50).fit(X)
	cluster_centers_indices = af.cluster_centers_indices_
	labels = af.labels_
			
	n_clusters_ = len(cluster_centers_indices)

	print("* Number of classes: %d" %len(set(int_labels)))
	print('* Estimated number of clusters: %d' % n_clusters_)
	print("* Homogeneity: %0.3f" % metrics.homogeneity_score(int_labels, labels))
	print("* Completeness: %0.3f" % metrics.completeness_score(int_labels, labels))
	print("* V-measure: %0.3f" % metrics.v_measure_score(int_labels, labels))
	print("* Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(int_labels, labels))
	print("* Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(int_labels, labels))
	print("* Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

	plt.close('all')
	plt.figure(1)
	plt.clf()
	
	colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	for k, col in zip(range(n_clusters_), colors):
		class_members = labels == k
		cluster_center = X[cluster_centers_indices[k]]
		plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
		plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=14)
		for x in X[class_members]:
			plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

	plt.title('Estimated number of affinity propagation clusters: %d' % n_clusters_)
	plt.show()	  
#	plt.savefig("aff_prop_2.png")

def aff_prop_nd(XX,lat_labels):	
	int_to_lat_labels,lat_to_int_labels = labels_to_int(lat_labels)
	int_labels = []
	for ll in lat_labels:
		ll = ";".join(list(ll))
		int_labels.append(lat_to_int_labels[ll])

	int_labels = np.array(int_labels)
	print (int_labels)
	# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	
	x_coordinates = []
	y_coordinates = []
	for x in XX:
		x_coordinates.append(x[0])
		y_coordinates.append(x[1])
				
	X = np.array(x_coordinates)
	y = int_labels
	af = AffinityPropagation(preference=-50).fit(X)
	cluster_centers_indices = af.cluster_centers_indices_
	n_clusters_ = len(cluster_centers_indices)
	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	plt.cla()

	labels = af.labels_
	
	ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('Petal width')
	ax.set_ylabel('Sepal length')
	ax.set_zlabel('Petal length')

	# Plot the ground truth
	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	plt.cla()

	for name, label in [('Setosa', 0),
						('Versicolour', 1),
						('Virginica', 2)]:
		ax.text3D(X[y == label, 3].mean(),
				  X[y == label, 0].mean() + 1.5,
				  X[y == label, 2].mean(), name,
				  horizontalalignment='center',
				  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
	# Reorder the labels to have colors matching the cluster results
	y = np.choose(y, [1, 2, 0]).astype(np.float)
	ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('Petal width')
	ax.set_ylabel('Sepal length')
	ax.set_zlabel('Petal length')
	plt.show()	


def dbscan(X,int_labels):	
	X = StandardScaler().fit_transform(X)
	
	db = DBSCAN(eps=0.3, min_samples=10).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print("* Number of classes: %d" %len(set(int_labels)))
	print('* Estimated number of clusters: %d' % n_clusters_)
	print("* Homogeneity: %0.3f" % metrics.homogeneity_score(int_labels, labels))
	print("* Completeness: %0.3f" % metrics.completeness_score(int_labels, labels))
	print("* V-measure: %0.3f" % metrics.v_measure_score(int_labels, labels))
	print("* Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(int_labels, labels))	  
	print("* Adjusted Mutual Information: %0.3f"  % metrics.adjusted_mutual_info_score(int_labels, labels))
	print("* Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
	
	plt.close('all')
	plt.figure(1)
	plt.clf()
	
	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = 'k'

		class_member_mask = (labels == k)

		xy = X[class_member_mask & core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=14)

		xy = X[class_member_mask & ~core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=6)

	plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters_)
#	plt.show()
	plt.savefig("dbscan.png")
	
def mean_shift(X,int_labels):	
	# The following bandwidth can be automatically detected using
	bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_

	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)

	print("* Number of classes: %d" %len(set(int_labels)))
	print('* Estimated number of clusters: %d' % n_clusters_)	
	print("* Homogeneity: %0.3f" % metrics.homogeneity_score(int_labels, labels))
	print("* Completeness: %0.3f" % metrics.completeness_score(int_labels, labels))
	print("* V-measure: %0.3f" % metrics.v_measure_score(int_labels, labels))
	print("* Adjusted Rand Index: %0.3f"  % metrics.adjusted_rand_score(int_labels, labels))	  
	print("* Adjusted Mutual Information: %0.3f"  % metrics.adjusted_mutual_info_score(int_labels, labels))
	print("* Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
	
	plt.close('all')
	plt.figure(1)
	plt.clf()

	colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	for k, col in zip(range(n_clusters_), colors):
		my_members = labels == k
		cluster_center = cluster_centers[k]
		plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
		plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=14)
	plt.title('Estimated number of Mean Shift clusters: %d' % n_clusters_)
#	plt.show()
	plt.savefig("mean_shift.png")

########################################
def run_1d_clustering():
	data = load_data("data_train.txt")
	lat_labels = load_labels("labels_train.txt")
	print (len(set(lat_labels)))
		
	##transform labels
	mlb = LabelBinarizer()
	bin_labels = mlb.fit_transform(lat_labels) 
	
	print ("initial data: ", np.array(data).shape)
	ress = run_svd(data)
	new_ress = reduce_data(ress)
	print ("reduced data: ", np.array(new_ress).shape)
	
	X = []
	for dat in new_ress:
		X.extend(dat)
	X = np.array(X)
	
	X_data = load_data("data_val.txt")
	lat_labels_test = load_labels("labels_val.txt")
	print ("initial data: ", np.array(X_data).shape)
	val_ress = run_svd(X_data)
	new_val_ress = reduce_data(val_ress)
	print ("reduced data: ", np.array(new_val_ress).shape)
	
	X_1= []
	for dat in new_val_ress:
		X_1.extend(dat)

	X_1.extend(X)
	ll = []
	for l in lat_labels:
		ll.append([l])
	lat_labels_test.extend(ll)
	mlb1 = MultiLabelBinarizer()
	bin_y =  mlb1.fit_transform(ll)
		
	print ("AFFINITY PROPAGAITION")
	# aff_prop(new_val_ress,lat_labels_test)
	# print ("==================")
	# print ("DBSCAN")
	# dbscan(new_ress,int_labels)
	# print ("==================")
	# print ("MEAN SHIFT")
	# mean_shift(new_ress,int_labels)
	

	
###########################	
# python clustering.py 
#########################	##

def main():

	#run_1d_clustering()
	
	train_data = load_data("data_train.txt")
	lat_labels_train = load_labels("labels_train.txt")
	val_data = load_data("data_val.txt")
	lat_labels_val = load_labels("labels_val.txt")
	
	X = []
	X.extend(val_data)
	X.extend(train_data)

	ll = []
	for l in lat_labels_train:
		ll.append([l])
	lat_labels_val.extend(ll)
	mlb1 = MultiLabelBinarizer()
	y =  mlb1.fit_transform(lat_labels_val)
		
	print ("AFFINITY PROPAGAITION")
	aff_prop_nd(X,lat_labels_val)
	# print ("==================")
	# print ("DBSCAN")
	# dbscan(new_ress,int_labels)
	# print ("==================")
	# print ("MEAN SHIFT")
	# mean_shift(new_ress,int_labels)
	
if __name__ == "__main__":
	main()		