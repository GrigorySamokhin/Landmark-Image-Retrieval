import os
import cv2
import itertools
import numpy as np
import pickle
from sklearn.cluster import KMeans
from collections import Counter

DIR = 'metadata'

class VladPrediction(object):
	def __init__(self, dataset, path, query):
		self.dataset = dataset
		self.name = DIR + os.path.sep + 'clusters_and_vlad_descriptors.pickle'
		self.path_to_dataset = path
		self.query_path = query

	def describe_SIFT(self, image, nfeatures=1000):
		'''
		:param image: image path
		:param nfeatures: Number of key-points

		Calculate SIFT descriptor with nfeatures key-points
		'''

		# Converting from BGR to GRAY
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Compute key-points and descriptors for each key-points
		# nfeatures = number of key-points for each image
		sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
		keypoints, descriptors = sift.detectAndCompute(image, None)
		return descriptors

	def get_SIFT_descriptors(self, dataset):
		'''
		:param dataset: Object of DataSet

		Calculate set of SIFT descriptors of all images in dataset
		'''
		descriptors = []

		# Get DataFrame with images paths
		df_data = dataset.get_data()

		# Compute SIFT descriptors
		print('Start computing descriptors using SIFT. ', end='')
		for image_temp in df_data.itertuples():
			im = cv2.imread(image_temp.image_path)
			des = self.describe_SIFT(im)
			if des.all() != None:
				descriptors.append(des)

		# Union all descriptors for each key-point to list
		descriptors = np.array(list(itertools.chain.from_iterable(descriptors)))
		print('Compete.')
		return descriptors

	def get_clasters(self, descriptors, k):
		'''
		:param descriptors: Set of all SIFT descriptors in dataset
		:param k: number of clusters to compute

		Get k number of clusters
		'''
		print('Start clustering descriptors. ', end='')

		# Compute k clusters
		clasters = KMeans(n_clusters=k).fit(descriptors)
		print('Compete.')
		return clasters

	def compute_vlad_descriptor(self, descriptor, kmeans_clusters):
		'''
		:param descriptor: SIFT descriptor of image
		:param kmeans_clusters: Object of Kmeans (sklearn)

		First we need to predict clusters fot key-points of image (row in
		input descriptor). Then for each cluster we get descriptors, which belong to it,
		and calculate sum of residuals between descriptor and centroid (cluster center)
		'''
		# Get SIFT dimension (default: 128)
		sift_dim = descriptor.shape[1]

		# Predict clusters for each key-point of image
		labels_pred = kmeans_clusters.predict(descriptor)

		# Get centers fot each cluster and number of clusters
		centers_cluster = kmeans_clusters.cluster_centers_
		numb_cluster = kmeans_clusters.n_clusters
		vlad_descriptors = np.zeros([numb_cluster, sift_dim])

		# Compute the sum of residuals (for belonging x for cluster) for each cluster
		for i in range(numb_cluster):
			if np.sum(labels_pred == i) > 0:

				# Get descritors which belongs to cluster and compute residuals between x and centroids
				x_belongs_cluster = descriptor[labels_pred == i, :]
				vlad_descriptors[i] = np.sum(x_belongs_cluster - centers_cluster[i], axis=0)

		# Create vector from matrix
		vlad_descriptors = vlad_descriptors.flatten()

		# Power and L2 normalization
		vlad_descriptors = np.sign(vlad_descriptors) * (np.abs(vlad_descriptors)**(0.5))
		vlad_descriptors = vlad_descriptors / np.sqrt(vlad_descriptors @ vlad_descriptors)
		return vlad_descriptors

	def get_vlad_descriptors(self, kmeans_clusters, dataset):
		'''
		:param kmeans_clusters: Object of Kmeans (sklearn)
		:param dataset: Object of DataSet

		Calculate VLAD descriptors for dataset
		'''
		vlad_descriptors = []

		# Get DataFrame with paths classes fro images
		df_data = dataset.get_data()

		print('Start computing vlad vectors. ', end='')
		for image_temp in df_data.itertuples():

			# Compute SIFT descriptors
			im = cv2.imread(image_temp.image_path)
			descriptor = self.describe_SIFT(im)
			if descriptor.all() != None:

				# Compute VLAD descriptors
				vlad_descriptor = self.compute_vlad_descriptor(descriptor, kmeans_clusters)
				vlad_descriptors.append({'image_path': image_temp.image_path,
										'class_image': image_temp.class_image,
										'feature_vector': vlad_descriptor
				})
		print('Complete. ', end='')
		return vlad_descriptors

	def get_prediction(self, kmeans_clusters, vlad_descriptors, query=None, mode=0):
		'''
		:param kmeans_clusters: Object of Kmeans (sklearn)
		:param vlad_descriptors: Set of VLAD descriptions

		Calculate VLAD vector for query image, and then
		get best distances in dataset.
		'''
		list_res = []
		if query == None:
			query = self.query_path
		# compute SIFT descriptor for query
		im = cv2.imread(query)
		descriptor = self.describe_SIFT(im)

		# compute VLAD descriptor for query
		v = self.compute_vlad_descriptor(descriptor, kmeans_clusters)

		# Get distances between query VLAD and dataset VLADs descriptors
		for i in range(len(vlad_descriptors)):
			temp_vec = vlad_descriptors[i]['feature_vector']
			dist = np.linalg.norm(temp_vec - v)
			list_res.append({'i': i,
							'dist': dist,
							'class': vlad_descriptors[i]['class_image'],
							 'image_path':vlad_descriptors[i]['image_path']
							 })
		res_ = sorted(list_res, key=lambda x: x['dist'])

		# Get most frequent class in (3) first classes
		res_count = Counter([x['class'] for x in res_][:3])
		res = min(res_count.items(), key=lambda x: (-x[1], x[0]))[0]

		if mode == 1:
			return res

		if mode == 2:
			return res_[:3]

		print('\nPredicted class for query image: {}.'.format(res))


	def get_clusters_vlad_descriptors(self, dataset, k=64):
		'''
		:param dataset: main dataset
		:param k: number os clusters to determine (default=64)

		Load computed clusters and vectors or compute SIFT descriptors, then
		compute clusters for these descriptors, then calculate VLAD descriptors
		and save.
		'''
		# if we have computed clusters and vlad vectors
		if os.path.exists(self.name):
			with open(self.name, 'rb') as file:
				kmeans_clusters, vlad_descriptors = pickle.load(file)
		else:
			# Get list of all SIFT descriptors
			descriptors = self.get_SIFT_descriptors(dataset)

			# Get Kmeans object with k clusters
			kmeans_clusters = self.get_clasters(descriptors, k)

			# Compute VLAD descriptors
			vlad_descriptors = self.get_vlad_descriptors(kmeans_clusters, dataset)

			# Save results
			with open(self.name, 'wb') as file:
				pickle.dump([kmeans_clusters, vlad_descriptors], file)
		return kmeans_clusters, vlad_descriptors

	def vlad_prediction(self, dataset):
		'''
		:param dataset: DataSet type of main dataset

		Compute clusters for SIFT descriptors and VLAD vectors
		'''
		# Get SIFT descriptors, clusters, VLAD vectors
		kmeans_clusters, vlad_descriptors = self.get_clusters_vlad_descriptors(dataset)

		# Get prediction
		self.get_prediction(kmeans_clusters, vlad_descriptors)
		return vlad_descriptors



