#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is part of the second assignment in class Methods in Bioinformatics and refers to
Clustering alorithms & in particular the K-means and Mixture of Gaussians.

This particular script contains the implementation of K-Mmeans algorithm.
algorithm.

"""

from functions import*


def K_means(Arr,k,Distance,n_iter,n_reps):

	start = time.time()

	N = Arr.shape[1] #;print(N)
	D = Arr.shape[0] #;print(D)

	Centroids = Arr[:,np.random.choice(range(N),k,replace=False)] #;print(Centroids) #; print(Centroids[:,0])

	# --- Plot of data set with the initial centroids
	Plot_Cen(Arr,Centroids)

	# --- Dictionary in which we will keep the labels (the cluster in which every input points belongs)
	labels = {}
	Dist_old = math.inf
	scores = []
	
	# We build in a failsafe mechanism: do at most 'n_iter' iterations, if convergence is not achieved.
	# If convergence is achieved, we break the loop manually.

	for i in range(n_reps):
		print("\nIn {} replication: \n".format(i))

		# Randomly choosing initial centroids  from the sample
		Centroids = Arr[:,np.random.choice(range(N),k,replace=False)] #;print(Centroids) #; print(Centroids[:,0])
		
		iterations = 0
		counter = 0		# setting a counter to check the number of iterations needed before convergence achieved.
		while iterations < n_iter:		
			iterations += 1
			counter +=1
			
			# partitions contains the partitioned data (it's a list of k lists, each one corresponding to a cluster)
			partitions = []
			# initialized with k empty lists
			for i in range(k):
				partitions.append([])
			
			# Iterate for each input point: we have to find to which cluster it belongs
			for i in range(N):			
				min_dist = dist_function(Arr,Distance,Arr[:,i], Centroids[:,0]) 		# initially, suppose that the minimum distance is achieved for the first cluster
				idx = 0																	# keep the index of the cluster (we need it)
				for cluster in range(1,k):												# check all clusters from 1 to k
					td = dist_function(Arr,Distance,Arr[:,i], Centroids[:,cluster]) 	# compute the distance of the point to the centroid 
					if min_dist > td:													# if distance is less than the (current) minimum...
						min_dist = td													# ...update the current minimum...
						idx = cluster													# ...ans also the index at which we found it.


				# At this point, min_dist contains the minimum distance, and idx is the index of the cluster for which we achieved it
				# min_dist = min { dist(Arr[:,i], Centroids[:,j]) }, 0 <= j <= k, idx is the j for which we achieve that
				partitions[idx].append(Arr[:,i].tolist())

				# idx is also the label that we assign to the point i
				labels[i] = idx;

			# --- Compute new centroids
			NCentroids = np.empty((D,k))
			for i in range(len(partitions)):
				NCentroids[:,i] = np.mean(np.vstack([partitions[i]]))

			# --- If the new centroids are too close to the old ones, there is no need to continue.
		
			dist =0
			for cluster in range(k):
				dist += dist_function(Arr,Distance,Centroids[:,cluster],NCentroids[:,cluster])
			
			if dist > 10**(-10):
				Centroids = NCentroids
			else:
				print("Convergence achieved!")
				break

		# --- End of maximum iterations ,achieving convergence or not.
		#print(Centroids)
		print('Number of iterations until convergence achieved: ' + str(counter))
		
		# --- Plot of the Clusters at each replication

		# colors = cm.rainbow(np.linspace(0, 1, k))
		# label_added = False
		# for i in range(k):
		# 	data = partitions[i]
		# 	if len(data) > 0:
		# 		ddata = np.vstack(data).T
		# 		plt.scatter(ddata[0,:], ddata[1,:], color=colors[i], s=50,alpha=0.5)
			
		# 	if not label_added:
		# 		plt.scatter(Centroids[0,i], Centroids[1,i], color='black', alpha=0.5, marker=(5,0), s=150,label='Centroids')
		# 		label_added = True
		# 	else:
		# 		plt.scatter(Centroids[0,i], Centroids[1,i], color='black', alpha=0.5,marker=(5,0),s=150)
		
		# plt.grid(True)
		# plt.legend( scatterpoints =1 ,loc='lower right')
		# plt.savefig("Final Clusters")
		# plt.show()
		# plt.close()
		
		
		# --- Silhouette score at each replication to value how good the clustering is:
		l = np.array(list(labels.values()))
		s=Silhouette(Arr.T,l,Distance)
		scores.append(s)


		# --- We calculate the distances inside each of the clusters at every replication
		# if the sum of the distances is smaller, we keep the centroids and the labels
		# calculated at the specific replication this achieved.##
		Dist_new = 0
		for i in range(k):
			data = partitions[i]
			if len(data) > 0:
				ddata = distance(Distance,(np.vstack(data).T))
			Dist_new += ddata
		if (Dist_new < Dist_old) or (math.isnan(Dist_new)):
			Labels = labels.copy()
			Centroids_F = Centroids.copy()

		Dist_old = Dist_new

	# --- End of replications. we take the final Labels and Centroids and a final plot of the clusters.
	
	end = time.time()

	y = np.array(list(Labels.values()))		# Labels
	Plot_Clusters(Arr,y,Centroids_F,k)		# Plot

	# --- Silhouette score for the final clustering (using the final labels)
	f = Silhouette(Arr.T,y,Distance)

	

	elapsed = (end - start)
	print("Time {}".format(elapsed))
	print("Scores at each replication: {}".format(scores))
	print("K Means....Final Silhouette score: {} with distance metric: {}".format(f,Distance))
	

	return(Centroids_F,Labels)


# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----

