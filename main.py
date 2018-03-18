#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is part of the second assignment in class Methods in Bioinformatics and refers to
Clustering alorithms & in particular the K-means and Mixture of Gaussians.

This particular script is the main call for the functions of:
Kmeans.py, Gauss.py, functions.py & Principal_Methods.py scripts 

"""


#==============================================================================
#   Libraries:
#==============================================================================

import time
import os
from functions import*
from KMeans import*
from Principal_Methods import*
from Gauss import*

#==============================================================================
#   Calls:
#==============================================================================


Problem = input("Choose either Theoretical or Practical Problem.\n Enter A or B for Theoretical or Practical respectively:")

if Problem == 'A':
	print("##........Theoretical Problem has been chosen.........##\n")
	
	Method = input("Choose Method: Enter K for K means Algorithm ,\n G for Mixture of Gaussians: ")

	Distance = input("Choose Distance Function\n E for Euklidean\n M for Manhattan\n H for Mahalanobis: ")
	n_iter = int(input("Choose number of iterations:  "))
	n_reps = int(input("Choose number of replications: "))

	# --- Constructing normally distributed data:  D =2 dimensions, N=500 observations (220 and 280). No labels.

	X = Data(220,280,1.0,-1.0,2)
	
	if Method == 'K':
		print("\n##........K means Algorithm........##\n")

		k = int(input("Choose number of k clusters: "))
		

		K_means(X,k,Distance,n_iter,n_reps)


	else:
		print("\n##........Mixture of Gaussians Algorithm........##\n")

		k = int(input("Choose number of k clusters: "))

		Mixture_of_Gaussians(X,k,Distance,n_iter,n_reps)


else:
	print("##........Practical Problem has been chosen.........##\n")


	print("\nData set should be downloaded automatically and the process shall begin.\n")


	if not os.path.exists("Final.txt"):

		Filename = os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz')

		os.system('gunzip <GDS6248.soft.gz> Data_set.txt')
		os.system('grep -i ILMN Data_set.txt > Data.txt')
		os.system('cut -f3- Data.txt > Final.txt')
	else:
		print('Skipping file download, Data file exists...')

	# --- Constructing the Data set Array:

	X = np.loadtxt("Final.txt")			
	print(X.shape)

	q_01 = input("\nPrincipal Component Analysis: Y or N: ")
	
	if q_01=='Y':
		print('Possible projection only in 2 Dimensions with Probabilistic or Kerne PCA.')
		y = [0]
		# Dimension = int(input("Give Dimensionality of Projection:"))
		M = 2
		q_02 = input("\nPlease enter EM for Probabilistic PCA or K for Kernel PCA: ")

		if q_02=='EM':
			Y = PPCA(X,y,M)
		else:
			Y = KERNEL(X,y,M)

		Method = input("Choose Method: Enter K for K means Algorithm ,\n G for Mixture of Gaussians: ")
		
		Distance = input("Choose Distance Function\n E for Euklidean\n M for Manhattan\n H for Mahalanobis: ")
		n_iter = int(input("Choose number of iterations:  "))
		n_reps = int(input("Choose number of replications: "))

		if Method == 'K':
			print("\n##........K means Algorithm........##\n")

			k = int(input("Choose number of k clusters: "))

			K_means(Y,k,Distance,n_iter,n_reps)


		else:
			print("\n##........Mixture of Gaussians Algorithm........##\n")

			k = int(input("Choose number of k clusters: "))

			Mixture_of_Gaussians(Y,k,Distance,n_iter,n_reps)

	else:
		Method = input("Choose Method: Enter K for K means Algorithm ,\n G for Mixture of Gaussians: ")

		Distance = input("Choose Distance Function\n E for Euklidean\n M for Manhattan\n H for Mahalanobis: ")
		n_iter = int(input("Choose number of iterations:  "))
		n_reps = int(input("Choose number of replications: "))

		if Method == 'K':
			print("\n##........K means Algorithm........##\n")

			k = int(input("Choose number of k clusters: "))
			

			K_means(X,k,Distance,n_iter,n_reps)


		else:
			print("\n##........Mixture of Gaussians Algorithm........##\n")

			k = int(input("Choose number of k clusters: "))

			Mixture_of_Gaussians(X,k,Distance,n_iter,n_reps)


# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
