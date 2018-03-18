#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is part of the second assignment in class Methods in Bioinformatics and refers to
Clustering alorithms & in particular the K-means and Mixture of Gaussians.

This particular script contains functions needed from
the core algorihms.

"""


#==============================================================================
#   Libraries:
#==============================================================================


import sys
import time
import math
from math import sqrt
import numpy as np
import scipy.spatial.distance
import sklearn.metrics
from scipy import linalg
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib.patches import Ellipse



#==============================================================================
#  Function returning the Silhouette Score (details in report):
#==============================================================================


def Silhouette(Arr,lab,dist):
	if dist == 'E':
		return(sklearn.metrics.silhouette_score(Arr, lab, metric='euclidean'))
	elif dist == 'M':
		return(sklearn.metrics.silhouette_score(Arr, lab, metric='cityblock'))
	else:
		return(sklearn.metrics.silhouette_score(Arr, lab, metric='mahalanobis'))


#==============================================================================
#  Function that constructs the data set for Theoretical Part:
#==============================================================================



def Data(N1,N2,m1,m2,D):

	# --- epsilon = np.random.uniform(-1,1,D)

	mean_01 = m1*np.ones(D) #;print(mean_01.shape)   # mu: Dx1 cov: DxD
	mean_02 = m2*np.ones(D) #;print(mean_02.shape)

	cov_01	= 0.5*np.eye(D) #+ epsilon
	cov_02	= 0.75*np.eye(D) #+ epsilon

	X_01 = np.random.multivariate_normal(mean_01,cov_01,N1).T
	X_02 = np.random.multivariate_normal(mean_02,cov_02,N2).T

	X = np.concatenate((X_01,X_02),axis=1)	#; print(X.shape)  ## should be DxN

	### Initial Data Plot:

	plt.scatter(X[0,:], X[1,:], color='blue',marker='o',alpha=0.5,s=50,label='Data')
	plt.grid(True)
	plt.legend( scatterpoints =1 ,loc='lower right')
	plt.savefig("Initial plot")
	plt.show()

	return(X)


#==============================================================================
#  Basic Metric Distances Functions:
#==============================================================================


def euclidean_distance(x, y):
	return sqrt(np.sum((x-y)**2))

def Manhattan_Distance(x,y):
	return np.sum(np.abs(x-y))

def Mahalanobis_Distance(Arr,x,y):
	D = Arr.shape[0]
	S = np.cov(Arr) 
	return np.sqrt((x-y).reshape(1,D).dot(np.linalg.inv(S)).dot((x-y).reshape(D,1)))

# --- Change distance function here according to needs

def dist_function(Arr,Distance,x, y):
	if Distance == 'E':
		return euclidean_distance(x, y)
	elif Distance == 'M':
		return Manhattan_Distance(x,y)
	else:
		return Mahalanobis_Distance(Arr,x,y)


# --- using built in functions for practical reasons in specific stages of the algorithm.

def distance(Distance,Arr):
	if Distance == 'E':
		return scipy.spatial.distance.pdist(Arr,metric='euclidean')
	elif Distance == 'M':
		return scipy.spatial.distance.pdist(Arr,metric='cityblock')
	else:
		S = np.linalg.inv(np.cov(Arr))
		return scipy.spatial.distance.pdist(Arr,metric='mahalanobis',VI = S)


#==============================================================================
#  Plots:
#==============================================================================



# --- Adding Ellipses in plots....Credits to: Dimitris Kyriakis :)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))


    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,lw=2,fill=False,ls='--')

    ax.add_artist(ellip)
    return ellip
    # Width and height are "full" widths, not radius
    # for i in range(nstd):
    # 	s = i+1
    # 	width, height = 2 * s * np.sqrt(vals)
    # 	ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,lw=2, fill=False,ls='--')
    # 	ax.add_artist(ellip)
    # return ellip



def Plot_Cen(Arr,Cen):
	plt.scatter(Arr[0,:], Arr[1,:], color='blue',marker='o',alpha=0.5,s=50,label='Data Set')
	plt.scatter(Cen[0,:],Cen[1,:],color='red', marker=(5,0),alpha=0.5,s=150,label='Centroids')
	plt.grid(True)
	plt.legend( scatterpoints =1 ,loc='lower right')
	plt.savefig("Initial Plot of Centroids")
	plt.show()


# --- Plot of final clusters, K means

def Plot_Clusters(Arr,lab,Cen,k):		

	N = Arr.shape[1]
	colors = cm.rainbow(np.linspace(0, 1, k))
	label_added = [False]*k
	
	for i,j in zip(range(N),lab):
		if not label_added[j]:
			plt.scatter(Arr[0,i], Arr[1,i], color= colors[j],s=50,alpha=0.5,label='Cluster {}: {}'.format(j+1,np.count_nonzero(lab == j)))
			label_added[j] = True
		else:
			plt.scatter(Arr[0,i], Arr[1,i], color= colors[j],s=50,alpha=0.5)
	
	label_added = False
	for i in range(k):
		if not label_added:
			plt.scatter(Cen[0,i], Cen[1,i], color='black', alpha=0.5, marker=(5,0), s=150,label='Centroids')
			label_added = True
		else:
			plt.scatter(Cen[0,i], Cen[1,i], color='black', alpha=0.5,marker=(5,0),s=150)	

	plt.grid(True)
	plt.legend( scatterpoints =1 ,loc='lower right')
	#plt.title("Final Clusters")
	plt.savefig("K means Final Clusters")
	plt.show()
	plt.close()




# --- Plot based on Probabilities ,MoG

def Plot_Prob(Arr,mu,gamma,cluster,k):		

	N = Arr.shape[1]
	colors = cm.rainbow(np.linspace(0, 1, k))
		
	label_added = False
	for i in range(k):	
		if not label_added:
			plt.scatter(mu[0,i], mu[1,i], color='black', alpha=0.5, marker=(5,0), s=150,label='Centroids')
			label_added = True
		else:
			plt.scatter(mu[0,i], mu[1,i], color='black', alpha=0.5,marker=(5,0),s=150)

	for i in range(N):
		col = np.multiply(gamma[0,i],colors[0])
		for j in range(1,k):
			col = col + np.multiply(gamma[j,i],colors[j])

		plt.scatter(Arr[0,i], Arr[1,i], color=np.asarray(col.astype(np.float32)),s=50, alpha=0.5)
	

	for i in range(k):
		plot_cov_ellipse(np.cov(cluster[i]), mu[:,i], nstd=2, ax=None)


	plt.grid(True)
	plt.legend( scatterpoints =1 ,loc='lower right')
	plt.savefig("MoG Final Clusters")
	plt.show()
	plt.close()


# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
