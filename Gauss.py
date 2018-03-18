#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is part of the second assignment in class Methods in Bioinformatics and refers to
Clustering alorithms & in particular the K-means and Mixture of Gaussians.

This particular script contains the implementation of Mixture of Gaussians 
algorithm.

"""


from functions import*
from KMeans import*


# --- Gaussian Density Function:

def N_gauss(x,y,S,D):
	# N = Arr.shape[1]
	# D = Arr.shape[0]

	a_01 = np.sqrt(((2*np.pi)**D)*np.linalg.det(S))
	a_02 = (((x-y).reshape(1,D)).dot(np.linalg.inv(S))).dot((x-y).reshape(D,1))

	return (1/a_01)*np.exp((-1/2)*a_02)

# --- Mixture of Gaussians Algorithm


def Mixture_of_Gaussians(Arr,k,Distance,n_iter,n_reps):

	start = time.time()

	N = Arr.shape[1] #;print(N)
	D = Arr.shape[0] #;print(D)
	
	# --- We have the option to perform K Means first, so as to take initial Centroids

	q_01 = input("\nRun K_Means first to take initial Centroids, Y/n: ")

	# if q_01 == 'Y':
	# 	Distance = input("\nChoose distance. Enter E for Euclidean, M for Manhattan and H for Mahalanobis: ")
	
	scores=[]
	Dist_old = math.inf

	# We build in a failsafe mechanism: do at most 'n_iter' iterations, if convergence is not achieved.
	# If convergence is achieved, we break the loop manually.

	for i in range(n_reps):
		print("\nIn {} replication: \n".format(i))
		
		if q_01 == 'Y':
			#Distance = input("\nChoose distance. Enter E for Euclidean, M for Manhattan and H for Mahalanobis: ")
			Centroids = K_means(Arr,k,Distance,n_iter,n_reps)
		else:
			Centroids = Arr[:,np.random.choice(range(N),k,replace=False)]
		
		### Initializing Parameters:

		π_old = np.random.uniform(size=k)
		π_old /= π_old.sum()
	
		μ_old =Arr[:,np.random.choice(range(N),k,replace=False)]
		#μ_old = Centroids		## Typically the means of each Gaussian are clusters' centroids.
		#print(μ_old.shape)

		Nk = np.zeros(k)
		
		Σk_old = np.zeros(k,dtype=object)
		for cluster in range(k):						
			Σk_old[cluster] = np.eye((D))
		
		Pz = np.zeros(k)		## Pz : array of k values ,each for every Gaussian
		γ_nk_new = np.zeros((k,N))

		L_old = 0
		Diff_Log = 1
		#L_new = 0
		
		
		
		iterations = 0
		counter = 0
		
		while iterations < n_iter:
			iterations += 1
			counter +=1

			L_new =0
			Athr =0 
			γ_nk = np.zeros((k,N)) 				# gama_nk a kxN array
			
			μ_new  = np.zeros((D,k))			# mean - centroids : Dxk
			Σk_new  = np.zeros(k,dtype=object)	# Sk : k objects /arrays of DxD dimension
			π_new = np.zeros(k)
			
			Pz_new=np.zeros(k)
		

			##.................. E step..................##
			for i in range(N):
				for cluster in range(k):
					Pz[cluster] = π_old[cluster]*N_gauss(Arr[:,i],μ_old[:,cluster],Σk_old[cluster],D)
				s = np.sum(Pz)
				γ_nk[:,i] = Pz/s


			Nk_new = np.sum(γ_nk,axis=1)
	

			##.................. M step..................##
			
			# μ_new
			for cluster in range(k):
				for i in range(N):
					μ_new[:,cluster] += (1/Nk_new[cluster])*γ_nk[cluster,i]*Arr[:,i]
			# print(μ_new)

			# Σk_new
			for cluster in range(k):
				Σk_new[cluster] = np.zeros((D,D))

			for cluster in range(k):
				for i in range(N):
					product = (((Arr[:,i] - μ_new[:,cluster]).reshape(D,1)).dot((Arr[:,i] - μ_new[:,cluster]).reshape(1,D)))
					Σk_new[cluster] += ((1/Nk_new[cluster])*(γ_nk[cluster,i]))*product
			# print(Σk_new)


			# π_new
			for cluster in range(k):
				π_new[cluster] = Nk_new[cluster]/N
			# print(π_new)

			##.............. Log-Likelihood................##

			for i in range(N):
				Athr =0 
				for cluster in range(k):
					Athr += π_new[cluster]*N_gauss(Arr[:,i],μ_new[:,cluster],Σk_new[cluster],D)
				L_new += math.log(Athr)
			
			Diff_Log = (abs(L_old-L_new))**2  		
			
			sys.stdout.write("\rDiff_Log:  {}".format(Diff_Log))
			sys.stdout.flush()
			
			L_old = L_new
			# print(Diff_Log)
			if Diff_Log > 10**(-5):			## Convergence criteria
				μ_old = μ_new.copy()
				Σk_old= Σk_new.copy()
				π_old = π_new.copy()
				Pz = Pz_new.copy()
				Nk = Nk_new.copy()
			
			else:
				print("\n\nConvergence achieved!")
				break
		
		print('\n\nNumber of iterations until convergence achieved: ' + str(counter))
		
	
		# --- End of maximum iterations ,achieving convergence or not.

		# --- Calculating the responsibilities after each replication:
		for i in range(N):
			for cluster in range(k):
				Pz[cluster] = π_old[cluster]*N_gauss(Arr[:,i],μ_old[:,cluster],Σk_old[cluster],D)
			s = np.sum(Pz)
			γ_nk_new[:,i] = Pz/s
		#print("density:{}\n".format(γ_nk_new))

		y = np.argmax(γ_nk_new,axis=0)			## Labels from the responsibilities array
		
		Clusters = np.zeros(k,dtype=object) 	## Forming the clusters using the labels 

		for i in range(k):
			Clusters[i]	= np.zeros((D,int(Nk[i])))

		for i in range(k):
			Clusters[i] = Arr[:,y== i]
		

		# --- Plot of responsibilities at each replication:

		Plot_Prob(Arr,μ_old,γ_nk_new,Clusters,k)


		s=Silhouette(Arr.T,y,Distance)
		scores.append(s)

		Dist_new = 0
		for i in range(k):
			#if Clusters[i].size() >2:
			Dist = distance(Distance,Clusters[i])
			Dist_new += Dist


		if (Dist_new < Dist_old) or (math.isnan(Dist_new)):
			Labels = y.copy()
			Centroids_F = μ_old.copy()
			Clusters_F  = Clusters.copy()
			Respons	= γ_nk_new.copy()

		Dist_old = Dist_new

	# --- End of replications. we take the final Labels,Centroids,Clusters and responsibilities
	# --- and a final plot of the clusters.

	end = time.time()

	# --- Final plot using Labels (discrete clusters)

	colors = cm.rainbow(np.linspace(0, 1, k))
	label_added = [False]*k
	for i,j in zip(range(N),Labels):
		if not label_added[j]:
			plt.scatter(Arr[0,i], Arr[1,i], color= colors[j],s=50,alpha=0.5,label='Cluster {}: {}'.format(j+1,np.count_nonzero(Labels == j)))
			label_added[j] = True
		else:
			plt.scatter(Arr[0,i], Arr[1,i], color= colors[j],s=50,alpha=0.5)

	for i in range(k):
			plot_cov_ellipse(np.cov(Clusters_F[i]), Centroids_F[:,i], nstd=2, ax=None)
	
	label_added = False
	for i in range(k):
		if not label_added:
			plt.scatter(Centroids_F[0,i], Centroids_F[1,i], color='black', alpha=0.5, marker=(5,0), s=150,label='Centroids')
			label_added = True
		else:
			plt.scatter(Centroids_F[0,i], Centroids_F[1,i], color='black', alpha=0.5,marker=(5,0),s=150)	
	

	plt.grid(True)
	plt.legend( scatterpoints =1 ,loc='lower right')
	#plt.title("Final Clusters_MoG")
	plt.savefig("MoG Labels Final Clusters")
	plt.show()
	plt.close()
	

	# --- Final plot using final responsibilities:
	Plot_Prob(Arr,Centroids_F,Respons,Clusters_F,k)

	# --- Silhouette score for the final clustering (using the final labels)
	f = Silhouette(Arr.T,Labels,Distance)

	elapsed = (end - start)
	print("Time {}".format(elapsed))
	print("Scores at each replication: {}".format(scores))
	print("Mixture of Gaussians....Final Silhouette score: {} with distance metric: {}".format(f,Distance))



	return(Centroids_F,Labels)

# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----

