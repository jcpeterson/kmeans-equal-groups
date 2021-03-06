
import matplotlib.pyplot as plt
import numpy as np
import random


def plot(tup):
	# plots the output in the format of k_means_equal_groups 
	# the centroids for each cluster are plotted as diamonds of the color of the cluster
	# each cluster is plotted in different color, supports up to 8 colors
	print "plot is a test function that supports 8 colors. if plotting >8 clusters, please consider using a different implementation"
	plt.cla()
	color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w' ]
	i=0
	for key in tup[1]:
		x=[]
		y=[]
		for member in tup[1][key]:
			x.append(member[0])
			y.append(member[1])
		plt.scatter(x,y,color=color[i])
		plt.scatter([tup[0][key][0]],[tup[0][key][1]], marker='D',color=color[i])
		i+=1
	plt.show()


def init_board(N):
	# generates random 2D points 
	X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
	return X

def init_board_gauss(N, k):
	# generates random 2D points with gaussian noise
	n = float(N)/k
	X = []
	for i in range(k):
		c = (random.uniform(-1, 1), random.uniform(-1, 1))
		s = random.uniform(0.05,0.5)
		x = []
		while len(x) < n:
			a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
			# Continue drawing points from the distribution in the range [-1,1]
			if abs(a) < 1 and abs(b) < 1:
				x.append([a,b])
		X.extend(x)
	X = np.array(X)[:N]
	return X

##################







