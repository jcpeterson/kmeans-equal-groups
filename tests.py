from kmeans_equal_groups import *
import matplotlib.pyplot as plt
from numpy.random import rand

def plot(tup):
	plt.cla()
	color = ['red', 'green', 'blue', "purple"]
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

board = init_board_gauss(223, 8)
result = kmeans_equal_groups(board, 4)
sizes = [len(result[1][k]) for k in result[1].keys()]
assert max(sizes) - min(sizes) < 2
plot(kmeans_equal_groups(board, 4))






