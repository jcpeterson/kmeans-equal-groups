import numpy as np
import sys
import random

"""
1. Run k-means
2. Find closest exemplar from another cluster to the mean of the smallest cluster and allocate it to that cluster
3. Recompute mean for that cluster
4. Repeat 2-3 until clusters are balanced

"""
 
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)
 
def is_balanced(X, K, clusters):
    if K == 1:
        return True  

    balanced_size = len(X) / K
    if len(X) % K == 0:
        tolerance = 0 # can make perfectly balanced groups
    else:
        tolerance = 1 # can't make perfectly balanced groups,
                      # some groups will be off by 1 member
    if max([len(clusters[k]) for k in clusters]) - min([len(clusters[k]) for k in clusters]) > tolerance:
        return False
    return True


def balance_clusters(X, mu, clusters, steal=True):
    # steal=True makes the smaller cluster steal from a larger cluster
    # of size at least n+2 to avoid infinite loops
    K = len(mu)

    while not is_balanced(X, K, clusters):
        smallest_cluster_key = find_smallest_cluster(clusters)
        smallest_mu = mu[smallest_cluster_key]
        smallest_cluster_size = len(clusters[smallest_cluster_key])

        target = None
        min_dist = sys.maxint
        for k in clusters.keys():
            # find closest exemplar from another cluster to the mean of the smallest cluster 
            if k == smallest_cluster_key:
                continue 
            cluster = clusters[k]
            if steal and len(cluster) < smallest_cluster_size + 2:
                continue
            for i in range(len(cluster)):
                # keep track of cluster index and element index to delete element later
                dist = np.linalg.norm(cluster[i] - smallest_mu)
                if dist < min_dist:
                    target = (k, i)
                    min_dist = dist
        clusters[smallest_cluster_key].append(clusters[target[0]].pop(target[1]))
        mu[target[0]] = np.mean(clusters[target[0]], axis = 0)

    return (mu, clusters)


def find_smallest_cluster(clusters):
    # returns the key of the smallest cluster
    return min(clusters.keys(), key=lambda i: len(clusters[i]))

def kmeans_equal_groups(X, K, steal=True):
    k_means = find_centers(X, K)
    mu = k_means[0]
    clusters = k_means[1]
    balanced = balance_clusters(X, mu, clusters, steal)
    return balanced



# for testing purposes
def init_board(N):
    X = np.array([(random.uniform(-10, 10), random.uniform(-10, 10)) for i in range(N)])
    return X

def init_board_gauss(N, k):
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



