import numpy as np
import matplotlib.pyplot as plt

# for coloring points and clusters
colorMap = plt.get_cmap('Accent')

# read points from file
inFile = open('filename', 'r')
N = int(inFile.readline())
points = np.zeros([N, 2])
for i in range(N):
    pt = inFile.readline()
    pt = pt.split()
    points[i, 0] = float(pt[0]) #x
    points[i, 1] = float(pt[1]) #y

K = 3

# generate random colors, one for each cluster
clusterColor = colorMap(np.array(range(K))/K)

# initialize distances from each point to corresponding centroid
dist = np.zeros(N)

# ids of clusters for each point: 
# clusterID[i] = -1 means point i doesn't belong to any cluster
# clusterID[i] = j means point i belongs to cluster j, j=[0..K-1]
clusterID = np.full(N, -1)

#randomly assign points to clusters
clusterID = np.random.randint(0, K, N)

centroids = np.zeros([K, 2]) # positions of centroids

#kmeans iterations

nrMaxIterations = 10

for iter in range(nrMaxIterations):
    pass #remove this line

    ### compute new centroids
    
    # centroids[j] = average of points i with clusterID[i] == j      
   
    # break when centroid positions don't change significantly from previous values
    
    ### assign points to clusters
     
    # for any point i, clusterID[i] = j, where j is the index of the centroid closest to i,  i = [0..N-1] , j =[0..K-1]

#end for

# plot points, centroids
pointColors = np.array(clusterColor[clusterID])
plt.scatter(points[:,0], points[:,1], color = pointColors, marker = 'o', s = 10)
centroidColors = np.array(clusterColor[range(K)])
plt.scatter(centroids[:, 0], centroids[:, 1], color = centroidColors, marker = 'x', s = 100)
plt.show()    







