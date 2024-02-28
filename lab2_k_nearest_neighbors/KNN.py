import numpy as np
import matplotlib.pyplot as plt

## read points from file
inFile = open('knnpoints.txt', 'r')
N = int(inFile.readline())
dataset = np.zeros([N, 3])
for i in range(N):
    pt = inFile.readline()
    pt = pt.split()
    dataset[i, 0] = float(pt[0]) #x
    dataset[i, 1] = float(pt[1]) #y
    dataset[i, 2] = float(pt[2]) #class label

#class labels are 0 - red , 1 - green

np.random.shuffle(dataset)
points = dataset[:, :2]
labels = dataset[:, 2]

K = 3

#implement knn:

#for a point x, y, find the closest K points from the dataset

# determine class label of new point: 
# in the non weighted version of KNN, it is the class of the majority of the closest neighbors
# in the weighted version, it is the class for which the sum of the corresponding neighbor weights is the highest 

labelColors = ['red', 'green']
unlabeledColor = 'black'


pointColors = [labelColors[int(labels[i])] for i in range(N)]
plt.scatter(points[:,0], points[:,1], color = pointColors)

#points to classify
myPoints = np.array([[6.2, 2.5], [5.37, 3.6], [4.65, 2.23]])

#find labels of myPoints ...

plt.scatter(myPoints[:,0], myPoints[:,1], color = unlabeledColor, s = 60, marker='x')

plt.show()