# Import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing
from matplotlib import pyplot as plt #plotting figures

df = pd.read_csv("./dataset/Iris.csv") #load the dataset
x = df.iloc[:, [1, 2, 3, 4]].values #drop categorical values

#Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
fig=plt.figure()

#Plotting the results onto a line graph, observe 'the elbow'
plt.plot(range(1, 11), wcss)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Within cluster sum of squares') 

plt.draw()
plt.savefig('./output/iris_kmeans3_elbowplot.png', dpi=300, bbox_inches='tight')

#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

fig=plt.figure()

#Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 30, c = 'orange', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 30, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 30, c = 'green', label = 'Iris-virginica')

#Plotting the centers of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Cluster Centers',marker='*')

plt.legend()

plt.draw()
plt.savefig('./output/iris_kmeans3_predictions.png', dpi=300, bbox_inches='tight')

import csv

with open('./output/iris_kmeans3_predictions.csv', 'w') as f:
    writer = csv.writer(f)
    for val in y_kmeans:
        writer.writerow([val])
